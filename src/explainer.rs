use std::collections::HashMap;

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use rand::Rng;
use serde::ser::SerializeSeq;

use crate::kiwi;

#[pyclass(module = "kiwi")]
pub struct Explainer {
    pub exp: kiwi::Explainer,
}

#[pymethods]
impl Explainer {
    #[getter]
    fn get_intercepts(&self) -> HashMap<usize, f64> {
        self.exp.get_intercepts()
    }

    #[getter]
    fn get_r2(&self) -> HashMap<usize, f64> {
        self.exp.get_r2s()
    }

    fn as_map(&self) -> HashMap<usize, Vec<(usize, f64)>> {
        let mut map = HashMap::new();

        for (label, coeffs) in self.exp.get_coeffs() {
            let mut coeffs = coeffs
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<(usize, f64)>>();

            coeffs.sort_by(|a: &(usize, f64), b: &(usize, f64)| {
                use std::cmp::Ordering::*;

                if !a.1.is_finite() {
                    return Less;
                }

                if !b.1.is_finite() {
                    return Greater;
                }

                if a.1.abs() < b.1.abs() {
                    Greater
                } else if a.1.abs() > b.1.abs() {
                    Less
                } else {
                    Equal
                }
            });

            map.insert(label, coeffs);
        }

        map
    }

    #[args(label = "1")]
    fn as_list(&self, label: usize) -> PyResult<Vec<(String, f64)>> {
        let vals = match self.as_map().get(&label) {
            Some(v) => v.to_owned(),
            None => return Err(PyValueError::new_err("invalid label")),
        };

        let mut list = vec![];

        for (feature, weight) in vals.iter() {
            let feature = *feature;
            let weight = *weight;

            let val = self.exp.row[feature];
            let name = if feature < self.exp.feature_names().len() {
                self.exp.feature_names()[feature].to_owned()
            } else {
                format!("{}", feature)
            };

            let feature_str = match self.exp.discretizer.get_bounds(feature, val)? {
                (Some(lower), Some(upper)) => {
                    format!("{} < {} <= {}", lower, name, upper)
                }
                (None, Some(upper)) => {
                    format!("{} <= {}", name, upper)
                }
                (Some(lower), None) => {
                    format!("{} > {}", name, lower)
                }
                (None, None) => {
                    format!("{}={}", name, val)
                }
            };

            list.push((feature_str, weight));
        }

        Ok(list)
    }

    #[args(labels = "None", predict_proba = "true")]
    fn show_in_notebook(
        &self,
        py: Python,
        labels: Option<Vec<usize>>,
        predict_proba: bool,
    ) -> PyResult<PyObject> {
        let module = PyModule::import(py, "IPython.core.display")?;

        let labels = labels.unwrap_or_else(|| (0..(self.exp.num_labels())).collect::<Vec<usize>>());

        let html = self.as_html(&labels, predict_proba)?;

        // Call HTML
        let obj = module.call1("HTML", (html,))?;

        // Call Display
        module.call1("display", (obj,)).map(|obj| obj.into_py(py))
    }
}

impl Explainer {
    fn as_html(&self, labels: &[usize], predict_proba: bool) -> PyResult<String> {
        let div_id: u64 = rand::thread_rng().gen();

        // Build the js
        let mut custom_js = format!(
            "
            var top_div = d3.select('#top_div{div_id}').classed('lime top_div', true);
            ",
            div_id = div_id,
        );

        let class_names_json = jsonize(&self.exp.class_names())?;

        if predict_proba {
            let predict_value_js = format!(
                "
                var pp_div = top_div.append('div').classed('lime predict_proba', true);
                var pp_svg = pp_div.append('svg').style('width', '100%%');
                var pp = new lime.PredictProba(pp_svg, {class_names_json}, {predict_proba_json});
                ",
                class_names_json = class_names_json,
                predict_proba_json = jsonize(&self.exp.proba())?,
            );

            custom_js.push_str(&predict_value_js);
        }

        // exp_js
        custom_js.push_str(&format!(
            "
            var exp_div;
            var exp = new lime.Explanation({class_names_json});
            ",
            class_names_json = class_names_json,
        ));

        for &l in labels.iter() {
            let exp_json = self.as_list(l).and_then(|l| jsonize(&l))?;

            custom_js.push_str(&format!(
                "
                exp_div = top_div.append('div').classed('lime explanation', true);
                exp.show({exp}, {label}, exp_div);
                ",
                exp = exp_json,
                label = l,
            ));
        }

        // data for the record
        let mut record_data = vec![];
        let feature_names = self.exp.feature_names();
        for (feature, weight) in &self.as_map()[&labels[0]] {
            let feature = *feature;
            let weight = *weight;

            record_data.push(RecordData {
                feature_name: &feature_names[feature],
                feature_value: format!("{}", self.exp.row[feature]),
                weight,
            });
        }

        custom_js.push_str(&format!(
            "
            var raw_div = top_div.append('div');
            exp.show_raw_tabular({record_data}, 0, raw_div);
            ",
            record_data = jsonize(&record_data)?,
        ));

        Ok(format!(
            "
            <html>
                <meta http-equiv=\"content-type\" content=\"text/html; charset=UTF8\">
                <head>
                    <script>{bundle_js}</script>
                </head>
                <body>
                    <div class=\"lime top_div\" id=\"top_div{div_id}\"></div>
                    <script>{custom_js}</script>
                </body>
            </html>
            ",
            bundle_js = include_str!("bundle.js"),
            div_id = div_id,
            custom_js = custom_js,
        ))
    }
}

struct RecordData<'a> {
    feature_name: &'a str,
    feature_value: String,
    weight: f64,
}

impl<'a> serde::ser::Serialize for RecordData<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(3))?;
        seq.serialize_element(self.feature_name)?;
        seq.serialize_element(&self.feature_value)?;
        seq.serialize_element(&self.weight)?;
        seq.end()
    }
}

fn jsonize<V: serde::Serialize>(v: &V) -> PyResult<String> {
    serde_json::to_string(v).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
