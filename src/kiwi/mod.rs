//!

use std::{collections::HashMap, error, fmt, io, result, sync::Arc};

mod tabularexplainer;
pub use tabularexplainer::{kernel_fn, ExplainerInstanceMem, TabularExplainer, ToComputeRecord};
mod discretizer;
pub use discretizer::Discretizer;

#[derive(Debug)]
pub enum KiwiError {
    LinAlg(ndarray_linalg::error::LinalgError),
    Io(io::Error),
    Custom(Arc<dyn error::Error + Send + Sync>),
    NonFiniteF64(f64),
    CategoricalNonInt(f64),
    WeightedError(rand::distributions::WeightedError),
    StatsError(statrs::StatsError),
    FeatureOutOfRange(usize),
    MutexError,
    SamplingThreadDied,
}

impl fmt::Display for KiwiError {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        use KiwiError::*;

        match self {
            LinAlg(e) => fmt::Display::fmt(e, w),
            Io(e) => fmt::Display::fmt(e, w),
            Custom(e) => fmt::Display::fmt(e, w),
            NonFiniteF64(f) => write!(w, "non-finite f64 [{}] invalid", f),
            CategoricalNonInt(f) => write!(w, "categorical feature value is not an int [{}]", f),
            WeightedError(ref e) => fmt::Display::fmt(e, w),
            StatsError(ref e) => fmt::Display::fmt(e, w),
            FeatureOutOfRange(u) => write!(w, "the feature index is out range [{}]", u),
            MutexError => write!(w, "couldn't lock mutex around background thread"),
            SamplingThreadDied => write!(w, "sampling thread died"),
        }
    }
}

pub type Result<T> = result::Result<T, KiwiError>;

pub struct LabelData {
    r2: f64,
    coeffs: Vec<f64>,
    intercept: f64,
    prediction: f64,
}

pub struct Explainer {
    tab_explainer: Arc<TabularExplainer>,
    pub row: Vec<f64>,

    pub discretizer: Arc<Discretizer>,

    label_data: HashMap<usize, LabelData>,
    highest_class_label: usize,
}

impl Explainer {
    fn new(tab_explainer: Arc<TabularExplainer>, row: Vec<f64>) -> Self {
        Self {
            discretizer: tab_explainer.discretizer.clone(),
            tab_explainer,
            row,
            label_data: HashMap::new(),
            highest_class_label: 0,
        }
    }

    fn add_class(
        &mut self,
        label: usize,
        r2: f64,
        coeffs: Vec<f64>,
        intercept: f64,
        prediction: f64,
    ) {
        self.label_data.insert(
            label,
            LabelData {
                r2,
                coeffs,
                intercept,
                prediction,
            },
        );

        if label > self.highest_class_label {
            self.highest_class_label = label;
        }
    }

    pub fn num_labels(&self) -> usize {
        self.label_data.len()
    }

    pub fn class_names(&self) -> Vec<String> {
        let class_names = self.tab_explainer.get_class_names();

        let mut names = vec![];

        if self.highest_class_label < class_names.len() {
            for (i, name) in (0..=self.highest_class_label).zip(class_names.iter()) {
                if self.label_data.contains_key(&i) {
                    names.push(name.to_owned());
                }
            }
        } else {
            for i in 0..=self.highest_class_label {
                if self.label_data.contains_key(&i) {
                    names.push(format!("{}", i));
                }
            }
        }

        names
    }

    pub fn get_coeffs(&self) -> HashMap<usize, &'_ [f64]> {
        let mut map = HashMap::new();

        for (&k, v) in self.label_data.iter() {
            map.insert(k, &v.coeffs[..]);
        }

        map
    }

    pub fn get_intercepts(&self) -> HashMap<usize, f64> {
        let mut map = HashMap::new();

        for (&k, v) in self.label_data.iter() {
            map.insert(k, v.intercept);
        }

        map
    }

    pub fn get_r2s(&self) -> HashMap<usize, f64> {
        let mut map = HashMap::new();

        for (&k, v) in self.label_data.iter() {
            map.insert(k, v.r2);
        }

        map
    }

    pub fn feature_names(&self) -> &'_ [String] {
        self.tab_explainer.get_feature_names()
    }

    pub fn proba(&self) -> Vec<f64> {
        self.label_data
            .iter()
            .map(|(_, v)| v.prediction)
            .collect::<Vec<f64>>()
    }
}
