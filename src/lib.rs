//! Python wrapper for the Rust impelmentation of Kiwi

#![allow(clippy::too_many_arguments)]

use std::{
    collections::HashMap,
    sync::{mpsc, Arc, Mutex},
    thread, time,
};

use ndarray::*;
use numpy::*;
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyList, PyString},
    PyErrArguments, PyIterProtocol,
};
use rand::SeedableRng;

mod explainer;
mod kiwi;
use explainer::Explainer;

#[pyclass]
// #[text_signature = "(training_data, *, class_names=None, categorical_features=None, discretizer=\"quartile\", feature_names=None, num_samples=5000, sample_background_thread=True)"]
struct KiwiTabularExplainer {
    tab_explainer: Arc<kiwi::TabularExplainer>,
}

#[pymethods]
impl KiwiTabularExplainer {
    /*
    KiwiTabularExplainer is the class used for explaining tabular data.

    Parameters
    ------------
    training_data : numpy.ndarray
        2D numpy array of float64

    class_names: None or list[Str], None
        List of strings to be used as class names
        This list must be the same length as the output of
        the predict_fn function called in explain_instance
        below, otherwise it is ignored.

    categorical_features: None or list[Int], None
        List of column indicies to be treated as categorical
        (i.e not discretized by the discretizer)

    discretizer: None or str, "quartile"
        type of discretizer, if None no column is discretized.
        The default is "quartile", which is currently the only
        supported discretizer

    feature_names: None or list[Str], None
        List of names of features (i.e columns).
        If given, the list must be the same length
        as the number of columns in the training_data,
        otherwise a ValueError is raised.

    default_num_samples: int, 5000
        A natural number corresponding to the number of
        samples to generate per record.

    sample_background_thread: bool, True
        If True then generate random samples in a background thread
        rather than dynamically when an explain_instance_* method is
        called. This gives performance benefits, at the cost of some
        overhead.
    */
    #[new]
    #[args(
        class_names = "None",
        categorical_features = "None",
        discretizer = "\"quartile\"",
        feature_names = "None",
        num_samples = "5000",
        sample_background_thread = "false"
    )]
    fn new(
        py_data: &PyArray2<f64>,
        class_names: Option<Vec<String>>,
        categorical_features: Option<Vec<usize>>,
        discretizer: Option<&str>,
        feature_names: Option<Vec<String>>,
        num_samples: usize,
        sample_background_thread: bool,
    ) -> PyResult<Self> {
        let training_data = unsafe { py_data.as_array() };
        let (_, num_features) = training_data.dim();

        // We can't determine if class names is valid
        // or not, until we call predict_fn, so store
        // the vector (or an empty) vector and adapt later.
        let class_names = class_names.unwrap_or_default();

        // Create/Check categorical features
        let mut categorical_features = categorical_features.unwrap_or_default();
        categorical_features.sort_unstable();
        if !categorical_features.is_empty()
            && categorical_features[categorical_features.len() - 1] >= num_features
        {
            // Categorical feature index out of range
            return Err(PyValueError::new_err("categorical feature index too large"));
        }

        // Generate continuous features
        let mut continuous_features = vec![];
        for i in 0..num_features {
            if !categorical_features.contains(&i) {
                continuous_features.push(i);
            }
        }

        // Check feature names
        let feature_names = feature_names.unwrap_or_else(|| {
            (0..num_features)
                .map(|i| format!("{}", i))
                .collect::<Vec<String>>()
        });

        if feature_names.len() != num_features {
            return Err(PyValueError::new_err("invalid number of feature names"));
        }

        // We can now create the discretizer
        let discretizer =
            create_discretizer(training_data.view(), &continuous_features, discretizer)?;

        let tab_explainer = kiwi::TabularExplainer::new(
            Arc::new(discretizer),
            num_features,
            class_names,
            categorical_features,
            continuous_features,
            feature_names,
            num_samples,
            sample_background_thread,
        )?;

        Ok(Self {
            tab_explainer: Arc::new(tab_explainer),
        })
    }

    /*
    explain_instance is a method to explain a single record from the dataset.

    Parameters
    -----------
    row: numpy.ndarray
        1D numpy array containing the row.
        It must have the same number of columns as the training
        data, or we will raise a ValueError.

    predict_fn: callable
        Parameters
        ------------
        rows: numpy.ndarray
            2D numpy array containing records from the dataset

        Returns
        -----------
        numpy.ndarray
            2D numpy array containing records x probability of each class.
            1. Only having one column is an error and not allowed.
            2. The returned array *must* be using float64.

    labels: None or list[Int], None
       list of labels (i.e classes) to include in the explanation
       of the record.
       NOTE: We always compute all labels, but won't add the final computation
       to the explanation object if not desired.
       If None, defaults to all classes.

    kernel_width: None or float, None
        A scale parameter used for computing weights for our linear regression.
        If None defaults to sqrt(num_columns * 0.75)

    Returns
    --------
    An instance of the Explainer object
    */
    #[text_signature = "($self, row, predict_fn, *, labels=None, kernel_width=None, regressor_class=None)"]
    #[args(labels = "None", kernel_width = "None")]
    fn explain_instance(
        &self,
        py: Python,
        py_row: &PyArray1<f64>,
        predict_fn: PyObject,
        labels: Option<Vec<usize>>,
        kernel_width: Option<f64>,
    ) -> PyResult<Explainer> {
        let row = unsafe { py_row.as_array() };

        // Call data_inverse
        let row_disc = self.tab_explainer.discretizer.discretize_single(row.view());

        let (data, inverse) = self
            .tab_explainer
            .data_inverse_single(row.view(), row_disc.view())?;

        self.explain_instance_with_data_inverse(
            py,
            py_row,
            PyArray2::from_owned_array(py, data),
            PyArray2::from_owned_array(py, inverse),
            predict_fn,
            labels,
            kernel_width,
        )
    }

    #[text_signature = "($self, row, row_disc, predict_fn, data, inverse, *, labels=None, kernel_width=None)"]
    #[args(labels = "None", kernel_width = "None")]
    fn explain_instance_with_data_inverse(
        &self,
        py: Python,
        py_row: &PyArray1<f64>,
        data: &PyArray2<f64>,
        inverse: &PyArray2<f64>,
        predict_fn: PyObject,
        labels: Option<Vec<usize>>,
        kernel_width: Option<f64>,
    ) -> PyResult<Explainer> {
        let row = unsafe { py_row.as_array() };
        let mut data = unsafe { data.as_array_mut() };

        let num_features = row.shape()[0];

        if num_features != self.tab_explainer.get_num_features() {
            return Err(PyValueError::new_err(
                "number of features in row doesn't match",
            ));
        }

        // Call predict_fn to get yss
        let py_yss = predict_fn
            .call1(py, (inverse,))
            .and_then(|yss| yss.extract::<Py<PyArray2<f64>>>(py))?;

        let ref_yss = py_yss.as_ref(py);
        let mut yss = unsafe { ref_yss.as_array_mut() };

        let num_classes = yss.shape()[1];
        if num_classes == 1 {
            return Err(PyValueError::new_err(
                "predict function returned single value",
            ));
        }

        // Create labels vectors
        let labels = labels.unwrap_or_else(|| (0..num_classes).collect::<Vec<usize>>());
        for &l in labels.iter() {
            if l >= yss.shape()[1] {
                return Err(PyValueError::new_err(
                    "label index too large for num classes",
                ));
            }
        }

        let kernel_width = kernel_width.unwrap_or_else(|| compute_kernel_width(num_features));

        // explain a single instance
        let num_samples = self.tab_explainer.get_num_samples();
        let mut mem = kiwi::ExplainerInstanceMem::new(num_samples, num_features, num_classes);
        let exp = kiwi::TabularExplainer::explain_single(
            self.tab_explainer.clone(),
            kernel_width,
            row.view(),
            data.view_mut(),
            yss.view_mut(),
            &mut mem,
            &labels,
        )?;

        Ok(Explainer { exp })
    }

    #[getter]
    fn get_discretizer(&self) -> PyResult<Discretizer> {
        Ok(Discretizer {
            discretizer: self.tab_explainer.discretizer.clone(),
        })
    }

    #[getter]
    fn get_kernel_width(&self) -> f64 {
        compute_kernel_width(self.tab_explainer.get_num_features())
    }

    // fn get_weights(
    //     &self,
    //     py: Python,
    //     py_data: &PyArray2<f64>,
    //     py_yss: &PyArray2<f64>,
    // ) -> Py<PyArray1<f64>> {
    //     let data = unsafe { py_data.as_array() };
    //     let yss = unsafe { py_yss.as_array() };
    //     let (_, num_features) = data.dim();
    //     let (_, num_classes) = yss.dim();
    //     let num_samples = self.tab_explainer.get_num_samples();

    //     let mut mem = kiwi::ExplainerInstanceMem::new(num_samples, num_features, num_classes);

    //     // TODO: mem alloc

    //     let weights = self.tab_explainer.compute_weights(1.5, &mut mem);

    //     PyArray1::from_array(py, &weights).into_py(py)
    // }

    #[text_signature = "($self, rows, predict_fn, *, labels=None, kernel_width=None, regressor_class=None)"]
    #[args(labels = "None", kernel_width = "None")]
    fn explain_instance_many(
        &self,
        py: Python,
        py_rows: &PyArray2<f64>,
        predict_fn: PyObject,
        labels: Option<Vec<usize>>,
        kernel_width: Option<f64>,
    ) -> PyResult<Vec<Explainer>> {
        let rows = unsafe { py_rows.as_array() };
        let (num_records, num_features) = rows.dim();
        let num_samples = self.tab_explainer.get_num_samples();

        if num_features != self.tab_explainer.get_num_features() {
            return Err(PyValueError::new_err(
                "number of features in row doesn't match",
            ));
        }

        // Do memory allocation
        let mut rows_disc = Array2::<f64>::zeros((num_records, num_features));
        let shape = (num_records, num_samples, num_features);
        let mut datas = Array3::<f64>::zeros(shape);
        let mut inverses = Array3::<f64>::zeros(shape);

        // discretize all the rows
        self.tab_explainer
            .discretizer
            .discretize_many(rows.view(), rows_disc.view_mut());

        // Call data_inverse
        self.tab_explainer.data_inverse_many(
            rows.view(),
            rows_disc.view(),
            datas.view_mut(),
            inverses.view_mut(),
        )?;

        // Call predict_fn to get yss
        let inverses = inverses
            .into_shape((num_records * num_samples, num_features))
            .map_err(|_| PyRuntimeError::new_err("couldn't reshape inverse array"))?;

        let inverses = PyArray2::from_owned_array(py, inverses);
        let py_ysss = predict_fn
            .call1(py, (inverses,))
            .and_then(|yss| yss.extract::<Py<PyArray2<f64>>>(py))?;

        let num_classes = py_ysss.as_ref(py).shape()[1];

        if num_classes == 1 {
            return Err(PyValueError::new_err(
                "predict function returned single value",
            ));
        }

        let py_ysss = py_ysss
            .call_method1(py, "reshape", (num_records, num_samples, num_classes))
            .and_then(|ysss| ysss.extract::<Py<PyArray3<f64>>>(py))?;

        let ref_ysss = py_ysss.as_ref(py);
        let mut ysss = unsafe { ref_ysss.as_array_mut() };

        // Create labels vectors
        let labels = labels.unwrap_or_else(|| (0..num_classes).collect::<Vec<usize>>());
        for &l in labels.iter() {
            if l >= num_classes {
                return Err(PyValueError::new_err(
                    "label index too large for num classes",
                ));
            }
        }

        let kernel_width = kernel_width.unwrap_or_else(|| compute_kernel_width(num_features));

        // // explain a single instance
        let mut mem = kiwi::ExplainerInstanceMem::new(num_samples, num_features, num_classes);
        let mut exps = vec![];
        for n in 0..num_records {
            let data = datas.slice_mut(s![n, .., ..]);
            let yss = ysss.slice_mut(s![n, .., ..]);

            let exp = kiwi::TabularExplainer::explain_single(
                self.tab_explainer.clone(),
                kernel_width,
                rows.row(n),
                data,
                yss,
                &mut mem,
                &labels,
            )?;

            exps.push(Explainer { exp });
        }

        Ok(exps)
    }

    /*
    explain_instance_iter will create background threads to compute
    the explanations of the desired records. It is recomended to use
    explain_instance_many above for anything less than a thousand records.

    Parameters
    ------------
    rows: numpy.ndarray
        see explain_instance
    predict_fn: callable
        see explain_instance
    labels: None or list[Int], None
        see explain_instance
    kernel_width: None or float, None
        see explain_instance
    num_threads: int, 2
        the number of background threads to create to start computing

    Returns
    --------
    An instance of ExplainerIter
    */
    #[text_signature = "($self, rows, predict_fn, *, labels=None, kernel_width=None, num_threads=2, max_memory=32_000_000)"]
    #[args(
        labels = "None",
        kernel_width = "None",
        num_threads = "2",
        max_memory = "32_000_000"
    )]
    fn explain_instance_iter(
        &self,
        py: Python,
        py_rows: &PyArray2<f64>,
        predict_fn: PyObject,
        labels: Option<Vec<usize>>,
        kernel_width: Option<f64>,
        num_threads: usize,
        max_memory: usize,
    ) -> PyResult<ExplainerIter> {
        let rows = py_rows.to_owned_array();

        let (_, num_features) = rows.dim();

        // Call predict_fn on the first row to get num_classes
        let mut row0 = Array2::<f64>::zeros((1, num_features));
        row0.assign(&rows.row(0));
        let ys0 = predict_fn
            .call1(py, (PyArray2::from_owned_array(py, row0),))
            .and_then(|ys| ys.extract::<Py<PyArray2<f64>>>(py))
            .map(|ys| ys.as_ref(py).to_owned_array())?;

        let num_classes = ys0.shape()[1];

        // Labels
        let labels = labels.unwrap_or_else(|| (0..num_classes).collect::<Vec<usize>>());
        for &l in labels.iter() {
            if l >= num_classes {
                return Err(PyValueError::new_err(
                    "label value greater than number classes",
                ));
            }
        }

        // Kernel Width
        let kernel_width = kernel_width.unwrap_or_else(|| (0.75 * num_features as f64).sqrt());

        ExplainerIter::new(
            self.tab_explainer.clone(),
            rows,
            predict_fn,
            kernel_width,
            labels,
            num_classes,
            num_threads,
            max_memory,
        )
    }
}

#[pyclass]
struct ExplainerIter {
    predict_fn: PyObject,
    batch_size: usize,

    rows_to_compute: usize,
    pending_rows: usize,
    computed_rows: HashMap<usize, kiwi::Explainer>,
    completed_rows: usize,

    // Queues to our backgrounds threads
    inq: mpsc::Sender<kiwi::ToComputeRecord>,
    outq: mpsc::Receiver<(usize, kiwi::Explainer)>,
    errq: mpsc::Receiver<kiwi::KiwiError>,
    data_q: mpsc::Receiver<DataInverseBatch>,
}

#[pyproto]
impl PyIterProtocol for ExplainerIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<Explainer>> {
        // Are we finished?
        if slf.done() {
            return Ok(None);
        }

        // Do we have any errors?
        if let Ok(err) = slf.errq.try_recv() {
            return Err(PyErr::from(err));
        }

        // Do we need to call predict_fn?
        let cond = (slf.pending_rows + slf.computed_rows.len()) < (2 * slf.batch_size);
        if slf.rows_to_compute > 0 && cond {
            // Call predict_fn + send to background thread
            slf.call_predict_fn()?;
        }

        loop {
            // Can we receive data?
            slf.recv_exps()?;

            let completed_rows = slf.completed_rows;
            match slf.computed_rows.remove(&completed_rows) {
                Some(exp) => {
                    slf.completed_rows += 1;
                    return Ok(Some(Explainer { exp }));
                }
                None => {
                    // We have to block
                    println!("sleeping");

                    // We might as well call predict_fn instead of sleeping
                    if slf.rows_to_compute > 0 && slf.pending_rows < (2 * slf.batch_size) {
                        slf.call_predict_fn()?;
                        continue;
                    }
                    slf.recv_exp_timeout(time::Duration::from_millis(500))?;
                }
            }
        }
    }
}

impl ExplainerIter {
    fn new(
        tab_explainer: Arc<kiwi::TabularExplainer>,
        rows: Array2<f64>,
        predict_fn: PyObject,
        kernel_width: f64,
        labels: Vec<usize>,
        num_classes: usize,
        num_threads: usize,
        max_memory: usize,
    ) -> PyResult<Self> {
        let (rows_to_compute, num_features) = rows.dim();
        let num_samples = tab_explainer.get_num_samples();

        // Create our q's
        let (inq_s, inq_r) = mpsc::channel();
        let (outq_s, outq_r) = mpsc::channel();
        let (errq_s, errq_r) = mpsc::channel();

        let inq_r = Arc::new(Mutex::new(inq_r));

        // Create our background threads
        for _ in 0..num_threads {
            kiwi::TabularExplainer::new_computation_thread(
                tab_explainer.clone(),
                kernel_width,
                labels.clone(),
                num_classes,
                inq_r.clone(),
                outq_s.clone(),
                errq_s.clone(),
            )?;
        }

        let batch_size = compute_batch_size(max_memory, num_samples, num_features, num_classes);

        let data_q = run_data_inverse(
            tab_explainer,
            rows,
            batch_size,
            num_samples,
            num_features,
            num_classes,
        )?;

        Ok(Self {
            predict_fn,
            rows_to_compute,
            pending_rows: 0,
            computed_rows: HashMap::new(),
            completed_rows: 0,
            batch_size,

            inq: inq_s,
            outq: outq_r,
            errq: errq_r,
            data_q,
        })
    }

    fn recv_exps(&mut self) -> PyResult<()> {
        use mpsc::TryRecvError::*;

        loop {
            match self.outq.try_recv() {
                Ok((id, exp)) => {
                    self.computed_rows.insert(id, exp);
                    self.pending_rows -= 1;
                    continue;
                }
                Err(e) => match e {
                    Disconnected => {
                        return Err(PyRuntimeError::new_err("computation thread died"));
                    }
                    Empty => return Ok(()),
                },
            }
        }
    }

    fn recv_exp_timeout(&mut self, timeout: time::Duration) -> PyResult<()> {
        use mpsc::RecvTimeoutError::*;

        match self.outq.recv_timeout(timeout) {
            Ok((id, exp)) => {
                self.computed_rows.insert(id, exp);
                self.pending_rows -= 1;
                Ok(())
            }
            Err(e) => match e {
                Timeout => Ok(()),
                Disconnected => Err(PyRuntimeError::new_err("computation thread died")),
            },
        }
    }

    fn call_predict_fn(&mut self) -> PyResult<()> {
        if self.rows_to_compute == 0 {
            return Ok(());
        }

        let mut dib = match self.data_q.recv() {
            Ok(dib) => dib,
            Err(_) => {
                return Err(PyRuntimeError::new_err("data inverse thread died"));
            }
        };

        let ysss_dim = dib.ysss.dim();
        let num_records = dib.rows.shape()[0];
        let ysss = &mut dib.ysss;
        let inverses = dib.inverses.take().unwrap();

        Python::with_gil(|py| {
            let inverses = PyArray2::from_owned_array(py, inverses);

            self.predict_fn
                .call1(py, (inverses,))
                .and_then(|yss| yss.extract::<Py<PyArray2<f64>>>(py))
                .and_then(|yss| yss.call_method1(py, "reshape", ysss_dim))
                .and_then(|py_ysss| py_ysss.extract::<Py<PyArray3<f64>>>(py))
                .map(|py_ysss| {
                    let r = py_ysss.as_ref(py);
                    ysss.assign(&unsafe { r.as_array() })
                })
        })?;

        if self.inq.send(dib.into()).is_err() {
            // Our background thread appears to have died
            return Err(PyRuntimeError::new_err("computation threads have died"));
        }

        self.rows_to_compute -= num_records;
        self.pending_rows += num_records;

        Ok(())
    }

    fn done(&self) -> bool {
        self.rows_to_compute == 0 && self.pending_rows == 0 && self.computed_rows.is_empty()
    }
}

#[pyclass(module = "kiwi")]
struct Discretizer {
    discretizer: Arc<kiwi::Discretizer>,
}

#[pymethods]
impl Discretizer {
    #[new]
    #[args(continuous_features = "None")]
    fn new(
        training_data: &PyArray2<f64>,
        percentiles: Vec<usize>,
        continuous_features: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let training_data = unsafe { training_data.as_array() };
        let num_features = training_data.shape()[1];

        let continuous_features =
            continuous_features.unwrap_or_else(|| (0..num_features).collect::<Vec<usize>>());

        for &f in continuous_features.iter() {
            if f >= num_features {
                return Err(PyValueError::new_err("continuous_features out of range"));
            }
        }

        let discretizer = kiwi::Discretizer::new(
            training_data,
            &continuous_features,
            &percentiles,
            compute_percentiles,
        )?;

        Ok(Self {
            discretizer: Arc::new(discretizer),
        })
    }

    #[text_signature = "($self, row)"]
    fn discretize(&self, py: Python, py_row: &PyArray1<f64>) -> Py<PyArray1<f64>> {
        let row = py_row.to_owned_array();
        let row_disc = self.discretizer.discretize_single(row.view());
        PyArray1::from_owned_array(py, row_disc).into_py(py)
    }

    #[text_signature = "($self, num)"]
    fn generate_samples(&self, py: Python, num: usize) -> (Py<PyArray2<f64>>, Py<PyArray2<f64>>) {
        let mut rand_disc = Array2::<f64>::zeros((num, self.discretizer.num_features()));
        let mut rand_undisc = Array2::<f64>::zeros((num, self.discretizer.num_features()));
        let mut rng = rand::rngs::SmallRng::from_entropy();

        self.discretizer
            .generate_samples(&mut rng, rand_disc.view_mut(), rand_undisc.view_mut());

        let rand_disc = PyArray2::from_owned_array(py, rand_disc);
        let rand_undisc = PyArray2::from_owned_array(py, rand_undisc);

        (rand_disc.into_py(py), rand_undisc.into_py(py))
    }

    #[getter]
    fn get_to_discretize(&self) -> Vec<usize> {
        self.discretizer.to_discretize().to_owned()
    }

    #[getter]
    fn get_bins(&self) -> Vec<Vec<f64>> {
        self.discretizer
            .bins()
            .iter()
            .map(|b| b.bins.clone())
            .collect::<Vec<Vec<f64>>>()
    }
}

// KiwiError is a generic exception raised
// for any error in the Kiwi module.
create_exception!(kiwi, KiwiError, PyException);

struct KiwiErrorArgs {
    err: kiwi::KiwiError,
}

impl PyErrArguments for KiwiErrorArgs {
    fn arguments(self, py: Python) -> PyObject {
        PyString::new(py, &self.err.to_string()).into()
    }
}

impl From<kiwi::KiwiError> for PyErr {
    fn from(err: kiwi::KiwiError) -> Self {
        match err {
            // If Kiwi has given us a custom error, it _might_ be a PyErr
            // returned from a callback, try and downcast it.
            kiwi::KiwiError::Custom(ref e) => match e.downcast_ref::<PyErr>() {
                Some(e) => Python::with_gil(|py| e.clone_ref(py)),
                None => PyErr::new::<KiwiError, _>(KiwiErrorArgs { err }),
            },
            // pyo3 maps Io errors to exceptions for us
            kiwi::KiwiError::Io(e) => e.into(),

            // Default to a KiwiError defined above.
            _ => PyErr::new::<KiwiError, _>(KiwiErrorArgs { err }),
        }
    }
}

fn create_discretizer(
    training_data: ArrayView2<f64>,
    continuous_features: &[usize],
    discretizer: Option<&str>,
) -> PyResult<kiwi::Discretizer> {
    let percentiles = match discretizer {
        None => vec![],
        Some(s) => match s {
            "quartile" => vec![25, 50, 75],
            "decile" => vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
            _ => return Err(PyValueError::new_err("invalid discretizer")),
        },
    };

    kiwi::Discretizer::new(
        training_data,
        continuous_features,
        &percentiles,
        compute_percentiles,
    )
    .map_err(|e| e.into())
}

// compute_percentiles will call numpy.percentile to get the values.
// This is done for parity with numpy.
// NOTE: There's no chance of numpy module not being avaliable with
// alll the PyArray variables in this file as well.
fn compute_percentiles(data: &[f64], percentiles: &[usize]) -> kiwi::Result<Vec<f64>> {
    Python::with_gil(|py| {
        let module = PyModule::import(py, "numpy")?;
        let data = PyList::new(py, data);
        let percentiles = PyList::new(py, percentiles);

        module
            .call1("percentile", (data, percentiles))
            .and_then(|obj| obj.extract::<Vec<f64>>())
    })
    .map_err(|e| kiwi::KiwiError::Custom(Arc::new(e)))
}

fn compute_kernel_width(num_features: usize) -> f64 {
    (num_features as f64).sqrt() * 0.75
}

// Module docstring, the comment below is exported to Python.

// Kiwi is an implemetation of the LIME algorithm, tuned for
// performance on a large number of records. It supports classification
// on a large number of records. It currently only supports classifcation
// on tabular data.
#[pymodule]
fn kiwi(py: Python, module: &PyModule) -> PyResult<()> {
    #[pyfn(module, "kernel_fn")]
    fn kernel_fn(_py: Python, width: f64, num: f64) -> f64 {
        kiwi::kernel_fn(width, num)
    }

    module.add_class::<KiwiTabularExplainer>()?;
    module.add_class::<Discretizer>()?;
    module.add("KiwiError", py.get_type::<KiwiError>())?;
    Ok(())
}

fn compute_batch_size(
    max_memory: usize,
    num_samples: usize,
    num_features: usize,
    num_classes: usize,
) -> usize {
    let factor = num_samples * (num_features + num_classes);

    ((max_memory as f64) / factor as f64).floor() as usize
}

struct DataInverseBatch {
    ysss: Array3<f64>,
    datas: Array3<f64>,
    inverses: Option<Array2<f64>>,
    rows: Array2<f64>,
    ids: Vec<usize>,
}

impl From<DataInverseBatch> for kiwi::ToComputeRecord {
    fn from(dib: DataInverseBatch) -> Self {
        Self {
            datas: dib.datas,
            ysss: dib.ysss,
            rows: dib.rows,
            ids: dib.ids,
        }
    }
}

fn run_data_inverse(
    tab_explainer: Arc<kiwi::TabularExplainer>,
    rows: Array2<f64>,
    batch_size: usize,
    num_samples: usize,
    num_features: usize,
    num_classes: usize,
) -> PyResult<mpsc::Receiver<DataInverseBatch>> {
    let (data_s, data_r) = mpsc::sync_channel(4);

    thread::Builder::new()
        .name("data-inverse".to_string())
        .spawn(move || {
            let mut rows_to_compute = rows.shape()[0];

            while rows_to_compute > 0 {
                let num_records = if rows_to_compute < batch_size {
                    rows_to_compute
                } else {
                    batch_size
                };

                let row_start = rows.shape()[0] - rows_to_compute;

                let records = rows
                    .slice(s![row_start..(row_start + num_records), ..])
                    .to_owned();
                let mut datas = Array3::zeros((num_records, num_samples, num_features));
                let mut inverses = Array3::zeros((num_records, num_samples, num_features));
                let mut rows_disc = Array2::zeros(records.dim());
                let ysss = Array3::<f64>::zeros((num_records, num_samples, num_classes));
                let ids = (row_start..(row_start + num_records)).collect::<Vec<usize>>();

                // discretize all the records
                tab_explainer
                    .discretizer
                    .discretize_many(records.view(), rows_disc.view_mut());

                // populate data and inverse
                tab_explainer
                    .data_inverse_many(
                        records.view(),
                        rows_disc.view(),
                        datas.view_mut(),
                        inverses.view_mut(),
                    )
                    .expect("data inverse many");

                // reshape inverses for querying predict_fn
                let inverses = inverses
                    .into_shape((num_records * num_samples, num_features))
                    .expect("couldn't reshape inverses");

                // Send
                if data_s
                    .send(DataInverseBatch {
                        ysss,
                        datas,
                        inverses: Some(inverses),
                        rows: records,
                        ids,
                    })
                    .is_err()
                {
                    // The main object no longer exists,
                    // exit this thread gracefully
                    break;
                }

                rows_to_compute -= num_records;
            }
        })?;

    Ok(data_r)
}
