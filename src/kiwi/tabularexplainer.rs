use std::{
    sync::{mpsc, Arc, Mutex},
    thread,
};

use ndarray::*;
use ndarray_linalg::*;
use rand::SeedableRng;

use crate::kiwi::{Discretizer, Explainer, KiwiError, Result};

#[allow(dead_code)]
pub struct TabularExplainer {
    num_features: usize,
    class_names: Vec<String>,
    categorical_features: Vec<usize>,
    continuous_features: Vec<usize>,
    feature_names: Vec<String>,
    sample_background_thread: bool,
    num_samples: usize,

    pub discretizer: Arc<Discretizer>,

    // sampling q
    sample_q: Mutex<mpsc::Receiver<(Array2<f64>, Array2<f64>)>>,
}

pub struct ToComputeRecord {
    pub datas: Array3<f64>,
    pub ysss: Array3<f64>,
    pub rows: Array2<f64>,
    pub ids: Vec<usize>,
}

impl TabularExplainer {
    pub fn new(
        discretizer: Arc<Discretizer>,
        num_features: usize,
        class_names: Vec<String>,
        categorical_features: Vec<usize>,
        continuous_features: Vec<usize>,
        feature_names: Vec<String>,
        num_samples: usize,
        sample_background_thread: bool,
    ) -> Result<Self> {
        let (sampleq_s, sampleq_r) = mpsc::sync_channel(512);

        let d = discretizer.clone();
        if sample_background_thread {
            thread::Builder::new()
                .name("generate-samples".to_string())
                .spawn(move || {
                    run_sample_background(d, num_features, num_samples, sampleq_s);
                })
                .map_err(KiwiError::Io)?;
        }

        Ok(Self {
            num_features,
            class_names,
            categorical_features,
            continuous_features,
            feature_names,
            num_samples,
            sample_background_thread,
            discretizer,

            sample_q: Mutex::new(sampleq_r),
        })
    }

    pub fn get_num_features(&self) -> usize {
        self.num_features
    }

    pub fn get_class_names(&self) -> &'_ [String] {
        &self.class_names
    }

    pub fn get_feature_names(&self) -> &'_ [String] {
        &self.feature_names
    }

    pub fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    fn get_n_samples(&self, n: usize) -> Result<Vec<(Array2<f64>, Array2<f64>)>> {
        let mut samples = vec![];

        if self.sample_background_thread {
            let sampleq = self.sample_q.lock().map_err(|_| KiwiError::MutexError)?;

            for _ in 0..n {
                match sampleq.recv() {
                    Ok(s) => samples.push(s),
                    Err(_) => return Err(KiwiError::SamplingThreadDied),
                }
            }
        } else {
            let mut rng = rand::thread_rng();

            for _ in 0..n {
                let mut rand_disc = Array2::<f64>::zeros((self.num_samples, self.num_features));
                let mut rand_undisc = Array2::<f64>::zeros((self.num_samples, self.num_features));
                self.discretizer.generate_samples(
                    &mut rng,
                    rand_disc.view_mut(),
                    rand_undisc.view_mut(),
                );
                samples.push((rand_disc, rand_undisc))
            }
        }

        Ok(samples)
    }

    fn data_inverse(
        &self,
        row: ArrayView1<f64>,
        row_disc: ArrayView1<f64>,
        mut data: ArrayViewMut2<f64>,
        mut inverse: ArrayViewMut2<f64>,
    ) {
        // Set the first row of rand_disc to our discretized row
        data.row_mut(0).assign(&row_disc);

        // Populate data
        azip!((mut d in data.rows_mut()) {
            d.zip_mut_with(&row_disc, |a, b| {
                *a = if same_f64_uint(*a, *b) {1.0} else {0.0};
            });
        });

        // Set the first row of rec_undisc to our record
        inverse.row_mut(0).assign(&row);
    }

    /// data_inverse takes a record and a discretized version of the record
    pub fn data_inverse_single(
        &self,
        row: ArrayView1<f64>,
        row_disc: ArrayView1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (mut rand_disc, mut rand_undisc) = self.get_n_samples(1)?.remove(0);
        self.data_inverse(row, row_disc, rand_disc.view_mut(), rand_undisc.view_mut());
        Ok((rand_disc, rand_undisc))
    }

    pub fn data_inverse_many(
        &self,
        rows: ArrayView2<f64>,
        rows_disc: ArrayView2<f64>,
        mut datas: ArrayViewMut3<f64>,
        mut inverses: ArrayViewMut3<f64>,
    ) -> Result<()> {
        let (num_records, _, _) = datas.dim();

        for ((sample, mut data), mut inverse) in self
            .get_n_samples(num_records)?
            .into_iter()
            .zip(datas.axis_iter_mut(Axis(0)))
            .zip(inverses.axis_iter_mut(Axis(0)))
        {
            data.assign(&sample.0);
            inverse.assign(&sample.1);
        }

        azip!((
            mut data in datas.axis_iter_mut(Axis(0)),
            mut inverse in inverses.axis_iter_mut(Axis(0)),
            row_disc in rows_disc.rows(),
            row in rows.rows(),
        ) {
            data.row_mut(0).assign(&row_disc);

            azip!((mut d in data.rows_mut()) {
                d.zip_mut_with(&row_disc, |a, b| {
                    *a = if same_f64_uint(*a, *b) { 1.0 } else { 0.0 };
                });
            });

            inverse.row_mut(0).assign(&row);
        });

        Ok(())
    }

    pub fn explain_single(
        slf: Arc<Self>,
        kernel_width: f64,
        row: ArrayView1<f64>,
        mut data: ArrayViewMut2<f64>,
        mut yss: ArrayViewMut2<f64>,
        mem: &mut ExplainerInstanceMem,
        labels: &[usize],
    ) -> Result<Explainer> {
        let proba = yss.slice(s![0, ..]).to_vec();
        mem.compute_weights(kernel_width, data.view());
        mem.compute_offsets(data.view(), yss.view());
        mem.apply_weights(data.view_mut(), yss.view_mut());

        // Perform regression
        let a = data.t().dot(&data);

        let mut exp = Explainer::new(slf, row.to_vec());

        for &l in labels {
            let xy = data.t().dot(&yss.slice(s![.., l]));
            let coeffs = a.solve(&xy).map_err(KiwiError::LinAlg)?;

            // Compute the intercept
            let y_off = mem.y_offsets[l];
            let intercept = y_off - mem.x_offsets.dot(&coeffs.t());

            exp.add_class(l, 0.0, coeffs.to_vec(), intercept, proba[l]);
        }

        Ok(exp)
    }

    fn run_computation_thread(
        slf: Arc<Self>,
        kernel_width: f64,
        labels: Vec<usize>,
        num_classes: usize,
        inq: Arc<Mutex<mpsc::Receiver<ToComputeRecord>>>,
        outq: mpsc::Sender<(usize, Explainer)>,
    ) -> Result<()> {
        loop {
            let mut to_compute = {
                let inq = inq.lock().map_err(|_| KiwiError::MutexError)?;

                match inq.recv() {
                    Ok(to_compute) => to_compute,
                    Err(_) => {
                        // We can only error if the sending thread
                        // has died for some reason, exit gracefully
                        return Ok(());
                    }
                }
                // Release Mutex
            };

            let (_, num_samples, num_features) = to_compute.datas.dim();

            let mut mem = ExplainerInstanceMem::new(num_samples, num_features, num_classes);

            for (i, &id) in to_compute.ids.iter().enumerate() {
                let data = to_compute.datas.slice_mut(s![i, .., ..]);
                let yss = to_compute.ysss.slice_mut(s![i, .., ..]);
                let row = to_compute.rows.row(i);

                let exp = Self::explain_single(
                    slf.clone(),
                    kernel_width,
                    row,
                    data,
                    yss,
                    &mut mem,
                    &labels,
                )?;

                // Send the result back
                if outq.send((id, exp)).is_err() {
                    // Exit gracefully
                    return Ok(());
                }
            }
        }
    }

    pub fn new_computation_thread(
        slf: Arc<Self>,
        kernel_width: f64,
        labels: Vec<usize>,
        num_classes: usize,
        inq: Arc<Mutex<mpsc::Receiver<ToComputeRecord>>>,
        outq: mpsc::Sender<(usize, Explainer)>,
        errq: mpsc::Sender<KiwiError>,
    ) -> Result<()> {
        thread::Builder::new()
            .name("computation".to_string())
            .spawn(move || {
                if let Err(e) =
                    Self::run_computation_thread(slf, kernel_width, labels, num_classes, inq, outq)
                {
                    errq.send(e).unwrap_or(());
                }
            })
            .map_err(KiwiError::Io)
            .map(|_| ())
    }

    // pub fn compute_weights(
    //     &self,
    //     kernel_width: f64,
    //     mem: &mut ExplainerInstanceMem,
    // ) -> Array1<f64> {
    //     mem.compute_weights(kernel_width);
    //     mem.weights_buf.clone()
    // }
}

pub struct ExplainerInstanceMem {
    weights_buf: Array1<f64>,
    x_offsets: Array1<f64>,
    y_offsets: Array1<f64>,
}

impl ExplainerInstanceMem {
    pub fn new(num_samples: usize, num_features: usize, num_classes: usize) -> Self {
        Self {
            weights_buf: Array1::<f64>::zeros(num_samples),
            x_offsets: Array1::<f64>::zeros(num_features),
            y_offsets: Array1::<f64>::zeros(num_classes),
        }
    }

    fn compute_weights(&mut self, kernel_width: f64, data: ArrayView2<f64>) {
        let row0 = data.row(0);

        azip!((w in &mut self.weights_buf, drow in data.rows()) {
            // Compute the distance
            *w = 0.0;

            azip!((d in drow, r in row0) {
                *w += (d - r).powi(2);
            });

            *w = w.sqrt();

            // compute the kernel_fn
            *w = kernel_fn(kernel_width, *w);
        });
    }

    fn compute_offsets(&mut self, data: ArrayView2<f64>, yss: ArrayView2<f64>) {
        let weights = &self.weights_buf;

        azip!((ys in yss.axis_iter(Axis(1)), offs in &mut self.y_offsets) {
            let mut average = 0.0;
            let mut sum = 0.0;

            azip!((w in weights, y in ys) {
                average += y * w;
                sum += w;
            });

            *offs = average / sum;
        });

        azip!((dcol in data.axis_iter(Axis(1)), offs in &mut self.x_offsets) {
            let mut average = 0.0;
            let mut sum = 0.0;

            azip!((w in weights, d in dcol) {
                average += d * w;
                sum += w;
            });

            *offs = average / sum;
        });
    }

    fn apply_weights(&mut self, mut data: ArrayViewMut2<f64>, mut yss: ArrayViewMut2<f64>) {
        azip!((mut col in yss.axis_iter_mut(Axis(1)), offs in &self.y_offsets) {
            col.mapv_inplace(|e| e - offs);
        });

        azip!((mut col in data.axis_iter_mut(Axis(1)), offs in &self.x_offsets) {
            col.mapv_inplace(|e| e - offs);
        });

        // Compute the sqrt of the all the weights
        self.weights_buf.mapv_inplace(|w| w.sqrt());

        let weights = &self.weights_buf;
        azip!((mut dcol in data.axis_iter_mut(Axis(1))) {
            azip!((w in weights, d in &mut dcol) {
                *d *= w;
            });
        });

        azip!((mut ycol in yss.axis_iter_mut(Axis(1))) {
            azip!((w in weights, y in &mut ycol) {
                *y *= w;
            });
        });
    }
}

#[inline]
fn same_f64_uint(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[inline]
pub fn kernel_fn(kernel_width: f64, d: f64) -> f64 {
    (-1.0 * d.powi(2) / kernel_width.powi(2)).exp().sqrt()
}

fn run_sample_background(
    discretizer: Arc<Discretizer>,
    num_features: usize,
    num_samples: usize,
    sampleq: mpsc::SyncSender<(Array2<f64>, Array2<f64>)>,
) {
    let mut rng = rand::rngs::SmallRng::from_entropy();

    loop {
        let mut rand_disc = Array2::<f64>::zeros((num_samples, num_features));
        let mut rand_undisc = Array2::<f64>::zeros((num_samples, num_features));

        discretizer.generate_samples(&mut rng, rand_disc.view_mut(), rand_undisc.view_mut());

        if sampleq.send((rand_disc, rand_undisc)).is_err() {
            break;
        }
    }
}
