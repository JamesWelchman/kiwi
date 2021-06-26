use std::collections::HashMap;

use ndarray::*;

use crate::kiwi::{KiwiError, Result};

use rand::distributions::Distribution;

pub struct Discretizer {
    samplers: Vec<Sampler>,

    to_discretize: Vec<usize>,
    bins: Vec<Bin>,
}

impl Discretizer {
    pub fn new<F>(
        training_data: ArrayView2<f64>,
        continuous_features: &[usize],
        percentiles: &[usize],
        compute_percentiles: F,
    ) -> Result<Self>
    where
        F: Fn(&[f64], &[usize]) -> Result<Vec<f64>>,
    {
        let (num_rows, num_features) = training_data.dim();
        let mut sorting_buffer: Vec<f64> = Vec::with_capacity(num_rows);
        let mut counting_buffer: Vec<usize> = Vec::with_capacity(num_rows);

        let mut samplers = vec![];
        let mut bins = vec![];

        for feature_ind in 0..num_features {
            sorting_buffer.clear();
            counting_buffer.clear();

            for &e in training_data.column(feature_ind) {
                sorting_buffer.push(e);
            }

            if !continuous_features.contains(&feature_ind) {
                // Not a continuous column
                fill_counting_buffer(&mut counting_buffer, &sorting_buffer)?;
                samplers.push(Sampler::from_counts(&counting_buffer)?);
                continue;
            }

            sort_buffer(&mut sorting_buffer)?;

            // compute our numerical percentiles
            let percentiles = compute_percentiles(&sorting_buffer, percentiles)?;

            // compute the bins
            let bin = Bin::from_percentiles(&percentiles);

            bin.discretize_column(&mut counting_buffer, &sorting_buffer);

            // Find the boundaries in the discretized column
            let boundaries = compute_boundaries(&counting_buffer);

            // Compute the stats for every boundary
            let mut slice_stats = vec![];
            for n in 0..(boundaries.len()) {
                let stats = SliceStats::from_sorted_slice(
                    &sorting_buffer[if n == (boundaries.len() - 1) {
                        boundaries[n]..sorting_buffer.len()
                    } else {
                        boundaries[n]..boundaries[n + 1]
                    }],
                );

                slice_stats.push(stats);
            }

            bins.push(bin);
            samplers.push(Sampler::from_slice_stats(&counting_buffer, slice_stats)?);
        }

        Ok(Self {
            samplers,
            bins,
            to_discretize: continuous_features.to_owned(),
        })
    }

    pub fn generate_samples(
        &self,
        rng: &mut impl rand::Rng,
        mut rand_disc: ArrayViewMut2<f64>,
        mut rand_undisc: ArrayViewMut2<f64>,
    ) {
        azip!((mut disc_row in rand_disc.rows_mut(), mut undisc_row in rand_undisc.rows_mut()) {
            let it = disc_row.iter_mut()
                .zip(undisc_row.iter_mut())
                .zip(self.samplers.iter());

            for ((d, ud), sampler) in it {
                let s = sampler.sample(rng);
                *d = s.0;
                *ud = s.1;
            }
        });
    }

    fn _discretize(&self, row: ArrayView1<f64>, mut row_disc: ArrayViewMut1<f64>) {
        for (&ind, bin) in self.to_discretize.iter().zip(self.bins.iter()) {
            row_disc[ind] = bin.discretize(row[ind]) as f64;
        }
    }

    pub fn discretize_single(&self, row: ArrayView1<f64>) -> Array1<f64> {
        let mut row_disc = Array1::<f64>::zeros(row.dim());
        self._discretize(row, row_disc.view_mut());
        row_disc
    }

    pub fn discretize_many(&self, rows: ArrayView2<f64>, mut rows_disc: ArrayViewMut2<f64>) {
        azip!((row in rows.rows(), row_disc in rows_disc.rows_mut()) {
            self._discretize(row, row_disc);
        });
    }

    pub fn num_features(&self) -> usize {
        self.samplers.len()
    }

    pub fn to_discretize(&self) -> &'_ [usize] {
        &self.to_discretize
    }

    pub fn bins(&self) -> Vec<&'_ Bin> {
        self.bins.iter().collect::<Vec<&Bin>>()
    }

    pub fn get_bounds(&self, feature: usize, val: f64) -> Result<(Option<f64>, Option<f64>)> {
        if feature >= self.samplers.len() {
            return Err(KiwiError::FeatureOutOfRange(feature));
        }

        let mut bin_ind: Option<usize> = None;
        for (idx, &feature_ind) in self.to_discretize.iter().enumerate() {
            if feature_ind == feature {
                bin_ind = Some(idx);
                break;
            }
        }

        let bin_ind = match bin_ind {
            Some(i) => i,
            None => return Ok((None, None)),
        };

        let bin = self.bins()[bin_ind];
        let mut upper_bound = bin.bins.len();

        for (n, &b) in bin.bins.iter().enumerate() {
            if val <= b {
                upper_bound = n;
                break;
            }
        }

        let bounds = if upper_bound == 0 {
            (None, Some(bin.bins[upper_bound]))
        } else if upper_bound == bin.bins.len() {
            (Some(bin.bins[upper_bound - 1]), None)
        } else {
            (Some(bin.bins[upper_bound - 1]), Some(bin.bins[upper_bound]))
        };

        Ok(bounds)
    }
}

struct Sampler {
    values: Vec<f64>,
    count_dist: rand::distributions::WeightedIndex<usize>,
    undiscretizers: Option<Vec<Undiscretizer>>,
}

impl Sampler {
    fn sample(&self, rng: &mut impl rand::Rng) -> (f64, f64) {
        let disc = self.count_dist.sample(rng);

        match self.undiscretizers {
            None => (self.values[disc], self.values[disc]),
            Some(ref undiscretizers) => {
                let undiscretizer = &undiscretizers[disc];

                (self.values[disc], undiscretizer.sample(rng))
            }
        }
    }

    fn from_slice_stats(counting_buffer: &[usize], slice_stats: Vec<SliceStats>) -> Result<Self> {
        let counts = count_values(&counting_buffer);

        let weights = counts.iter().map(|(_, w)| *w).collect::<Vec<usize>>();

        let count_dist =
            rand::distributions::WeightedIndex::new(&weights).map_err(KiwiError::WeightedError)?;

        let values = counts.iter().map(|(v, _)| *v as f64).collect::<Vec<f64>>();

        let undiscretizers = slice_stats
            .into_iter()
            .map(Undiscretizer::from_slice_stats)
            .collect::<Result<Vec<Undiscretizer>>>()?;

        Ok(Self {
            values,
            count_dist,
            undiscretizers: Some(undiscretizers),
        })
    }

    fn from_counts(counting_buffer: &[usize]) -> Result<Self> {
        let counts = count_values(&counting_buffer);

        let weights = counts.iter().map(|(_, w)| *w).collect::<Vec<usize>>();

        let count_dist =
            rand::distributions::WeightedIndex::new(&weights).map_err(KiwiError::WeightedError)?;

        Ok(Self {
            count_dist,
            undiscretizers: None,
            values: counts.iter().map(|(v, _)| *v as f64).collect::<Vec<f64>>(),
        })
    }
}

struct Undiscretizer {
    stats: SliceStats,
    dist: Option<statrs::distribution::Normal>,
}

impl Undiscretizer {
    fn sample(&self, rng: &mut impl rand::Rng) -> f64 {
        match self.dist {
            None => self.stats.min,
            Some(dist) => {
                let mut v = self.stats.max + 1.0;

                while v < self.stats.min || v > self.stats.max {
                    v = dist.sample(rng);
                }

                v
            }
        }
    }

    fn from_slice_stats(stats: SliceStats) -> Result<Self> {
        let dist = if stats.all_same() {
            None
        } else {
            let dst = statrs::distribution::Normal::new(stats.mean, stats.std)
                .map_err(KiwiError::StatsError)?;

            Some(dst)
        };

        Ok(Self { stats, dist })
    }
}

pub struct Bin {
    pub bins: Vec<f64>,
}

impl Bin {
    fn discretize(&self, val: f64) -> usize {
        let mut boundary: Option<usize> = None;

        for (j, &b) in self.bins.iter().enumerate() {
            if val <= b {
                boundary = Some(j);
                break;
            }
        }

        match boundary {
            Some(j) => j,
            None => self.bins.len(),
        }
    }

    fn from_percentiles(percentiles: &[f64]) -> Self {
        let mut bins = vec![percentiles[0]];

        for &f in percentiles[1..].iter() {
            // Only push f if it's different from the previous value
            if (f - bins[bins.len() - 1]) < 1e-12 {
                continue;
            }

            bins.push(f);
        }

        Self { bins }
    }

    fn discretize_column(&self, dst: &mut Vec<usize>, src: &[f64]) {
        for &s in src.iter() {
            dst.push(self.discretize(s));
        }
    }
}

fn sort_buffer(buf: &mut Vec<f64>) -> Result<()> {
    // Check we don't have any NaN/inf
    for &e in buf.iter() {
        if !e.is_finite() {
            return Err(KiwiError::NonFiniteF64(e));
        }
    }

    buf.sort_by(|a: &f64, b: &f64| {
        use std::cmp::Ordering::*;

        if a < b {
            Less
        } else if a > b {
            Greater
        } else {
            Equal
        }
    });

    Ok(())
}

fn compute_boundaries(buf: &[usize]) -> Vec<usize> {
    let mut ans = vec![0];
    let mut highest = buf[0];

    for (n, &v) in buf[1..].iter().enumerate() {
        if v != highest {
            ans.push(n);
            highest = v;
        }
    }

    ans
}

#[derive(Debug, Copy, Clone)]
struct SliceStats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

impl SliceStats {
    fn from_sorted_slice(slice: &[f64]) -> Self {
        let mean = statistics::mean(slice);
        let std = statistics::variance(slice).sqrt();
        let min = slice[0];
        let max = slice[slice.len() - 1];
        Self {
            mean,
            std,
            min,
            max,
        }
    }

    fn all_same(self) -> bool {
        (self.max - self.min) < 1e-12 || self.std < 1e-12
    }
}

fn count_values(buf: &[usize]) -> Vec<(usize, usize)> {
    let mut map = HashMap::new();

    for &b in buf {
        if let Some(v) = map.get_mut(&b) {
            *v += 1;
        } else {
            map.insert(b, 1);
        }
    }

    let mut counts = map.into_iter().collect::<Vec<(usize, usize)>>();
    counts.sort_by(|a: &(usize, usize), b: &(usize, usize)| {
        use std::cmp::Ordering::*;

        match a.0 < b.0 {
            true => Less,
            false => match a.0 == b.0 {
                true => Equal,
                false => Greater,
            },
        }
    });

    counts
}

fn fill_counting_buffer(dst: &mut Vec<usize>, src: &[f64]) -> Result<()> {
    for &f in src {
        if !f.is_finite() {
            return Err(KiwiError::NonFiniteF64(f));
        }

        if (f - f.floor()) > 1e-12 {
            return Err(KiwiError::CategoricalNonInt(f));
        }

        dst.push(f.floor() as usize);
    }

    Ok(())
}
