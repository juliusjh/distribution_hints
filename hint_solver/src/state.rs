use std::usize;

use num_complex::Complex;
use rayon::prelude::*;
use rustfft::FftPlanner;

use crate::llo::*;
use crate::rhs::*;
use crate::util::*;

pub struct BP<const SZ_MSG: usize, const NVAR: usize, RHS: Rhs<SZ_MSG, NVAR>> {
    fac_results: Vec<[[f64; SZ_MSG]; NVAR]>, //[[f64; SZ_MSG]; NVAR]; NFAC],
    var_results: Vec<[[f64; SZ_MSG]; NVAR]>, //[[[f64; SZ_MSG]; NVAR]; NFAC],
    nfac: usize,
    prior: [f64; SZ_MSG],
    niter: usize,
    chk_coeff: Vec<[i32; NVAR]>,
    chk_rhs: Vec<RHS>, //[RHS; NFAC],
    chk_sz: Vec<usize>,
}

pub fn normalize(msg: &mut [f64]) {
    let max = msg
        .iter()
        .max_by(|a, b| a.partial_cmp(b).expect("Failed to normalize"))
        .expect("Failed to normalize")
        .clone();
    if max > 0.0 {
        msg.iter_mut().for_each(|x| *x /= max)
    } else {
        msg.iter_mut().for_each(|x| *x = 1.0)
    }
}

pub fn to_probabilities(msg: &mut [f64]) {
    let sum: f64 = msg.iter().sum();
    if sum > 0.0 {
        msg.iter_mut().for_each(|x| *x /= sum)
    } else {
        let l = 1.0 / msg.len() as f64;
        msg.iter_mut().for_each(|x| *x = l)
    }
}

fn normalize_complex(msg: &mut [Complex<f64>]) {
    let m: f64 = msg
        .iter()
        .map(|x| x.norm())
        .max_by(|a, b| a.partial_cmp(b).expect("Failed to normalize fft."))
        .expect("Failed to normalize fft.");
    if m > 0.0 {
        msg.iter_mut().for_each(|x| *x /= m);
    } else {
        msg.iter_mut()
            .for_each(|x| *x = Complex::<f64> { re: 1.0, im: 0.0 })
    }
}

impl<const SZ_MSG: usize, const NVAR: usize> BP<SZ_MSG, NVAR, ValueRhs<SZ_MSG, NVAR>> {
    pub fn initialize_value(
        prior: [f64; SZ_MSG],
        coeff: Vec<[i32; NVAR]>,
        rhs: Vec<Vec<(i32, f64)>>,
        chk_sz: Option<Vec<usize>>,
    ) -> Self {
        let rhs_rhs = rhs.into_iter().map(|x| ValueRhs { values: x }).collect();
        Self::initialize(prior, coeff, rhs_rhs, chk_sz)
    }
}

impl<const SZ_MSG: usize, const NVAR: usize> BP<SZ_MSG, NVAR, ProbValueRhs<SZ_MSG, NVAR>> {
    pub fn initialize_prob_value(
        prior: [f64; SZ_MSG],
        coeff: Vec<[i32; NVAR]>,
        rhs: Vec<Vec<(i32, f64)>>,
        p: Vec<f64>,
        chk_sz: Option<Vec<usize>>,
    ) -> Self {
        let rhs_rhs = rhs
            .into_iter()
            .zip(p.into_iter())
            .map(|(x, p)| ProbValueRhs { p: p, values: x })
            .collect();
        Self::initialize(prior, coeff, rhs_rhs, chk_sz)
    }
}

impl<const SZ_MSG: usize, const NVAR: usize, RHS: Rhs<SZ_MSG, NVAR> + Sync> BP<SZ_MSG, NVAR, RHS> {
    pub fn initialize(
        mut prior: [f64; SZ_MSG],
        coeff: Vec<[i32; NVAR]>,
        mut rhs: Vec<RHS>,
        chk_sz: Option<Vec<usize>>,
    ) -> Self {
        let nfac = coeff.len();
        let fac_results = vec![[[1.0; SZ_MSG]; NVAR]; nfac];
        let var_results = vec![[prior.clone(); NVAR]; nfac];
        let mut sums_lens = Vec::new();
        let chk_sz = if let Some(chk_sz) = chk_sz {
            chk_sz
        } else {
            for coeff_i in &coeff {
                let mut sum = 0;
                for c in coeff_i {
                    sum += c.abs() as usize * SZ_MSG / 2;
                }
                sums_lens.push(f64::log2(sum as f64).ceil() as usize + 1);
            }
            sums_lens
        };
        for rhs_i in &mut rhs {
            rhs_i.normalize();
        }
        normalize(&mut prior);
        Self {
            prior,
            chk_coeff: coeff,
            niter: 0,
            fac_results,
            var_results,
            chk_rhs: rhs,
            chk_sz,
            nfac,
        }
    }

    pub fn get_chk_sz(&self) -> Vec<usize> {
        self.chk_sz.clone()
    }

    pub fn get_results(&self) -> Vec<Vec<f64>> {
        (0..NVAR)
            .into_par_iter()
            .map(|i| {
                let mut res = vec![1.0; SZ_MSG];
                for j in 0..self.nfac {
                    for k in 0..SZ_MSG {
                        res[k] *= self.fac_results[j][i][k];
                    }
                }
                for k in 0..SZ_MSG {
                    res[k] *= self.prior[k];
                }
                to_probabilities(&mut res);
                res
            })
            .collect()
    }

    pub fn get_results_max_normed(&self) -> Vec<Vec<f64>> {
        (0..NVAR)
            .into_par_iter()
            .map(|i| {
                let mut res = vec![1.0; SZ_MSG];
                for j in 0..self.nfac {
                    for k in 0..SZ_MSG {
                        res[k] *= self.fac_results[j][i][k];
                    }
                }
                normalize(&mut res);
                res
            })
            .collect()
    }

    pub fn normalize_fac(&mut self) {
        self.fac_results.par_iter_mut().for_each(|v_res| {
            for i in 0..NVAR {
                normalize(&mut v_res[i]);
            }
        });
    }

    pub fn normalize_var(&mut self) {
        self.var_results.par_iter_mut().for_each(|v_res| {
            for i in 0..NVAR {
                normalize(&mut v_res[i]);
            }
        });
    }

    pub fn propagate(
        &mut self,
        fac_batch: Option<usize>,
        damping: Option<f64>,
        worst_vars: Option<usize>,
    ) {
        let fac_batch = fac_batch.unwrap_or(self.nfac);
        self.propagate_fac(fac_batch);
        self.normalize_fac();

        self.propagate_var(fac_batch, damping, worst_vars);
        self.normalize_var();

        self.niter += 1;
    }

    pub fn get_fac_results(&self) -> Vec<[[f64; SZ_MSG]; NVAR]> {
        self.fac_results.clone()
    }

    pub fn get_var_results(&self) -> Vec<[[f64; SZ_MSG]; NVAR]> {
        self.var_results.clone()
    }

    // pub fn fill_var_results(&mut self, start: usize, end: usize) {
    //     self.var_results[start..end]
    //         .par_iter_mut()
    //         .for_each(|v_var| {
    //             for msg in v_var.iter_mut() {
    //                 msg.fill(0.0);
    //             }
    //         });
    // }

    pub fn fill_fac_results(&mut self, fac_batch: usize) {
        let (start, end) = self.get_fac_batch_range(fac_batch);
        self.fac_results[start..end]
            .par_iter_mut()
            .for_each(|v_var| {
                for msg in v_var.iter_mut() {
                    msg.fill(0.0);
                }
            });
    }

    pub fn get_fac_batch_range(&self, fac_batch: usize) -> (usize, usize) {
        let start = (self.niter * fac_batch) % self.nfac;
        let end = std::cmp::min(start + fac_batch, self.nfac);
        (start, end)
    }

    pub fn propagate_var(
        &mut self,
        fac_batch: usize,
        damping: Option<f64>,
        worst_vars: Option<usize>,
    ) {
        let (start, end) = self.get_fac_batch_range(fac_batch);
        let mut res: Vec<(usize, Vec<[f64; SZ_MSG]>)> = (0..NVAR)
            .into_par_iter()
            .map(|i| {
                let mut output = vec![[1f64; SZ_MSG]; self.nfac];
                llo_prior::<SZ_MSG, NVAR>(
                    &mut output[start..end],
                    &self.fac_results[start..end],
                    self.prior.clone(),
                    i,
                );
                (i, output)
            })
            .collect();
        if let Some(worst_vars) = worst_vars {
            let mut res_l = self
                .get_results_max_normed()
                .iter()
                .map(|x| x.iter().sum())
                .enumerate()
                .collect::<Vec<(usize, f64)>>();
            res_l.sort_by(|(_, s), (_, s1)| s.partial_cmp(s1).unwrap());
            let consider = &mut res_l.iter().map(|(i, _)| *i).collect::<Vec<usize>>()[worst_vars..];
            res = res
                .into_iter()
                .filter(|(i, _)| consider.contains(i))
                .collect();
        }
        if let Some(damping) = damping {
            let a_prior = damping;
            let a_new = 1.0 - a_prior;
            self.var_results
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, v_res)| {
                    for (i, output) in res.iter() {
                        for l in 0..SZ_MSG {
                            v_res[*i][l] = a_prior * v_res[*i][l] + a_new * output[j][l];
                        }
                    }
                });
        } else {
            self.var_results
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, v_res)| {
                    for (i, output) in res.iter() {
                        for l in 0..SZ_MSG {
                            v_res[*i][l] = output[j][l];
                        }
                    }
                });
        }
    }

    pub fn propagate_fac(&mut self, fac_batch: usize) {
        let (start, end) = self.get_fac_batch_range(fac_batch);
        self.fill_fac_results(fac_batch);
        self.fac_results[start..end]
            .par_iter_mut()
            .enumerate()
            .for_each(|(mut j, f_res)| {
                j += start;
                let mut chk_mem_fft =
                    vec![vec![Complex { re: 0.0, im: 0.0 }; 1 << self.chk_sz[j]]; NVAR];
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(1 << self.chk_sz[j]);
                for i in 0..NVAR {
                    let coeff = self.chk_coeff[j][i];
                    chk_enlarge_and_mult(
                        &mut chk_mem_fft[i],
                        &self.var_results[j][i],
                        coeff,
                        SZ_MSG,
                        self.chk_sz[j],
                    );
                    assert!(
                        chk_mem_fft[i].iter().map(|x| x.norm()).sum::<f64>() > 0.001,
                        "{:?}\n, j={:?}, i={:?}",
                        self.var_results[j][i],
                        j,
                        i
                    );
                    fft.process(&mut chk_mem_fft[i]);
                    normalize_complex(&mut chk_mem_fft[i]);
                }
                let mut chk_mem_fft_llo =
                    vec![vec![Complex { re: 1.0, im: 0.0 }; 1 << self.chk_sz[j]]; NVAR];
                llo::<SZ_MSG, NVAR>(&mut chk_mem_fft_llo, chk_mem_fft);
                let mut planner = FftPlanner::new();
                let ifft = planner.plan_fft_inverse(1 << self.chk_sz[j]);
                for i in 0..NVAR {
                    normalize_complex(&mut chk_mem_fft_llo[i]);
                    ifft.process(&mut chk_mem_fft_llo[i]);
                    chk_mem_fft_llo[i].iter_mut().for_each(|x| x.re = x.re.abs());
                    normalize_complex(&mut chk_mem_fft_llo[i]);
                }
                self.chk_rhs[j].compute_msg_from_sumdist(
                    f_res,
                    chk_mem_fft_llo,
                    self.chk_coeff[j],
                    SZ_MSG,
                    self.chk_sz[j],
                );
            });
    }

    pub fn get_nfac(&self) -> usize {
        self.nfac
    }

    pub fn get_iteration(&self) -> usize {
        self.niter
    }

    pub fn get_prior(&self) -> Vec<f64> {
        self.prior.to_vec()
    }
}
