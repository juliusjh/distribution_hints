use crate::{
    state::to_probabilities,
    util::{from_compl, to_compl},
};
use num_complex::Complex;

pub trait Rhs<const SZ_MSG: usize, const NVAR: usize> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut [[f64; SZ_MSG]; NVAR],
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: [i32; NVAR],
        sz_msg: usize,
        chk_sz: usize,
    );
    fn normalize(&mut self);
}

pub struct EqRhs<const SZ_MSG: usize, const NVAR: usize> {
    pub value: i32,
}

impl<const SZ_MSG: usize, const NVAR: usize> Rhs<SZ_MSG, NVAR> for EqRhs<SZ_MSG, NVAR> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut [[f64; SZ_MSG]; NVAR],
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: [i32; NVAR],
        sz_msg: usize,
        chk_sz: usize,
    ) {
        let sz_chk = 1 << chk_sz;
        for i in 0..out_msgs.len() {
            let coeff = coeffs[i];
            for v in 0..sz_msg as u32 {
                let v_m = from_compl(v, sz_msg) * coeff;
                let v_acc = to_compl(self.value - v_m, sz_chk);
                out_msgs[i][v as usize] += llo[i][v_acc].re;
            }
        }
    }
    fn normalize(&mut self) {}
}

pub struct ProbValueRhs<const SZ_MSG: usize, const NVAR: usize> {
    pub values: Vec<(i32, f64)>,
    pub p: f64,
}

impl<const SZ_MSG: usize, const NVAR: usize> Rhs<SZ_MSG, NVAR> for ProbValueRhs<SZ_MSG, NVAR> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut [[f64; SZ_MSG]; NVAR],
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: [i32; NVAR],
        sz_msg: usize,
        chk_sz: usize,
    ) {
        let sz_chk = 1 << chk_sz;
        for i in 0..out_msgs.len() {
            let coeff = coeffs[i];
            for v in 0..sz_msg as u32 {
                for (value, p) in &self.values {
                    let v_m = from_compl(v, sz_msg) * coeff;
                    let v_acc = to_compl(value - v_m, sz_chk);
                    out_msgs[i][v as usize] += p * llo[i][v_acc].re;
                }
            }
            to_probabilities(&mut out_msgs[i]);
            let q = (1.0 - self.p) / (sz_msg as f64);
            for v in 0..sz_msg {
                out_msgs[i][v] *= self.p;
                out_msgs[i][v] += q;
            }
        }
    }

    fn normalize(&mut self) {
        let sum: f64 = *self
            .values
            .iter()
            .map(|(_v, p)| p)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        self.values.iter_mut().for_each(|x| x.1 /= sum);
    }
}

pub struct ValueRhs<const SZ_MSG: usize, const NVAR: usize> {
    pub values: Vec<(i32, f64)>,
}

impl<const SZ_MSG: usize, const NVAR: usize> Rhs<SZ_MSG, NVAR> for ValueRhs<SZ_MSG, NVAR> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut [[f64; SZ_MSG]; NVAR],
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: [i32; NVAR],
        sz_msg: usize,
        chk_sz: usize,
    ) {
        let sz_chk = 1 << chk_sz;
        for i in 0..out_msgs.len() {
            let coeff = coeffs[i];
            for v in 0..sz_msg as u32 {
                for (value, p) in &self.values {
                    let v_m = from_compl(v, sz_msg) * coeff;
                    let v_acc = to_compl(value - v_m, sz_chk);
                    // if i == 0 {
                    //     println!("v={}, v_m={}, v_acc={}, p={}, llo[i][v_acc]={}", v, v_m, v_acc, p, llo[i][v_acc]);
                    // }
                    out_msgs[i][v as usize] += p * llo[i][v_acc].re;
                }
            }
        }
    }

    fn normalize(&mut self) {
        let sum: f64 = *self
            .values
            .iter()
            .map(|(_v, p)| p)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        self.values.iter_mut().for_each(|x| x.1 /= sum);
    }
}
