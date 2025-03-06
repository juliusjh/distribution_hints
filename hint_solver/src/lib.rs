use num_complex::Complex;
use pyo3::prelude::*;
use std::thread::available_parallelism;

mod greedy;
mod llo;
mod rhs;
mod state;
mod util;
use greedy::Greedy;
use rhs::{ProbValueRhs, ValueRhs};
use state::BP;

const NVAR: usize = 1536;
const SZ_MSG: usize = 5;

pub fn create_pool(num_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Could not create thread pool.")
}

#[pyclass]
pub struct PyGreedy {
    greedy: Greedy,
    thread_pool: rayon::ThreadPool,
    actions: Vec<(i64, f64, usize)>,
}

#[pyclass]
pub struct PyBP {
    bp: BP<SZ_MSG, NVAR, ValueRhs<SZ_MSG, NVAR>>,
    thread_pool: rayon::ThreadPool,
}

#[pyclass]
pub struct PyProbBP {
    bp: BP<SZ_MSG, NVAR, ProbValueRhs<SZ_MSG, NVAR>>,
    thread_pool: rayon::ThreadPool,
}

#[pymodule]
fn hint_solver(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBP>()?;
    m.add_class::<PyProbBP>()?;
    m.add_class::<PyGreedy>()?;
    m.add_function(wrap_pyfunction!(llo_py, m)?)?;
    m.add_function(wrap_pyfunction!(llo_prior, m)?)?;
    Ok(())
}

fn to_array<T: Copy + Clone, const N: usize>(v: Vec<T>) -> Option<[T; N]> {
    if v.len() != N {
        None
    } else {
        Some(core::array::from_fn(|i| v[i].clone()))
    }
}

#[pyfunction]
#[pyo3(name = "llo")]
fn llo_py(input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    assert!(input.len() == NVAR);
    assert!(input.iter().all(|x| x.len() == SZ_MSG));
    let inp = input
        .into_iter()
        .map(|x| x.into_iter().map(|x| Complex { re: x, im: 0.0 }).collect())
        .collect();
    let mut out = vec![vec![Complex { re: 1.0, im: 0.0 }; SZ_MSG]; NVAR];
    llo::llo::<SZ_MSG, NVAR>(&mut out, inp);
    out.into_iter()
        .map(|x| x.into_iter().map(|x| x.re).collect())
        .collect()
}

#[pyfunction]
fn llo_prior(input: Vec<Vec<f64>>, prior: Vec<f64>) -> Vec<Vec<f64>> {
    assert!(input.len() == NVAR);
    assert!(input.iter().all(|x| x.len() == SZ_MSG));
    assert!(prior.len() == SZ_MSG);
    let mut out = vec![[1.0; SZ_MSG]; NVAR];
    let mut inp = vec![[[1.0; SZ_MSG]; NVAR]; NVAR];
    for i in 0..NVAR {
        for l in 0..SZ_MSG {
            inp[i][0][l] = input[i][l];
        }
    }
    llo::llo_prior::<SZ_MSG, NVAR>(&mut out, &inp, to_array::<f64, SZ_MSG>(prior).unwrap(), 0);
    out.into_iter().map(|x| x.to_vec()).collect()
}



#[pymethods]
impl PyBP {
    #[new]
    fn new(
        prior: Vec<f64>,
        coeff: Vec<Vec<i32>>,
        rhs: Vec<Vec<(i32, f64)>>,
        sz_chk: Option<usize>,
    ) -> Self {
        let n = coeff.len();
        let coeffs = coeff
            .into_iter()
            .map(|x| to_array::<i32, NVAR>(x).expect("Coefficients have incorrect length."))
            .collect();
        let nthr = available_parallelism().unwrap().get();
        let pool = create_pool(nthr / 2);
        let sz_chk = sz_chk.map(|x| vec![x; n]);
        Self {
            bp: BP::initialize_value(
                to_array::<f64, SZ_MSG>(prior).expect("Prior has wrong size."),
                coeffs,
                rhs,
                sz_chk,
            ),
            thread_pool: pool,
        }
    }

    fn get_chk_sz(&self) -> Vec<usize> {
        self.bp.get_chk_sz()
    }

    fn get_nthreads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    fn set_nthreads(&mut self, nthreads: usize) {
        let pool = create_pool(nthreads);
        self.thread_pool = pool;
    }

    fn get_prior(&self) -> Vec<f64> {
        self.bp.get_prior()
    }

    fn get_nfac(&self) -> usize {
        self.bp.get_nfac()
    }

    fn get_nvar(&self) -> usize {
        NVAR
    }

    fn get_fac_results(&self) -> Vec<Vec<Vec<f64>>> {
        self.bp
            .get_fac_results()
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.to_vec()).collect())
            .collect()
    }

    fn get_var_results(&self) -> Vec<Vec<Vec<f64>>> {
        self.bp
            .get_var_results()
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.to_vec()).collect())
            .collect()
    }

    fn propagate_fac(&mut self, batch: usize) {
        self.thread_pool.install(|| self.bp.propagate_fac(batch))
    }

    fn propagate_var(&mut self, batch: usize, damping: Option<f64>, worst_vars: Option<usize>) {
        self.thread_pool
            .install(|| self.bp.propagate_var(batch, damping, worst_vars))
    }
    fn normalize_var(&mut self) {
        self.thread_pool.install(|| self.bp.normalize_var())
    }

    fn normalize_fac(&mut self) {
        self.thread_pool.install(|| self.bp.normalize_fac())
    }

    fn get_results(&mut self) -> Vec<Vec<f64>> {
        self.thread_pool.install(|| self.bp.get_results())
    }

    fn propagate(&mut self, batch: Option<usize>, damping: Option<f64>, worst_vars: Option<usize>) {
        self.thread_pool
            .install(|| self.bp.propagate(batch, damping, worst_vars))
    }

    fn get_niter(&self) -> usize {
        self.bp.get_iteration()
    }
}

#[pymethods]
impl PyGreedy {
    #[new]
    fn new(coeffs: Vec<Vec<i64>>, rhs: Vec<Vec<(i64, f64)>>) -> Self {
        let nthr = available_parallelism().unwrap().get();
        let thread_pool = create_pool(nthr / 2);
        let nvar = coeffs[0].len();
        let eta = SZ_MSG / 2;
        let mut actions = vec![(0, f64::INFINITY, 0); nvar];
        for i in 0..nvar {
            actions[i] = (0, f64::INFINITY, i);
        }
        let greedy = Greedy::initialize(vec![0; nvar], coeffs, rhs, eta as i64);
        Self {
            greedy,
            thread_pool,
            actions,
        }
    }
    fn solve(&mut self, num_changes: usize) {
        self.actions = self.thread_pool
            .install(|| self.greedy.solve(num_changes));
    }

    fn get_guess(&self) -> Vec<i64> {
        self.greedy.get_guess()
    }

    fn get_actions(&self) -> Vec<(i64, f64, usize)> {
        self.actions.clone()
    }

    fn get_nthreads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    fn set_nthreads(&mut self, nthreads: usize) {
        let pool = create_pool(nthreads);
        self.thread_pool = pool;
    }
}

#[pymethods]
impl PyProbBP {
    #[new]
    fn new(
        prior: Vec<f64>,
        coeff: Vec<Vec<i32>>,
        rhs: Vec<Vec<(i32, f64)>>,
        p: Vec<f64>,
        sz_chk: Option<usize>,
    ) -> Self {
        let n = coeff.len();
        let coeffs = coeff
            .into_iter()
            .map(|x| to_array::<i32, NVAR>(x).expect("Coefficients have incorrect length."))
            .collect();
        let nthr = available_parallelism().unwrap().get();
        let pool = create_pool(nthr / 2);
        let sz_chk = sz_chk.map(|x| vec![x; n]);
        Self {
            bp: BP::initialize_prob_value(
                to_array::<f64, SZ_MSG>(prior).expect("Prior has wrong size."),
                coeffs,
                rhs,
                p,
                sz_chk,
            ),
            thread_pool: pool,
        }
    }

    fn get_chk_sz(&self) -> Vec<usize> {
        self.bp.get_chk_sz()
    }

    fn get_nthreads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    fn set_nthreads(&mut self, nthreads: usize) {
        let pool = create_pool(nthreads);
        self.thread_pool = pool;
    }

    fn get_prior(&self) -> Vec<f64> {
        self.bp.get_prior()
    }

    fn get_nfac(&self) -> usize {
        self.bp.get_nfac()
    }

    fn get_nvar(&self) -> usize {
        NVAR
    }

    fn get_fac_results(&self) -> Vec<Vec<Vec<f64>>> {
        self.bp
            .get_fac_results()
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.to_vec()).collect())
            .collect()
    }

    fn get_var_results(&self) -> Vec<Vec<Vec<f64>>> {
        self.bp
            .get_var_results()
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.to_vec()).collect())
            .collect()
    }

    fn propagate_fac(&mut self, batch: usize) {
        self.thread_pool.install(|| self.bp.propagate_fac(batch))
    }

    fn propagate_var(&mut self, batch: usize, damping: Option<f64>, worst_vars: Option<usize>) {
        self.thread_pool
            .install(|| self.bp.propagate_var(batch, damping, worst_vars))
    }
    fn normalize_var(&mut self) {
        self.thread_pool.install(|| self.bp.normalize_var())
    }

    fn normalize_fac(&mut self) {
        self.thread_pool.install(|| self.bp.normalize_fac())
    }

    fn get_results(&mut self) -> Vec<Vec<f64>> {
        self.thread_pool.install(|| self.bp.get_results())
    }

    fn propagate(&mut self, batch: Option<usize>, damping: Option<f64>, worst_vars: Option<usize>) {
        self.thread_pool
            .install(|| self.bp.propagate(batch, damping, worst_vars))
    }

    fn get_niter(&self) -> usize {
        self.bp.get_iteration()
    }
}
