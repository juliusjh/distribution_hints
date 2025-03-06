use pyo3::prelude::*;

#[allow(unused)]
pub mod kyber;
pub use kyber::*;


#[pyfunction]
pub fn set_seed(seed: Vec<u32>) {
    set_seed_rust(&seed)
}

#[pymodule]
#[cfg(feature = "kyber1024")]
fn python_kyber1024(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ciphertext>()?;
    m.add_class::<SecretKey>()?;
    m.add_class::<PublicKey>()?;
    m.add_class::<KyberSample>()?;
    m.add_class::<Poly>()?;
    m.add_class::<Polyvec>()?;
    m.add_class::<kyber::constants::KyberConstants>()?;
    Ok(())
}

#[pymodule]
#[cfg(feature = "kyber768")]
fn python_kyber768(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ciphertext>()?;
    m.add_class::<SecretKey>()?;
    m.add_class::<PublicKey>()?;
    m.add_class::<KyberSample>()?;
    m.add_class::<Poly>()?;
    m.add_class::<Polyvec>()?;
    m.add_class::<kyber::constants::KyberConstants>()?;
    m.add_function(wrap_pyfunction!(set_seed, m)?)?;
    Ok(())
}

#[pymodule]
#[cfg(feature = "kyber512")]
fn python_kyber512(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ciphertext>()?;
    m.add_class::<SecretKey>()?;
    m.add_class::<PublicKey>()?;
    m.add_class::<KyberSample>()?;
    m.add_class::<Poly>()?;
    m.add_class::<Polyvec>()?;
    m.add_class::<kyber::constants::KyberConstants>()?;
    Ok(())
}
