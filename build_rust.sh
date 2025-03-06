(cd rust_python_kyber/PQClean/crypto_kem/kyber768/clean && make)
(cd hint_solver && RUSTFLAGS=" -C target-cpu=native" maturin develop --release)
(cd rust_python_kyber && RUSTFLAGS=" -C target-cpu=native" maturin develop --release)
