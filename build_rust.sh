cd hint_solver && RUSTFLAGS=" -C target-cpu=native" maturin develop --release
cd ..
cd rust_python_kyber && RUSTFLAGS=" -C target-cpu=native" maturin develop --release
cd ..
