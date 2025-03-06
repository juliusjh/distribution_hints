use num_complex::Complex;

pub fn llo<const SZ_MSG: usize, const N: usize>(
    out: &mut Vec<Vec<Complex<f64>>>,
    inp: Vec<Vec<Complex<f64>>>,
) {
    let mut acc_fwd = inp.first().unwrap().clone();
    let mut acc_bwd = inp.last().unwrap().clone();
    for l in 1..N {
        out[l]
            .iter_mut()
            .zip(acc_fwd.iter())
            .for_each(|(o, a)| *o *= a);
        acc_fwd
            .iter_mut()
            .zip(inp[l].iter())
            .for_each(|(o, a)| *o *= a);
    }
    for l in (0..N - 1).rev() {
        out[l]
            .iter_mut()
            .zip(acc_bwd.iter())
            .for_each(|(o, a)| *o *= a);
        acc_bwd
            .iter_mut()
            .zip(inp[l].iter())
            .for_each(|(o, a)| *o *= a);
    }
}

pub fn llo_prior<const SZ_MSG: usize, const NVAR: usize>(
    out: &mut [[f64; SZ_MSG]],
    inp: &[[[f64; SZ_MSG]; NVAR]],
    prior: [f64; SZ_MSG],
    i: usize,
) {
    let mut acc_fwd = prior;
    let mut acc_bwd = [1f64; SZ_MSG];
    let nfac = out.len();
    for j in 0..nfac {
        out[j]
            .iter_mut()
            .zip(acc_fwd.iter())
            .for_each(|(o, a)| *o *= a);
        acc_fwd
            .iter_mut()
            .zip(inp[j][i].iter())
            .for_each(|(o, a)| *o *= a);
    }
    for j in (0..nfac).rev() {
        out[j]
            .iter_mut()
            .zip(acc_bwd.iter())
            .for_each(|(o, a)| *o *= a);
        acc_bwd
            .iter_mut()
            .zip(inp[j][i].iter())
            .for_each(|(o, a)| *o *= a);
    }
}

// pub fn llo_prior_filter<const SZ_MSG: usize, const NVAR: usize>(
//     out: &mut [[f64; SZ_MSG]],
//     inp: &[[[f64; SZ_MSG]; NVAR]],
//     prior: [f64; SZ_MSG],
//     i: usize,
//     consider: &[usize],
// ) {
//     let mut acc_fwd = prior;
//     let mut acc_bwd = [1f64; SZ_MSG];
//     for j in consider {
//         let j = *j;
//         out[j]
//             .iter_mut()
//             .zip(acc_fwd.iter())
//             .for_each(|(o, a)| *o *= a);
//         acc_fwd
//             .iter_mut()
//             .zip(inp[j][i].iter())
//             .for_each(|(o, a)| *o *= a);
//     }
//     for j in consider.iter().rev() {
//         let j = *j;
//         out[j]
//             .iter_mut()
//             .zip(acc_bwd.iter())
//             .for_each(|(o, a)| *o *= a);
//         acc_bwd
//             .iter_mut()
//             .zip(inp[j][i].iter())
//             .for_each(|(o, a)| *o *= a);
//     }
// }
