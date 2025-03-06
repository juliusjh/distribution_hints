use num_complex::Complex;

pub fn from_compl(v: u32, msg_sz: usize) -> i32 {
    if v > (msg_sz as u32 / 2) {
        return v as i32 - msg_sz as i32;
    }
    v as i32
}

pub fn to_compl(mut v: i32, msg_sz: usize) -> usize {
    // (v + ((v >> 31) & 1) * msg_sz as i32) as usize
    if v < 0 {
        v += msg_sz as i32;
    }
    v as usize
}

pub fn chk_enlarge_and_mult(
    out_mem: &mut [Complex<f64>],
    in_msg: &[f64],
    coeff: i32,
    msg_sz: usize,
    chk_sz: usize,
) {
    let chk_sz_msg = (1 << chk_sz) as usize;
    let add = (chk_sz_msg as i32 - msg_sz as i32) as usize;
    let mask = (chk_sz_msg - 1) as i32;
    let msg_sz_half = (msg_sz >> 1) as usize;
    for l in 0..msg_sz as usize {
        let i2 = if l > msg_sz_half { l + add } else { l };
        let v = ((coeff * i2 as i32) & mask) as usize;
        out_mem[v] += Complex {
            re: in_msg[l as usize],
            im: 0f64,
        }
    }
}
