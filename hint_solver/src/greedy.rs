use rayon::prelude::*;

pub struct Greedy {
    nfac: usize,
    nvar: usize,
    coeff: Vec<Vec<i64>>,
    rhs: Vec<Vec<(i64, f64)>>, //[RHS; nfac],
    guess: Vec<i64>,
    eta: i64,
}

impl Greedy {
    pub fn initialize(
        guess: Vec<i64>,
        coeff: Vec<Vec<i64>>,
        rhs: Vec<Vec<(i64, f64)>>,
        eta: i64,
    ) -> Self {
        let nfac = rhs.len();
        let nvar = guess.len();
        assert!(coeff.iter().all(|x| x.len() == nvar));
        assert!(coeff.len() == nfac);
        Self {
            guess,
            coeff,
            rhs,
            nfac,
            nvar,
            eta,
        }
    }

    fn change_to_index(&self, change: i64) -> usize {
        let max_amount_changes = 4 * self.eta + 1;
        if change < 0 {
            (change + max_amount_changes) as usize
        } else {
            change as usize
        }
    }

    fn index_to_change(&self, idx: usize) -> i64 {
        let max_amount_changes = 4 * self.eta + 1;
        if idx as i64 > max_amount_changes / 2 {
            idx as i64 - max_amount_changes
        } else {
            idx as i64
        }
    }

    pub fn compute_scores(&mut self) -> Vec<Vec<f64>> {
        let mut actions = vec![vec![0.0; 2 * 2 * self.eta as usize + 1]; self.nvar];
        let actions_i: Vec<Vec<Vec<f64>>> = (0..self.nfac).into_par_iter().map(|i| {
            let mut actions_i = vec![vec![f64::INFINITY; 2 * 2 * self.eta as usize + 1]; self.nvar];
            let mut partial_sum = 0;
            for k in 0..self.nvar {
                partial_sum += self.coeff[i][k] * self.guess[k];
            }
            for j in 0..self.nvar {
                if self.coeff[i][j] == 0 {
                    continue;
                }
                let g = self.guess[j];
                let min = -self.eta - g;
                let max = self.eta - g;
                for c in min..=max {
                    actions_i[j][self.change_to_index(c)] = 0.0;
                    let partial_sum_c = partial_sum + c * self.coeff[i][j];
                    let mut local_score = 0f64;
                    for (a, p) in &self.rhs[i] {
                        local_score += *p * (partial_sum_c - a).abs() as f64;
                    }
                    actions_i[j][self.change_to_index(c)] += local_score;
                }
            }
            actions_i
        }).collect();
        for i in 0..self.nfac {
            for j in 0..self.nvar {
                if self.coeff[i][j] == 0 {
                    continue;
                }
                for idx in 0..(4*self.eta as usize + 1) {
                    actions[j][idx] += actions_i[i][j][idx]
                }
            }
        }
        actions
    }

    pub fn solve(&mut self, changes: usize) -> Vec<(i64, f64, usize)> {
        let actions = self.compute_scores();
        let mut best_actions = vec![(0, f64::INFINITY, 0); self.nvar];
        for j in 0..self.nvar {
            let actions_j = &actions[j];
            let mut best_action = actions_j[0];
            let mut ba_idx = 0;
            for (k, score) in actions_j.iter().enumerate() {
                if score < &best_action {
                    best_action = *score;
                    ba_idx = k;
                }
            }
            let c = self.index_to_change(ba_idx);
            best_actions[j] = (c, best_action, j);
        }
        best_actions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (c, _, j) in &best_actions[..std::cmp::min(changes, self.nvar)] {
            self.guess[*j] += c;
        }
        best_actions
    }

    pub fn get_guess(&self) -> Vec<i64> {
        self.guess.clone()
    }
}
