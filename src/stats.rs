pub struct TokenStats {
    pub token_count: Vec<u64>,
    pub pair_count: Vec<u64>,
    pub literal_count: [u64; 256],
    pub scanned_bytes: u64,
}

impl TokenStats {
    pub fn new(ntokens: usize, literal_cost: f64) -> Self {
        let mut token_count = Vec::new();
        token_count.resize(ntokens, 0);

        let mut pair_count = Vec::new();
        pair_count.resize(ntokens * ntokens, 0);

        TokenStats {
            token_count,
            pair_count,
            literal_count: [0; 256],
            scanned_bytes: 0,
        }
    }

    pub fn add(&mut self, other: &TokenStats) {
        for i in 0..self.token_count.len() {
            self.token_count[i] += other.token_count[i];
        }
        for i in 0..self.pair_count.len() {
            self.pair_count[i] += other.pair_count[i];
        }
        for i in 0..256 {
            self.literal_count[i] += other.literal_count[i];
        }
        self.scanned_bytes += other.scanned_bytes;
    }

    pub fn total_literals(&self) -> u64 {
        self.literal_count.iter().sum()
    }

    pub fn total_tokens(&self) -> u64 {
        self.token_count.iter().sum()
    }

    pub fn cost(&self, literal_cost: f64) -> f64 {
        self.token_count.iter().sum::<u64>() as f64
            + literal_cost * self.literal_count.iter().sum::<u64>() as f64
    }
}
