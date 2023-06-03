use json::JsonValue;

pub struct TokenStats {
    pub token_count: Vec<u64>,
    pub pair_count: Vec<u64>,
    pub literal_count: [u64; 256],
    pub scanned_bytes: u64,
}

impl TokenStats {
    pub fn new(ntokens: usize) -> Self {
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

    pub fn cost(&self, literal_cost: u64) -> u64 {
        self.token_count.iter().sum::<u64>()
            + literal_cost * self.literal_count.iter().sum::<u64>()
    }

    pub fn to_json(&self, initial_size: u64, literal_cost: u64, literal_dist_entropy: f64, reserved_tokens: usize) -> JsonValue {
        let total_cost = self.cost(literal_cost);
        json::object! {
            ntokens: self.token_count.len() + reserved_tokens,
            initial_size: initial_size,
            processed_size: self.scanned_bytes,
            total_cost: total_cost,
            bytes_per_token: initial_size as f64 / total_cost as f64,
            total_tokens: self.total_tokens(),
            total_literals: self.total_literals(),
            literal_dist_entropy: literal_dist_entropy,
            literal_cost: literal_cost,
            literal_entropy_per_input_byte: literal_dist_entropy * self.total_literals() as f64 / initial_size as f64
        }
    }
}
