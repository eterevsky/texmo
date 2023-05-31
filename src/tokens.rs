use std::collections::HashMap;

use crate::stats::TokenStats;

#[derive(Clone, Copy, Debug)]
pub enum TokenIdx {
    Token(u32),
    Literal(u8),
    None,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub string: Vec<u8>,
    pub is_mandatory: bool,
    // The longest other token or literal which is a suffix of this one.
    pub suffix: TokenIdx,
}

impl Token {
    fn new(string: &[u8], is_mandatory: bool) -> Self {
        Token {
            string: string.to_vec(),
            is_mandatory,
            suffix: TokenIdx::None,
        }
    }
}

#[derive(Clone)]
enum Fallback {
    Bits(usize),
    Distribution,
    HexLiteral,
}

#[derive(Clone)]
pub struct TokenSet {
    pub tokens: Vec<Token>,
    pub tokens_by_string: HashMap<Vec<u8>, u32>,
    pub literal_cost: f64,
    fallback: Fallback,

    /// Number of tokens that are not added to the token set since they don't
    /// have a string representation.
    reserved_tokens: usize,

    /// Number of each literal in the latest tokenization. Smoothed by +1 for
    /// all non-token literals.
    literal_count: [u64; 256],
}

impl TokenSet {
    pub fn build_with_fallback_bits(bits: usize) -> Self {
        let token_set = TokenSet {
            tokens: Vec::new(),
            tokens_by_string: HashMap::new(),
            literal_cost: 8.0 / bits as f64,
            fallback: Fallback::Bits(bits),
            literal_count: [0; 256],
            reserved_tokens: 1 << bits,
        };

        token_set
    }

    pub fn build_with_dist_fallback(literal_count: [u64; 256]) -> Self {
        let token_set = TokenSet {
            tokens: Vec::new(),
            tokens_by_string: HashMap::new(),
            literal_cost: 8.0,
            fallback: Fallback::Distribution,
            literal_count,
            reserved_tokens: 1,
        };

        token_set
    }

    pub fn build_with_hex_literals() -> Self {
        let mut token_set = TokenSet {
            tokens: Vec::new(),
            tokens_by_string: HashMap::new(),
            literal_cost: 3.0,
            fallback: Fallback::HexLiteral,
            literal_count: [0; 256],
            reserved_tokens: 0,
        };

        token_set.add_mandatory_token(&[0x10]);
        for i in ('0' as u8)..=('9' as u8) {
            token_set.add_mandatory_token(&[i]);
        }
        for i in ('a' as u8)..=('f' as u8) {
            token_set.add_mandatory_token(&[i]);
        }

        token_set
    }

    pub fn ntokens(&self) -> usize {
        self.tokens.len() + self.reserved_tokens
    }

    pub fn has_dist_fallback(&self) -> bool {
        if let Fallback::Distribution = self.fallback {
            true
        } else {
            false
        }
    }

    /// A cost of a literal in tokens:
    /// The number of tokens to represent a literal + in case literal can
    /// represent different tokens, the entropy of literal / average expected
    /// entropy of a token in the model.
    pub fn dist_entropy(&self) -> f64 {
        if !self.has_dist_fallback() {
            return 0.0;
        }
        let total_literals: u64 = self.literal_count.iter().sum();
        if total_literals == 0 {
            return 0.0;
        }

        let mut entropy: f64 = 0.0;

        for b in 0..256 {
            if self.literal_count[b] > 0 {
                let fraction = self.literal_count[b] as f64 / total_literals as f64;
                entropy -= fraction * fraction.log2();
            }
        }

        entropy
    }

    pub fn update_stats(&mut self, stats: &TokenStats) {
        self.literal_count = stats.literal_count;
        let total_literals: u64 = self.literal_count.iter().sum();
        let total_tokens: u64 = stats.token_count.iter().sum();

        for b in 0..=255 {
            if !self.tokens_by_string.contains_key(&vec![b]) {
                self.literal_count[b as usize] += 1;
            }
        }

        if !self.has_dist_fallback() {
            return;
        }

        if total_tokens == 0 {
            self.literal_cost = 8.0;
            return;
        }

        // Suppose 1 byte has entropy 1 bit
        // Then 1 token = 1 / log2(ntokens) bits of entropy

        let bytes_per_token =
            (stats.scanned_bytes - total_literals) as f64 / (total_tokens as f64 + 1.0);

        self.literal_cost = 1.0 + self.dist_entropy() / bytes_per_token
    }

    fn add_mandatory_token(&mut self, string: &[u8]) {
        assert!(!self.tokens_by_string.contains_key(string));
        let index = self.tokens.len();
        let token = Token::new(string, true);
        self.tokens_by_string
            .insert(token.string.clone(), index as u32);
        self.tokens.push(token);
    }

    pub fn add_token(&mut self, string: &[u8]) {
        if let Some(&existing) = self.tokens_by_string.get(string) {
            let existing = &self.tokens[existing as usize];
            assert!(existing.is_mandatory);
            return;
        }

        let index = self.tokens.len();
        let token = Token::new(string, false);
        self.tokens_by_string
            .insert(token.string.clone(), index as u32);
        self.tokens.push(token);
    }

    pub fn remove_token(&mut self, token_str: &[u8]) {
        let token_id = *self.tokens_by_string.get(token_str).unwrap() as usize;

        assert!(!self.tokens[token_id].is_mandatory);
        self.tokens.remove(token_id);

        self.tokens_by_string.clear();
        for i in 0..self.tokens.len() {
            let token = &self.tokens[i];
            self.tokens_by_string.insert(token.string.clone(), i as u32);
        }
    }

    pub fn from_json(filename: &str) -> Self {
        let contents = std::fs::read_to_string(filename).unwrap();
        let parsed = json::parse(&contents).unwrap();

        let mut token_set = match parsed["type"].as_str().unwrap() {
            "str_with_fallback_bits" => {
                TokenSet::build_with_fallback_bits(parsed["fallback_bits"].as_usize().unwrap())
            }
            "fallback16" => TokenSet::build_with_hex_literals(),
            "fallback_distribution" => {
                let mut counts = [0; 256];
                let mut i = 0;
                for v in parsed["literal_count"].members() {
                    counts[i] = v.as_u64().unwrap();
                    i += 1;
                }
                TokenSet::build_with_dist_fallback(counts)
            }
            _ => unimplemented!(),
        };

        for token_str in parsed["tokens"].members() {
            if token_str.is_string() {
                token_set.add_token(token_str.as_str().unwrap().as_bytes());
            } else if token_str.is_array() {
                let mut s = vec![];
                for b in token_str.members() {
                    s.push(b.as_u8().unwrap());
                }
                token_set.add_token(&s);
            } // Skip fallback values
        }

        token_set
    }

    pub fn generate_suffixes(&mut self) {
        for token in self.tokens.iter_mut() {
            if token.string.len() == 1 {
                token.suffix = TokenIdx::None;
                continue;
            }

            token.suffix = TokenIdx::Literal(token.string[token.string.len() - 1]);

            for start in 1..token.string.len() {
                let suffix = &token.string[start..];
                if let Some(&idx) = self.tokens_by_string.get(suffix) {
                    token.suffix = TokenIdx::Token(idx as u32);
                    break;
                }
            }
        }
    }

    pub fn to_json(&self, stats: &TokenStats, initial_size: u64) -> json::JsonValue {
        let mut out = json::object! {
            tokens: [],
            stats: stats.to_json(initial_size, self.literal_cost, self.dist_entropy())
        };

        let mut token_strs = vec![];

        for token in self.tokens.iter() {
            token_strs.push(token.string.clone());
        }

        token_strs.sort_unstable();

        for x in 0..self.reserved_tokens {
            out["tokens"].push(x).unwrap();
        }

        for token_str in token_strs.iter() {
            let value: json::JsonValue = match std::str::from_utf8(&token_str) {
                Ok(s) => s.into(),
                Err(_) => token_str.as_slice().into(),
            };

            out["tokens"].push(value).unwrap();
        }

        match self.fallback {
            Fallback::HexLiteral => {
                out["type"] = "fallback16".into();
            }
            Fallback::Bits(b) => {
                out["type"] = "str_with_fallback_bits".into();
                out["fallback_bits"] = b.into();
            }
            Fallback::Distribution => {
                out["type"] = "fallback_distribution".into();
                out["literal_count"] = json::JsonValue::new_array();
                for &count in self.literal_count.iter() {
                    out["literal_count"].push(count).unwrap();
                }
            }
        }

        out
    }
}
