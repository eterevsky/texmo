use super::tokens::{CharsToken, CharsTokenSet};

const MAX_UNICODE: usize = 0x110000;

pub struct CharsTokenStats {
    token_set: CharsTokenSet,
    total_tokens_count: u64,
    literals_count: u64,
}

impl CharsTokenStats {
    pub fn new(token_set: CharsTokenSet) -> Self {
        CharsTokenStats {
            token_set,
            total_tokens_count: 0,
            literals_count: 0,
        }
    }

    pub fn merge(&mut self, other: &CharsTokenStats) {
        self.total_tokens_count += other.total_tokens_count;
        self.literals_count += other.literals_count;
    }

    pub fn total_tokens(&self) -> u64 { self.total_tokens_count }

    pub fn total_literals(&self) -> u64 { self.literals_count }

    // Count a token 
    pub fn count_token(&mut self, idx: usize) {
        self.total_tokens_count += 1;
        let token = &self.token_set.tokens[idx];
        if let CharsToken::Char(_) = token {
            self.literals_count += 1;
        }
    }

    // Count a literal which is _not_ covered by a single token.
    pub fn count_literal(&mut self, ch: char) {
        let cost = self.token_set.char_cost(ch);
        self.total_tokens_count += cost as u64;
        self.literals_count += 1;
    }
}
