use serde_json::{json, Value};

use super::tokenset::TokenSet;

pub struct TokenStats {
    pub token_set: TokenSet,
    pub total_tokens: u64,
    pub initial_size: Option<u64>,
}

impl TokenStats {
    pub fn new(token_set: TokenSet, initial_size: Option<u64>) -> Self {
        TokenStats {
            token_set,
            total_tokens: 0,
            initial_size,
        }
    }

    pub fn ntokens(&self) -> usize {
        self.token_set.ntokens()
    }

    pub fn to_json(&self) -> Value {
        let mut result = self.token_set.to_json();

        let mut stats = json!({
            "ntokens": self.ntokens(),
            "total_tokens": self.total_tokens
        });
        if let Some(s) = self.initial_size {
            stats["initial_size"] = s.into();
            stats["bytes_per_token"] = (s as f64 / self.total_tokens as f64).into();
        }

        result["stats"] = stats;

        result
    }
}
