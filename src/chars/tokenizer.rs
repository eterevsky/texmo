

pub struct CharsTokenizer {
    token_set: CharsTokenSet,
}

impl CharsTokenizer {
    pub fn new(token_set: CharsTokenSet) -> Self {
        CharsTokenizer { token_set }
    }

    pub fn process(&self, string: &str) -> CharTokenStats {
    }
}
