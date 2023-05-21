use std::collections::HashMap;


#[derive(Clone, Debug)]
pub struct Token {
    pub string: Vec<u8>,
    pub is_literal: bool,
    pub is_mandatory: bool,
    // Index of the longest other token which is a suffix of this one.
    pub suffix: Option<usize>,
    pub cost: usize,
}

impl Token {
    fn new(string: &[u8], is_literal: bool, is_mandatory: bool, cost: usize) -> Self {
        Token {
            string: string.to_vec(),
            is_literal,
            is_mandatory,
            suffix: None,
            cost,
        }
    }
}

#[derive(Clone)]
pub struct TokenSet {
    pub tokens: Vec<Token>,
    pub tokens_by_string: HashMap<Vec<u8>, usize>,
    literal_cost: usize,
    pub fallback16: bool,
}

impl TokenSet {
    pub fn ntokens(&self) -> usize {
        self.tokens.len()
    }

    fn add_mandatory_token(&mut self, string: &[u8]) {
        if let Some(&existing) = self.tokens_by_string.get(string) {
            let existing = &self.tokens[existing];
            assert!(existing.is_literal);
        }
        let index = self.tokens.len();
        let token = Token::new(string, false, true, 1);
        self.tokens_by_string.insert(token.string.clone(), index);
        self.tokens.push(token);
    }

    pub fn add_token(&mut self, string: &[u8]) {
        if let Some(&existing) = self.tokens_by_string.get(string) {
            let existing = &self.tokens[existing];
            assert!(existing.is_literal || existing.is_mandatory);
            if !existing.is_literal {
                return;
            }
        }

        let index = self.tokens.len();
        let token = Token::new(string, false, false, 1);
        self.tokens_by_string.insert(token.string.clone(), index);
        self.tokens.push(token);
    }

    fn add_literal(&mut self, value: u8) {
        let token = Token::new(&[value], true, false, self.literal_cost);
        self.tokens_by_string
            .insert(token.string.clone(), self.tokens.len());
        self.tokens.push(token);
    }

    pub fn remove_token(&mut self, token_str: &[u8]) {
        let token_id = *self.tokens_by_string.get(token_str).unwrap();

        assert!(token_id >= 256); // Can't remove literals
        assert!(!self.tokens[token_id].is_literal);
        assert!(!self.tokens[token_id].is_mandatory);
        self.tokens.remove(token_id);

        self.tokens_by_string.clear();
        for i in 0..self.ntokens() {
            let token = &self.tokens[i];
            self.tokens_by_string.insert(token.string.clone(), i);
        }
    }

    pub fn build_with_hex_literals() -> Self {
        let mut token_set = TokenSet {
            tokens: Vec::new(),
            tokens_by_string: HashMap::new(),
            literal_cost: 3,
            fallback16: true,
        };

        for i in 0..=255 {
            token_set.add_literal(i);
        }
        token_set.add_mandatory_token(&[0x10]);
        for i in ('0' as u8)..=('9' as u8) {
            token_set.add_mandatory_token(&[i]);
        }
        for i in ('a' as u8)..=('f' as u8) {
            token_set.add_mandatory_token(&[i]);
        }

        token_set
    }

    pub fn build_with_bin_literals() -> Self {
        let mut token_set = TokenSet {
            tokens: Vec::new(),
            tokens_by_string: HashMap::new(),
            literal_cost: 8,
            fallback16: false,
        };

        for i in 0..=255 {
            token_set.add_literal(i);
        }
        token_set.add_mandatory_token(&[0x11]);
        token_set.add_mandatory_token(&[0x12]);

        token_set
    }

    pub fn from_json(filename: &str) -> Self {
        let contents = std::fs::read_to_string(filename).unwrap();
        let parsed = json::parse(&contents).unwrap();

        let mut token_set = if parsed["config"]["fallback16"].as_bool().unwrap() {
            TokenSet::build_with_hex_literals()
        } else {
            TokenSet::build_with_bin_literals()
        };

        for token_str in parsed["tokens"].members() {
            if token_str.is_string() {
                token_set.add_token(token_str.as_str().unwrap().as_bytes());
            } else {
                let mut s = vec![];
                for b in token_str.members() {
                    s.push(b.as_u8().unwrap());
                }
                token_set.add_token(&s);
            }
        }

        token_set

    }

    pub fn generate_suffixes(&mut self) {
        for i in 256..self.tokens.len() {
            let mut token = &mut self.tokens[i];
            for start in 1..token.string.len() {
                if let Some(&idx) = self.tokens_by_string.get(&token.string[start..]) {
                    token.suffix = Some(idx);
                    break;
                }
            }
        }
    }

    pub fn to_json(&self) -> json::JsonValue {
        let mut out = json::object! {
            tokens: []
        };

        let mut token_strs = vec![];

        for token in self.tokens.iter() {
            if !token.is_literal {
                token_strs.push(token.string.clone());
            }
        }

        token_strs.sort_unstable();

        for token_str in token_strs.iter() {
            let value: json::JsonValue = match std::str::from_utf8(&token_str) {
                Ok(s) => s.into(),
                Err(_) => token_str.as_slice().into(),
            };

            out["tokens"].push(value).unwrap();
        }

        out["config"]["fallback16"] = self.fallback16.into();

        out
    }
}
