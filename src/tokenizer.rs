use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::stats::TokenStats;
use crate::tokens::{TokenIdx, TokenSet};

#[derive(Debug)]
struct SuffixState {
    suffix: Vec<u8>,
    token_idx: TokenIdx,
    next: [usize; 256],
}

impl SuffixState {
    fn new(suffix: Vec<u8>, token_idx: TokenIdx) -> Self {
        SuffixState {
            suffix,
            token_idx,
            next: [0; 256],
        }
    }
}

struct DynState {
    cost: f64,
    token_id: TokenIdx,
}

struct Tokenizer {
    token_set: TokenSet,
    suffix_states: Vec<SuffixState>,
    // current_state: usize,
    cost_array: Vec<DynState>,
}

impl Tokenizer {
    fn new(mut token_set: TokenSet) -> Self {
        token_set.generate_suffixes();
        let suffix_states = Self::create_suffix_states(&token_set);

        Tokenizer {
            token_set,
            suffix_states,
            cost_array: Vec::new(),
        }
    }

    fn create_suffix_states(token_set: &TokenSet) -> Vec<SuffixState> {
        let mut suffix_states = Vec::new();
        let mut state_by_str: HashMap<Vec<u8>, usize> = HashMap::new();

        suffix_states.push(SuffixState::new(Vec::new(), TokenIdx::None));

        state_by_str.insert(Vec::new(), 0);

        for token in token_set.tokens.iter() {
            for end in 1..=token.string.len() {
                // The suffix is a token prefix
                let suffix = token.string[..end].to_vec();

                if state_by_str.contains_key(&suffix) {
                    continue;
                }

                let mut suffix_token = TokenIdx::Literal(suffix[suffix.len() - 1]);

                for token_start in 0..suffix.len() {
                    if let Some(&idx) = token_set.tokens_by_string.get(&suffix[token_start..]) {
                        suffix_token = TokenIdx::Token(idx);
                        break;
                    }
                }

                let suffix_state = SuffixState::new(suffix, suffix_token);

                state_by_str.insert(suffix_state.suffix.clone(), suffix_states.len());
                suffix_states.push(suffix_state);
            }
        }

        // Add literals, not covered by tokens
        for literal in 0..=255 {
            let suffix = vec![literal];
            if state_by_str.contains_key(&suffix) { continue; }
            let suffix_state = SuffixState::new(suffix, TokenIdx::Literal(literal));
            state_by_str.insert(suffix_state.suffix.clone(), suffix_states.len());
            suffix_states.push(suffix_state);
        }

        for mut state in suffix_states.iter_mut() {
            let mut suffix = state.suffix.to_vec();

            for last_byte in 0..=255 {
                suffix.push(last_byte);

                let mut suffix_id: Option<usize> = None;

                for start in 0..suffix.len() {
                    let suffix_suffix = &suffix[start..];

                    if let Some(&id) = state_by_str.get(suffix_suffix) {
                        suffix_id = Some(id);
                        break;
                    }
                }

                state.next[last_byte as usize] = suffix_id.unwrap();

                suffix.pop();
            }
        }

        suffix_states
    }

    fn process_slice(&mut self, bytes: &[u8]) -> TokenStats {
        self.cost_array.truncate(0);
        self.cost_array.push(DynState {
            cost: 0.0,
            token_id: TokenIdx::None,
        });

        let mut state = &self.suffix_states[0];

        for &byte in bytes.iter() {
            state = &self.suffix_states[state.next[byte as usize]];

            let best_dyn_state = match state.token_idx {
                TokenIdx::Literal(id) => {
                    let prev_cost = self.cost_array.last().unwrap().cost;
                    let new_cost = prev_cost + self.token_set.literal_cost;

                    DynState {
                        cost: new_cost,
                        token_id: TokenIdx::Literal(id),
                    }
                }
                TokenIdx::Token(id) => {
                    let mut token = &self.token_set.tokens[id as usize];
                    let prev_cost =
                        self.cost_array[self.cost_array.len() - token.string.len()].cost;
                    let new_cost = prev_cost + 1.0;

                    let mut best_dyn_state = DynState {
                        cost: new_cost,
                        token_id: TokenIdx::Token(id),
                    };
                    loop {
                        match token.suffix {
                            TokenIdx::Token(id) => {
                                token = &self.token_set.tokens[id as usize];
                                let prev_cost = self.cost_array
                                    [self.cost_array.len() - token.string.len()]
                                .cost;
                                let new_cost = prev_cost + 1.0;

                                if new_cost < best_dyn_state.cost {
                                    best_dyn_state.cost = new_cost;
                                    best_dyn_state.token_id = TokenIdx::Token(id);
                                }
                            }
                            TokenIdx::Literal(id) => {
                                let prev_cost = self.cost_array[self.cost_array.len() - 1].cost;
                                let new_cost = prev_cost + self.token_set.literal_cost;

                                if new_cost < best_dyn_state.cost {
                                    best_dyn_state.cost = new_cost;
                                    best_dyn_state.token_id = TokenIdx::Literal(id);
                                }
                                break;
                            }
                            TokenIdx::None => break,
                        }
                    }
                    best_dyn_state
                }
                TokenIdx::None => unreachable!(),
            };

            self.cost_array.push(best_dyn_state);
        }
        self.get_stats(&self.cost_array)
    }

    fn get_stats(&self, cost_array: &[DynState]) -> TokenStats {
        let mut token_stats =
            TokenStats::new(self.token_set.tokens.len(), self.token_set.literal_cost);

        let mut pos = cost_array.len() - 1;
        token_stats.scanned_bytes = pos as u64;

        let mut next_token_id = TokenIdx::None;

        while pos > 0 {
            let token_id = cost_array[pos].token_id;
            match token_id {
                TokenIdx::Token(id) => {
                    token_stats.token_count[id as usize] += 1;

                    if let TokenIdx::Token(next_id) = next_token_id {
                        token_stats.pair_count
                            [id as usize * self.token_set.tokens.len() + next_id as usize] += 1;
                    }
                    let token = &self.token_set.tokens[id as usize];
                    pos -= token.string.len();
                }
                TokenIdx::Literal(l) => {
                    token_stats.literal_count[l as usize] += 1;
                    pos -= 1;
                }
                TokenIdx::None => unreachable!(),
            }

            next_token_id = token_id;
        }

        token_stats
    }
}

pub const CHUNK_SIZE: usize = 16 * 1024 * 1024;

struct Job {
    data: Vec<u8>,
}

fn worker(token_set: TokenSet, jobs_rx: Arc<Mutex<Receiver<Job>>>, results_tx: Sender<TokenStats>) {
    let mut tokenizer = Tokenizer::new(token_set);

    loop {
        let data = {
            match jobs_rx.lock().unwrap().recv() {
                Ok(Job { data }) => data,
                Err(_) => break,
            }
        };

        assert!(!data.is_empty());

        results_tx.send(tokenizer.process_slice(&data)).unwrap();
    }
}

pub fn tokenize_file(token_set: &TokenSet, filename: &str) -> TokenStats {
    let nthreads = std::thread::available_parallelism().unwrap().get();
    // let nthreads = 1;

    let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(2);
    let jobs_rx_shared = Arc::new(Mutex::new(jobs_rx));
    let (results_tx, results_rx) = mpsc::channel::<TokenStats>();
    let mut file = File::open(filename).unwrap();

    let mut total_stats = TokenStats::new(token_set.tokens.len(), token_set.literal_cost);

    std::thread::scope(|s| {
        let mut join_handles = Vec::new();

        for _ in 0..nthreads {
            let jobs_rx_clone = jobs_rx_shared.clone();
            let results_tx_clone = results_tx.clone();
            join_handles
                .push(s.spawn(move || worker(token_set.clone(), jobs_rx_clone, results_tx_clone)));
        }

        let start = std::time::Instant::now();
        let mut jobs_in_flight = 0;

        loop {
            let mut buffer = Vec::new();
            buffer.resize(CHUNK_SIZE, 0);

            let read_bytes = file.read(&mut buffer).unwrap();

            if read_bytes == 0 {
                break;
            }

            buffer.truncate(read_bytes);

            jobs_tx.send(Job { data: buffer }).unwrap();
            jobs_in_flight += 1;

            for result in results_rx.try_iter() {
                total_stats.add(&result);
                jobs_in_flight -= 1;
                let elapsed = std::time::Instant::now() - start;
                if total_stats.scanned_bytes > 100000000 {
                    eprint!(
                        "\rAvg pace: {:.1} MB / s",
                        total_stats.scanned_bytes as f64 / 1000000.0 / elapsed.as_secs_f64()
                    );
                }
            }
        }

        std::mem::drop(jobs_tx);

        while jobs_in_flight > 0 {
            let result = results_rx.recv().unwrap();
            total_stats.add(&result);
            jobs_in_flight -= 1;
        }
        let elapsed = std::time::Instant::now() - start;
        if total_stats.scanned_bytes > 100000000 {
            eprintln!(
                "\rAvg pace: {:.1} MB / s",
                total_stats.scanned_bytes as f64 / 1000000.0 / elapsed.as_secs_f64()
            );
        }

        while !join_handles.is_empty() {
            join_handles.pop().unwrap().join().unwrap();
        }
    });

    total_stats
}
