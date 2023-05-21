use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};

mod tokens;

use self::tokens::TokenSet;

#[derive(Debug)]
struct SuffixState {
    suffix: Vec<u8>,
    token_id: usize,
    next: [usize; 256],
}

impl SuffixState {
    fn new(suffix: Vec<u8>, token_id: usize) -> Self {
        SuffixState {
            suffix,
            token_id,
            next: [0; 256],
        }
    }
}

struct DynState {
    cost: usize,
    token_id: usize,
}

struct TokenStats {
    token_count: Vec<usize>,
    pair_count: Vec<usize>,
    cost: usize,
    scanned_bytes: usize,
}

impl TokenStats {
    fn new(token_set: &TokenSet) -> Self {
        let n = token_set.tokens.len();

        let mut token_count = Vec::with_capacity(n);
        token_count.resize(n, 0);

        let mut pair_count = Vec::with_capacity(n * n);
        pair_count.resize(n * n, 0);

        TokenStats {
            token_count,
            pair_count,
            cost: 0,
            scanned_bytes: 0,
        }
    }

    fn add(&mut self, other: &TokenStats) {
        for i in 0..self.token_count.len() {
            self.token_count[i] += other.token_count[i];
        }
        for i in 0..self.pair_count.len() {
            self.pair_count[i] += other.pair_count[i];
        }
        self.cost += other.cost;
        self.scanned_bytes += other.scanned_bytes;
    }

    fn token_to_add(&self, token_set: &TokenSet) -> Vec<u8> {
        let mut top_literal = 0;
        let mut top_literal_count = 0;

        for i in 0..256 {
            if self.token_count[i] > top_literal_count {
                top_literal = i;
                top_literal_count = self.token_count[i];
            }
        }

        // println!(
        //     "Top literal: {} with count {}",
        //     top_literal, top_literal_count
        // );

        let mut top_pair = 0;
        let mut top_pair_count = 0;
        for ipair in 0..self.pair_count.len() {
            if self.pair_count[ipair] > top_pair_count {
                top_pair = ipair;
                top_pair_count = self.pair_count[ipair];
            }
        }

        let ifirst = top_pair / token_set.ntokens();
        let isecond = top_pair % token_set.ntokens();

        let mut token_str = token_set.tokens[ifirst].string.clone();
        token_str.extend(token_set.tokens[isecond].string.clone());

        if top_literal_count * (token_set.literal_cost - 1) > top_pair_count {
            vec![top_literal as u8]
        } else {
            token_str
        }
    }
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
            // current_state: 0,
            // cost_array: vec![DynState {
            //     cost: 0,
            //     token_id: 0x13,
            // }],
        }
    }

    fn create_suffix_states(token_set: &TokenSet) -> Vec<SuffixState> {
        let mut suffix_states = Vec::new();
        let mut state_by_str: HashMap<Vec<u8>, usize> = HashMap::new();

        suffix_states.push(SuffixState::new(Vec::new(), 0));

        state_by_str.insert(Vec::new(), 0);

        for token in token_set.tokens.iter() {
            for end in 1..=token.string.len() {
                // The suffix is a token prefix
                let suffix = token.string[..end].to_vec();

                if state_by_str.contains_key(&suffix) {
                    continue;
                }

                let mut suffix_token = None;
                for token_start in 0..suffix.len() {
                    if let Some(&idx) = token_set.tokens_by_string.get(&suffix[token_start..]) {
                        suffix_token = Some(idx);
                        break;
                    }
                }

                assert!(suffix_token.is_some());

                let suffix_state = SuffixState::new(suffix, suffix_token.unwrap());

                let state_idx = suffix_states.len();
                state_by_str.insert(suffix_state.suffix.to_vec(), state_idx);
                suffix_states.push(suffix_state);
            }
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
            cost: 0,
            token_id: 0x13,
        });

        let mut state = &self.suffix_states[0];

        for &byte in bytes.iter() {
            state = &self.suffix_states[state.next[byte as usize]];

            let mut best_dyn_state = DynState {
                cost: std::usize::MAX,
                token_id: 0,
            };

            let mut token_id = state.token_id;
            loop {
                let token = &self.token_set.tokens[token_id];
                let prev_cost = self.cost_array[self.cost_array.len() - token.string.len()].cost;
                let new_cost = prev_cost + token.cost;

                if new_cost < best_dyn_state.cost {
                    best_dyn_state.cost = new_cost;
                    best_dyn_state.token_id = token_id;
                }

                if let Some(t) = token.suffix {
                    token_id = t;
                } else {
                    break;
                }
            }

            self.cost_array.push(best_dyn_state);
        }
        self.get_stats(&self.cost_array)
    }

    fn get_stats(&self, cost_array: &[DynState]) -> TokenStats {
        let mut token_stats = TokenStats::new(&self.token_set);

        let mut pos = cost_array.len() - 1;
        token_stats.cost = cost_array[pos].cost;
        token_stats.scanned_bytes = pos;

        let mut next_token_id = 0;

        while pos > 0 {
            let token_id = cost_array[pos].token_id;
            let token = &self.token_set.tokens[token_id];
            token_stats.token_count[token_id] += 1;

            token_stats.pair_count[token_id * self.token_set.ntokens() + next_token_id] += 1;

            next_token_id = token_id;
            pos = pos.checked_sub(token.string.len()).unwrap();
        }

        token_stats
    }
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;

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

fn tokenize_file(token_set: &TokenSet, filename: &str) -> TokenStats {
    let nthreads = std::thread::available_parallelism().unwrap().get();

    let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(2);
    let jobs_rx_shared = Arc::new(Mutex::new(jobs_rx));
    let (results_tx, results_rx) = mpsc::channel::<TokenStats>();
    let mut file = File::open(filename).unwrap();

    let mut total_stats = TokenStats::new(token_set);

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

fn format_token(s: &[u8]) -> String {
    match String::from_utf8(s.to_vec()) {
        Ok(string) => format!("{:?}", string),
        Err(_) => format!("{:?}", s),
    }
}

fn optimize_bpe(token_set: &TokenSet, ntokens: usize, filename: &str) -> (TokenSet, TokenStats) {
    let mut token_set = token_set.clone();
    let mut prev_stats = None;

    loop {
        let initial_stats = match prev_stats {
            Some(s) => s,
            None => tokenize_file(&token_set, filename),
        };

        let new_token_str = initial_stats.token_to_add(&token_set);
        let mut new_token_set = token_set.clone();
        new_token_set.add_token(&new_token_str);
        prev_stats = None;

        println!(
            "{} Adding token {}",
            initial_stats.cost,
            format_token(&new_token_str)
        );

        if new_token_set.ntokens() > 256 + ntokens {
            let stats = tokenize_file(&new_token_set, filename);
            let mut token_ids: Vec<usize> = (0..new_token_set.tokens.len()).collect();
            token_ids.sort_unstable_by_key(|&i| {
                let token = &new_token_set.tokens[i];
                if token.string.len() == 1 {
                    stats.token_count[i] * (new_token_set.literal_cost - 1)
                } else {
                    stats.token_count[i]
                }
            });

            let mut token_strs = Vec::new();

            for &token_id_to_remove in token_ids.iter() {
                let token_to_remove = &new_token_set.tokens[token_id_to_remove];
                if token_to_remove.is_literal || token_to_remove.is_mandatory {
                    continue;
                }
                // Because the ids will change when we are removing and adding
                // tokens.
                token_strs.push(token_to_remove.string.clone());
            }

            let mut found = false;
            let mut tries = 0;

            for token_str in token_strs.iter() {
                let token_str = token_str.as_slice();
                tries += 1;

                new_token_set.remove_token(token_str);

                let stats = tokenize_file(&new_token_set, filename);

                // if stats.cost < initial_stats.cost
                //     && stats.token_to_add(&new_token_set) == token_str
                // {
                //     println!("Cost after removing {} would be {}, but it would be added on the next iteration.", format_token(&token_str), stats.cost);
                // }

                if stats.cost < initial_stats.cost
                // && stats.token_to_add(&new_token_set) != token_str
                {
                    // Found a token to remove.
                    found = true;
                    prev_stats = Some(stats);
                    println!(
                        "Removing token {} after {} tries",
                        format_token(&token_str),
                        tries
                    );
                    break;
                }

                new_token_set.add_token(&token_str);
            }

            if !found {
                return (token_set, initial_stats);
            }
        }

        token_set = new_token_set;
    }
}

fn optimize_tokens(
    data_filename: &str,
    input_tokens: &Option<String>,
    output_tokens: &Option<String>,
    ntokens: usize,
    fallback2: bool,
    initial_size: Option<usize>,
    processing: &Option<String>,
) {
    println!(
        "Using {} threads",
        std::thread::available_parallelism().unwrap().get()
    );

    let token_set = if let Some(tokens_file) = input_tokens {
        TokenSet::from_json(tokens_file.as_str())
    } else {
        if fallback2 {
            TokenSet::build_with_bin_literals()
        } else {
            TokenSet::build_with_hex_literals()
        }
    };

    let tokens_json = token_set.to_json();
    println!(
        "Initial token set:\n{}",
        json::stringify_pretty(tokens_json, 2)
    );

    let (token_set, token_stats) = optimize_bpe(&token_set, ntokens, data_filename);

    let mut tokens_json = token_set.to_json();

    if token_set.fallback16 {
        tokens_json["type"] = "fallback16".into();
    } else {
        tokens_json["type"] = "fallback2".into();
    }

    tokens_json["stats"]["ntokens"] = (token_set.ntokens() - 256).into();
    let initial_size = match initial_size {
        Some(x) => x,
        None => token_stats.scanned_bytes,
    };
    tokens_json["stats"]["initial_size"] = initial_size.into();
    tokens_json["stats"]["processed_size"] = token_stats.scanned_bytes.into();
    tokens_json["stats"]["total_tokens"] = token_stats.cost.into();
    tokens_json["stats"]["bytes_per_token"] =
        (initial_size as f64 / token_stats.cost as f64).into();

    let mut processing_json: Vec<json::JsonValue> = Vec::new();
    if let Some(processing) = processing {
        for stage in processing.split(",") {
            if !stage.is_empty() {
                processing_json.push(stage.into());
            }
        }
    }

    tokens_json["processing"] = processing_json.into();

    let tokens_json_str = json::stringify_pretty(tokens_json, 2);
    println!("{}", &tokens_json_str);

    if let Some(out) = output_tokens {
        std::fs::write(&out, &tokens_json_str).unwrap();
    }
}

fn filter_text(filename: &str, caps: bool, words: bool, output: &str) {
    let input = File::open(filename).unwrap();
    let mut output = File::create(output).unwrap();

    let reader = BufReader::new(input);
    let mut writer = BufWriter::new(&mut output);

    let mut word = Vec::new();

    for line in reader.lines() {
        // println!("{:?}", line);
        let line = line.unwrap();
        let mut out_line = Vec::new();
        let mut hanging_space = false;
        word.clear();

        for ch in line.chars() {
            // println!("{:?}", ch);
            if ch.is_alphabetic() {
                hanging_space = false;
                word.push(ch);
            } else {
                if !word.is_empty() {
                    if caps {
                        if word[0].is_uppercase() && word[1..].iter().all(|&c| c.is_lowercase()) {
                            out_line.push('\x14');
                            word[0] = word[0].to_lowercase().next().unwrap();
                        } else if word.iter().all(|&c| c.is_uppercase()) {
                            out_line.push('\x15');
                            for i in 0..word.len() {
                                word[i] = word[i].to_lowercase().next().unwrap();
                            }
                        }
                    }

                    out_line.extend(word.iter());
                    word.clear();

                    if words {
                        out_line.push('\x16');
                    }

                    if words && ch == ' ' {
                        hanging_space = true;
                    } else {
                        out_line.push(ch);
                        hanging_space = false;
                    }
                } else {
                    if hanging_space {
                        out_line.push(' ');
                    }
                    hanging_space = false;
                    out_line.push(ch);
                }
            }
        }

        if !word.is_empty() {
            if caps {
                if word[0].is_uppercase() && word[1..].iter().all(|&c| c.is_lowercase()) {
                    out_line.push('\x14');
                    word[0] = word[0].to_lowercase().next().unwrap();
                } else if word.iter().all(|&c| c.is_uppercase()) {
                    out_line.push('\x15');
                    for i in 0..word.len() {
                        word[i] = word[i].to_lowercase().next().unwrap();
                    }
                }
            }

            out_line.extend(word.iter());

            if words {
                out_line.push('\x16');
            }
        }

        if hanging_space {
            out_line.push(' ');
        }

        out_line.push('\n');
        let output_string: String = out_line.iter().collect();

        writer.write_all(output_string.as_bytes()).unwrap();
    }
}

fn count_hex_digits(filename: &str) {
    let input = File::open(filename).unwrap();
    let reader = BufReader::new(input);

    let mut counts: [usize; 16] = [0; 16];

    for byte in reader.bytes() {
        let byte = byte.unwrap();
        counts[(byte >> 4) as usize] += 1;
        counts[(byte & 15) as usize] += 1;
    }

    let mut digits = (0..16).collect::<Vec<usize>>();
    digits.sort_unstable_by_key(|&d| counts[d]);

    for &d in digits.iter() {
        println!("{:x}  {}", d, counts[d]);
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    data: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Tokenize {
        #[arg(short, long)]
        tokens: String,
    },

    OptimizeTokens {
        #[arg(short, long)]
        input_tokens: Option<String>,

        #[arg(short, long)]
        output_tokens: Option<String>,

        #[arg(short, long)]
        ntokens: usize,

        /// If true, the tokens that are not present will be encoded by bits
        /// rather then as hexadecimal. Only taken into account if
        /// --input_tokens is not specified.
        #[arg(long)]
        fallback2: bool,

        /// If the input is pre-processed, this argument specifies the size
        /// of the input before pre-processing.
        #[arg(long)]
        initial_size: Option<usize>,

        /// A comma-separated list of the processing stages applied to the input
        /// for the purposes of including it into the output.
        #[arg(long)]
        processing: Option<String>,
    },

    FilterText {
        #[arg(short, long)]
        caps: bool,

        #[arg(short, long)]
        words: bool,

        #[arg(short, long)]
        output: String,
    },

    CountHexDigits,
}

fn main() {
    let args = Args::parse();
    let filename = args.data.as_str();

    match &args.command {
        Command::Tokenize { tokens } => {
            unimplemented!();
        }
        Command::OptimizeTokens {
            input_tokens,
            output_tokens,
            ntokens,
            fallback2,
            initial_size,
            processing,
        } => optimize_tokens(
            filename,
            input_tokens,
            output_tokens,
            *ntokens,
            *fallback2,
            *initial_size,
            processing,
        ),

        Command::FilterText {
            caps,
            words,
            output,
        } => filter_text(filename, *caps, *words, output.as_str()),

        Command::CountHexDigits => count_hex_digits(filename),
    }
}
