use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use clap::{Parser, Subcommand};

mod optimizer;
mod sampler;
mod stats;
mod tokenizer;
mod tokens;

use crate::sampler::SelectionSampler;

use self::optimizer::optimize_bpe;
use self::sampler::{FileSampler, MemorySampler, Sampler};
use self::stats::TokenStats;
use self::tokenizer::tokenize_file;
use self::tokens::{LiteralEncoding, TokenSet};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Processing {
    Raw,
    Caps,
    CapsWords,
}

impl fmt::Display for Processing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                Processing::Raw => "raw",
                Processing::Caps => "caps",
                Processing::CapsWords => "capswords",
            }
        )
    }
}

fn optimize_tokens(
    data_filename: &str,
    input_tokens: &Option<String>,
    output_tokens: &Option<String>,
    ntokens: usize,
    initial_size: Option<u64>,
    processing: &Option<String>,
    in_memory: bool,
    nchunks: Option<usize>,
    chunk_size: usize,
    add_block: usize,
    literal_encoding: LiteralEncoding,
) {
    println!(
        "Using {} threads",
        std::thread::available_parallelism().unwrap().get()
    );

    let token_set = if let Some(tokens_file) = input_tokens {
        TokenSet::from_json(tokens_file.as_str())
    } else {
        TokenSet::new(literal_encoding)
    };

    let file_size = std::fs::metadata(data_filename).unwrap().len();
    let initial_size = initial_size.unwrap_or(file_size);

    let fast_sampler = SelectionSampler::new(data_filename, 16384, 1024);

    let (token_set, token_stats) = if nchunks.is_some() {
        let nchunks = nchunks.unwrap();
        let sampler = SelectionSampler::new(data_filename, chunk_size, nchunks);
        let (token_set, _) = optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block);

        let full_sampler = FileSampler::new(data_filename, chunk_size);
        let stats = tokenize_file(&token_set, &full_sampler);

        (token_set, stats)
    } else if in_memory {
        let sampler = MemorySampler::new(data_filename, chunk_size);
        optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block)
    } else {
        let sampler = FileSampler::new(data_filename, chunk_size);
        optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block)
    };

    let mut tokens_json = token_set.to_json(&token_stats, initial_size);

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

fn optimize_token_set<'a, S1: Sampler<'a>, S2: Sampler<'a>>(
    prev_token_set: Option<TokenSet>,
    fast_sampler: &'a SelectionSampler,
    sampler: &'a S1,
    slow_sampler: &'a S2,
    ntokens: usize,
    literal_encoding: LiteralEncoding,
    add_block: usize,
) -> (TokenSet, TokenStats) {
    let token_set = if let Some(ts) = prev_token_set {
        ts
    } else {
        TokenSet::new(literal_encoding)
    };

    let (token_set, _) = optimize_bpe(&token_set, ntokens, sampler, fast_sampler, add_block);

    let stats = tokenize_file(&token_set, slow_sampler);

    (token_set, stats)
}

fn optimize_all_for_proc(
    filename: &str,
    processing: Processing,
    initial_size: u64,
    output_dir: &str,
    min_tokens: usize,
    max_tokens: usize,
) {
    println!(
        "Optimizing for proc '{}', initial size {}",
        processing, initial_size
    );
    let fast_sampler = SelectionSampler::new(filename, 16384, 1024);
    let sampler = SelectionSampler::new(filename, 1 << 20, 1 << 14);
    let slow_sampler = FileSampler::new(filename, 1 << 24);
    let output_dir = Path::new(output_dir);

    for &literal_encoding in &[
        // LiteralEncoding::All,
        // LiteralEncoding::Bits1,
        // LiteralEncoding::Bits2,
        // LiteralEncoding::Bits4,
        LiteralEncoding::Dist2,
        LiteralEncoding::Dist4,
        // LiteralEncoding::Dist8,
    ] {
        // if processing == Processing::Raw && (literal_encoding == LiteralEncoding::Bits1 || literal_encoding == LiteralEncoding::Bits2) {
        //     continue;
        // }

        let mut ntokens = min_tokens;
        let mut prev_token_set = None;
        while ntokens <= max_tokens {
            if literal_encoding.reserved_tokens() <= ntokens
                && (literal_encoding != LiteralEncoding::All || ntokens >= 256)
                && (processing == Processing::Raw || ntokens >= 16)
            {
                let mut block = ntokens / 2;
                while block * block >= ntokens {
                    block /= 2;
                }

                println!(
                    "Optimizing tokens for processing '{}', literals {}, ntokens {}, block {}",
                    processing, literal_encoding, ntokens, block
                );

                let (token_set, stats) = optimize_token_set(
                    prev_token_set,
                    &fast_sampler,
                    &sampler,
                    &slow_sampler,
                    ntokens,
                    literal_encoding,
                    block,
                );

                let mut tokens_json = token_set.to_json(&stats, initial_size);
                tokens_json["processing"] = processing.to_string().into();

                println!(
                    "{}",
                    json::stringify_pretty(tokens_json["stats"].clone(), 2)
                );

                let output_filename =
                    format!("tokens{}_{}_{}.json", ntokens, processing, literal_encoding);
                let output_path = output_dir.join(output_filename);
                println!("Writing to {}", output_path.display());

                std::fs::write(&output_path, json::stringify_pretty(tokens_json, 2)).unwrap();

                prev_token_set = Some(token_set);
            }

            ntokens *= 2;
        }
    }
}

fn optimize_all(
    filename: &str,
    filename_caps: &str,
    filename_caps_words: &str,
    output_dir: &str,
    min_tokens: usize,
    max_tokens: usize,
) {
    let initial_size = std::fs::metadata(filename).unwrap().len();

    optimize_all_for_proc(
        filename,
        Processing::Raw,
        initial_size,
        output_dir,
        min_tokens,
        max_tokens,
    );
    optimize_all_for_proc(
        filename_caps,
        Processing::Caps,
        initial_size,
        output_dir,
        min_tokens,
        max_tokens,
    );
    optimize_all_for_proc(
        filename_caps_words,
        Processing::CapsWords,
        initial_size,
        output_dir,
        min_tokens,
        max_tokens,
    );
}

fn filter_text(filename: &str, caps: bool, words: bool, output: &str) {
    let input = File::open(filename).unwrap();
    let mut output = File::create(output).unwrap();

    let reader = BufReader::with_capacity(1 << 20, input);
    let mut writer = BufWriter::new(&mut output);

    let mut word = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let mut out_line = Vec::new();
        let mut hanging_space = false;
        word.clear();

        for ch in line.chars() {
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

pub const CHUNK_SIZE: usize = 16 * 1024 * 1024;

fn count_bytes(filename: &str) -> [u64; 256] {
    let mut file = File::open(filename).unwrap();
    let mut buffer = Vec::new();
    buffer.resize(CHUNK_SIZE, 0);

    let mut counts = [0; 256];

    let mut read_bytes = file.read(&mut buffer).unwrap();
    while read_bytes > 0 {
        for byte in buffer[..read_bytes].iter() {
            counts[*byte as usize] += 1;
        }
        read_bytes = file.read(&mut buffer).unwrap();
    }

    counts
}

fn count_bytes_command(filename: &str) {
    let counts = count_bytes(filename);

    for c in counts.iter() {
        print!(" {}", c);
    }
    println!();
}

fn tokenize(
    filename: &str,
    tokens_file: &str,
    initial_size: u64,
    chunk_size: usize,
    in_memory: bool,
) {
    let token_set = TokenSet::from_json(tokens_file);

    let stats = if in_memory {
        let sampler = MemorySampler::new(filename, chunk_size);
        tokenize_file(&token_set, &sampler)
    } else {
        let sampler = FileSampler::new(filename, chunk_size);
        tokenize_file(&token_set, &sampler)
    };

    let stats_json = stats.to_json(
        initial_size,
        token_set.literal_cost(),
        token_set.literal_encoding.tokens_in_literal(),
        token_set.dist_entropy(),
        token_set.reserved_tokens(),
    );
    println!("{}", json::stringify_pretty(stats_json, 2));
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

        /// If the input is pre-processed, this argument specifies the size
        /// of the input before pre-processing.
        #[arg(long)]
        initial_size: u64,

        #[arg(long, default_value_t = 1024 * 1024)]
        chunk_size: usize,

        #[arg(long)]
        in_memory: bool,
    },

    OptimizeTokens {
        #[arg(short, long)]
        input_tokens: Option<String>,

        #[arg(short, long)]
        output_tokens: Option<String>,

        #[arg(short, long)]
        ntokens: usize,

        #[arg(long, default_value_t=LiteralEncoding::Dist8)]
        literals: LiteralEncoding,

        /// If the input is pre-processed, this argument specifies the size
        /// of the input before pre-processing.
        #[arg(long)]
        initial_size: Option<u64>,

        /// A comma-separated list of the processing stages applied to the input
        /// for the purposes of including it into the output.
        #[arg(long)]
        processing: Option<String>,

        #[arg(long, default_value_t=16 * 1024 * 1024)]
        chunk_size: usize,

        /// Sample this number of training data chunks, keep them in memory
        /// and train the token set on them. Final stats will be calculated
        /// the whole dataset.
        #[arg(long)]
        nchunks: Option<usize>,

        /// Use in-memory sampler in case all training data fits in memory.
        /// Ignored when nchunks is specified.
        #[arg(long)]
        in_memory: bool,

        /// How many tokens will be added after each pass.
        #[arg(long, default_value_t = 1)]
        add_block: usize,
    },

    OptimizeAll {
        #[arg(long)]
        filename_caps: String,

        #[arg(long)]
        filename_caps_words: String,

        #[arg(short, long)]
        output_dir: String,

        #[arg(long, default_value_t = 2)]
        min_tokens: usize,

        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
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
    CountBytes,
}

fn main() {
    let args = Args::parse();
    let filename = args.data.as_str();

    match &args.command {
        Command::Tokenize {
            tokens,
            initial_size,
            chunk_size,
            in_memory,
        } => tokenize(filename, tokens, *initial_size, *chunk_size, *in_memory),

        Command::OptimizeTokens {
            input_tokens,
            output_tokens,
            ntokens,
            literals,
            initial_size,
            processing,
            in_memory,
            nchunks,
            chunk_size,
            add_block,
        } => optimize_tokens(
            filename,
            input_tokens,
            output_tokens,
            *ntokens,
            *initial_size,
            processing,
            *in_memory,
            *nchunks,
            *chunk_size,
            *add_block,
            *literals,
        ),

        Command::OptimizeAll {
            filename_caps,
            filename_caps_words,
            output_dir,
            min_tokens,
            max_tokens,
        } => optimize_all(
            filename,
            filename_caps.as_str(),
            filename_caps_words.as_str(),
            output_dir.as_str(),
            *min_tokens,
            *max_tokens,
        ),

        Command::FilterText {
            caps,
            words,
            output,
        } => filter_text(filename, *caps, *words, output.as_str()),

        Command::CountHexDigits => count_hex_digits(filename),

        Command::CountBytes => count_bytes_command(filename),
    }
}
