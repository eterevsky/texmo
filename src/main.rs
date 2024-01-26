use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, BufRead};

use clap::{Parser, Subcommand};

use tempfile::NamedTempFile;

mod chars;
mod optimizer;
mod processing;
mod input;
mod stats;
mod tokenizer;
mod tokens;

use self::optimizer::optimize_bpe;
use self::processing::process_file;
use self::input::memory_sampler::MemorySampler; 
use self::input::file_sampler::FileSampler;
use self::input::preloaded_sampler::PreloadedSampler;
use self::tokenizer::tokenize_file;
use self::tokens::{LiteralEncoding, TokenSet};
use self::chars::optimize_chars_tokens;

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

    let fast_sampler = PreloadedSampler::new(data_filename, 16384, 1024);

    let (token_set, token_stats) = if nchunks.is_some() {
        let nchunks = nchunks.unwrap();
        let sampler = PreloadedSampler::new(data_filename, chunk_size, nchunks);
        let (token_set, _) = optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block);

        let full_sampler = FileSampler::new(data_filename, chunk_size, None);
        let stats = tokenize_file(&token_set, &full_sampler, false);

        (token_set, stats)
    } else if in_memory {
        let sampler = MemorySampler::from_file(data_filename, chunk_size);
        optimize_bpe(&token_set, ntokens, &sampler, &fast_sampler, add_block)
    } else {
        let sampler = FileSampler::new(data_filename, chunk_size, None);
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

fn optimize_all_for_proc(
    filename_raw: &str,
    filename_processed: Option<&str>,
    processing: Processing,
    tokens_dir: &str,
    min_tokens: usize,
    max_tokens: usize,
) {
    let initial_size = std::fs::metadata(filename_raw).unwrap().len();
    println!("Optimizing for proc '{}'", processing);

    let (filename, _temp) = match filename_processed {
        Some(f) => (f.to_string(), None),
        None => match processing {
            Processing::Raw => (filename_raw.to_string(), None),
            Processing::Caps => unimplemented!("Caps processing no longer supported"),
            Processing::CapsWords => {
                println!("Pre-processing the data file... ");
                let mut temp_processed = NamedTempFile::new().unwrap();
                let mut input = File::open(filename_raw).unwrap();
                process_file(&mut input, &mut temp_processed).unwrap();
                println!("done");
                let filename = temp_processed.path().to_str().unwrap().to_string();
                (filename, Some(temp_processed))
            }
        },
    };

    if initial_size < 1 << 24 {
        let fast_sampler = MemorySampler::from_file(&filename, 16384);
        let sampler = MemorySampler::from_file(&filename, 1 << 20);
        let slow_sampler = MemorySampler::from_file(&filename, 1 << 24);

        optimizer::optimize_all(
            initial_size,
            &slow_sampler,
            &sampler,
            &fast_sampler,
            tokens_dir,
            min_tokens,
            max_tokens,
            &processing.to_string(),
        );
    } else if initial_size < 1 << 32 {
        let fast_sampler = PreloadedSampler::new(&filename, 16384, 1024);
        let sampler = MemorySampler::from_file(&filename, 1 << 24);
        let slow_sampler = FileSampler::new(&filename, 1 << 24, None);

        optimizer::optimize_all(
            initial_size,
            &slow_sampler,
            &sampler,
            &fast_sampler,
            tokens_dir,
            min_tokens,
            max_tokens,
            &processing.to_string(),
        );
    } else {
        let fast_sampler = FileSampler::new(
            &filename, 131072, Some(1024));
        // let sampler = FileSampler::new(&filename, 1 << 20, Some(4096));
        let sampler = PreloadedSampler::new(&filename, 1 << 20, 1 << 14);
        let slow_sampler = FileSampler::new(&filename, 1 << 24, None);

        optimizer::optimize_all(
            initial_size,
            &slow_sampler,
            &sampler,
            &fast_sampler,
            tokens_dir,
            min_tokens,
            max_tokens,
            &processing.to_string(),
        );
    }
}

fn optimize_all(
    filename: &str,
    filename_caps_words: Option<&str>,
    tokens_dir: &str,
    min_tokens: usize,
    max_tokens: usize,
) {
    optimize_all_for_proc(
        filename,
        Some(filename),
        Processing::Raw,
        tokens_dir,
        min_tokens,
        std::cmp::min(max_tokens, 256),
    );
    optimize_all_for_proc(
        filename,
        filename_caps_words,
        Processing::CapsWords,
        tokens_dir,
        min_tokens,
        max_tokens,
    );
}

fn process(filename: &str, output: &str) {
    let mut input = File::open(filename).unwrap();
    let mut output = File::create(output).unwrap();

    process_file(&mut input, &mut output).unwrap();
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

fn count_chars(filename: &str) {
    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);

    let mut total = 0;
    let mut counts_low: [usize; 256] = [0; 256];
    let mut counts = HashMap::new();

    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
        let chars_in_line = line.chars().count();
        let next_total = total + chars_in_line;
        if total / 1000000000 != next_total / 1000000000 {
            println!("{}", next_total);
        }
        total = next_total;

        for c in line.chars() {
            let c = c as u32;
            if c < 256 {
                counts_low[c as usize] += 1;
            } else {
                *counts.entry(c).or_insert(0) += 1;
            }
        }
    }
    println!("Total: {}", total);

    for (c, count) in counts_low.iter().enumerate() {
        if *count > 0 {
            println!("{:?} {}", std::char::from_u32(c as u32).unwrap(), count);
        }
    }

    let mut max_c = 0;
    for c in counts.keys() {
        if *c > max_c {
            max_c = *c;
        }
    }

    println!("Max char: {:?}", std::char::from_u32(max_c).unwrap());
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
        let sampler = MemorySampler::from_file(filename, chunk_size);
        tokenize_file(&token_set, &sampler, false)
    } else {
        let sampler = FileSampler::new(filename, chunk_size, None);
        let stats = tokenize_file(&token_set, &sampler, false);
        stats
    };

    let stats_json = stats.to_json(
        initial_size,
        token_set.literal_encoding.tokens_in_literal(),
        token_set.dist_entropy(&stats),
        token_set.reserved_tokens(),
    );
    println!("{}", json::stringify_pretty(stats_json, 2));
}

#[derive(Parser, Debug)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Tokenize {
        #[arg(short, long)]
        data: String,

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
        data: String,

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

    OptimizeCharsTokens {
        #[arg(short, long)]
        data: String,

        #[arg(short, long)]
        ntokens: usize, 

        #[arg(short, long)]
        output: String,
    },

    OptimizeAll {
        #[arg(short, long)]
        data: String,

        // #[arg(long)]
        // data_caps: Option<String>,
        #[arg(long)]
        data_caps_words: Option<String>,

        #[arg(short, long)]
        tokens_dir: String,

        #[arg(long, default_value_t = 2)]
        min_tokens: usize,

        #[arg(long, default_value_t = 16384)]
        max_tokens: usize,
    },

    Process {
        #[arg(short, long)]
        data: String,

        #[arg(short, long)]
        output: String,
    },

    CountHexDigits {
        #[arg(short, long)]
        data: String,
    },
    CountBytes {
        #[arg(short, long)]
        data: String,
    },
    CountChars {
        #[arg(short, long)]
        data: String,
    },
}

fn main() {
    let args = Args::parse();

    match &args.command {
        Command::Tokenize {
            data,
            tokens,
            initial_size,
            chunk_size,
            in_memory,
        } => tokenize(
            data.as_str(),
            tokens,
            *initial_size,
            *chunk_size,
            *in_memory,
        ),

        Command::OptimizeTokens {
            data,
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
            data.as_str(),
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
            data,
            // data_caps,
            data_caps_words,
            tokens_dir,
            min_tokens,
            max_tokens,
        } => optimize_all(
            data.as_str(),
            data_caps_words.as_deref(),
            &tokens_dir.as_str(),
            *min_tokens,
            *max_tokens,
        ),

        Command::Process { data, output } => process(data.as_str(), output.as_str()),

        Command::CountHexDigits { data } => count_hex_digits(data.as_str()),
        Command::CountBytes { data } => count_bytes_command(data.as_str()),
        Command::CountChars { data } => count_chars(data.as_str()),

        Command::OptimizeCharsTokens { data , ntokens, output } => {
            let sampler = FileSampler::new(data, 1 << 32, None);
            optimize_chars_tokens(&sampler, &sampler, &sampler, *ntokens, output);
        }
    }
}
