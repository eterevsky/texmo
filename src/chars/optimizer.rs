use crate::input::sample::Sampler;

fn count_chars<'a, S: Sampler<'a>>(sampler: &'a S) -> Vec<u64> {
    let mut counts = Vec::new();

    for sample in sampler.iter() {
        eprint!(".");
        for c in sample.as_str().chars() {
            let idx = c as usize;
            if idx >= counts.len() {
                counts.resize(idx + 1, 0);
            }
            counts[idx] += 1;
        }
    }
    eprintln!();

    counts
}

pub fn optimize_chars_tokens<'a, SS: Sampler<'a>, S: Sampler<'a>, FS: Sampler<'a>>(
    slow_sampler: &'a SS,
    sampler: &'a S,
    fast_sampler: &'a FS,
) {
    let counts = count_chars(slow_sampler);
    let mut indices = (0..counts.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| -(counts[i] as i64));
    for idx in indices.iter().take(256) {
        println!("{:?}:  {}", char::from_u32(*idx as u32).unwrap(), counts[*idx]);
    }
}