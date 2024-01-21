use crate::input::sample::{Sample, Sampler};

pub struct MemorySampler {
    data: Vec<u8>,
    chunk_size: usize,
}

impl MemorySampler {
    pub fn new(filename: &str, chunk_size: usize) -> Self {
        let data = std::fs::read(filename).unwrap();
        MemorySampler { data, chunk_size }
    }

    pub fn new_from_str(data: &str, chunk_size: usize) -> Self {
        MemorySampler {
            data: data.as_bytes().to_vec(),
            chunk_size,
        }
    }
}

impl<'a> Sampler<'a> for MemorySampler {
    type Iter = MemoryIterator<'a>;

    fn iter(&'a self) -> Self::Iter {
        MemoryIterator {
            sampler: self,
            position: 0,
        }
    }

    fn total_size(&'a self) -> u64 {
        self.data.len() as u64
    }
}

pub struct MemoryIterator<'a> {
    sampler: &'a MemorySampler,
    position: usize,
}

impl<'a> Iterator for MemoryIterator<'a> {
    type Item = Sample<'a>;

    fn next(&mut self) -> Option<Sample<'a>> {
        if self.position < self.sampler.data.len() {
            let start = self.position;
            self.position = std::cmp::min(start + self.sampler.chunk_size, self.sampler.data.len());
            Some(Sample::from_bytes(&self.sampler.data[start..self.position]))
        } else {
            None
        }
    }
}

