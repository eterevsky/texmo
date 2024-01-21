use rand::Rng;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::iter::Iterator;

use crate::input::sample::{Sample, Sampler};


pub struct FileSampler {
    filename: String,
    sample_size: usize,
    _total_size: u64,
    max_samples: Option<usize>,
}

impl FileSampler {
    pub fn new(filename: &str, sample_size: usize, max_samples: Option<usize>) -> Self {
        let _total_size = if let Some(cs) = max_samples {
            (sample_size * cs) as u64
        } else {
            std::fs::metadata(filename).unwrap().len()
        };
        FileSampler {
            filename: filename.to_string(),
            sample_size,
            _total_size,
            max_samples,
        }
    }
}

impl<'a> Sampler<'a> for FileSampler {
    type Iter = FileIterator<'a>;

    fn iter(&'a self) -> Self::Iter {
        let file = File::open(self.filename.as_str()).unwrap();

        if let Some(chunks_selection) = self.max_samples {
            FileIterator {
                _sampler: self,
                file,
                sample_size: self.sample_size,
                total_size: self._total_size,
                samples_left: Some(chunks_selection),
                read_bytes: 0,
            }
        } else {
            FileIterator {
                _sampler: self,
                file,
                sample_size: self.sample_size,
                total_size: self._total_size,
                samples_left: None,
                read_bytes: 0,
            }
        }
    }

    fn total_size(&self) -> u64 {
        self._total_size
    }
}

pub struct FileIterator<'a> {
    _sampler: &'a FileSampler,
    file: File,
    sample_size: usize,
    total_size: u64,
    samples_left: Option<usize>,

    pub read_bytes: usize,
}

impl<'a> Iterator for FileIterator<'a> {
    type Item = Sample<'a>;

    fn next(&mut self) -> Option<Sample<'a>> {
        let mut buffer = Vec::new();
        buffer.resize(self.sample_size, 0);

        if let Some(chunks_left) = self.samples_left {
            if chunks_left == 0 {
                None
            } else {
                self.samples_left = Some(chunks_left - 1);

                let mut rng = rand::thread_rng();
                let max_seek = self.total_size - self.sample_size as u64;
                let start = rng.gen_range(0..max_seek);

                self.file.seek(SeekFrom::Start(start)).unwrap();
                let read_bytes = self.file.read(&mut buffer).unwrap();

                buffer.truncate(read_bytes);
                self.read_bytes += read_bytes;
                Some(Sample::from_vec(buffer))
            }
        } else {
            let read_bytes = self.file.read(&mut buffer).unwrap();

            if read_bytes == 0 {
                None
            } else {
                buffer.truncate(read_bytes);
                self.read_bytes += read_bytes;
                // if self.read_bytes & ((1<<30) - 1) == 0 {
                //     dbg!(self.read_bytes);
                // }
                Some(Sample::from_vec(buffer))
            }
        }
    }
}

