// implementation of common parallel algorithms in rust
// - parallel prefix/sum
// - parallel pack
// - parallel map (coming soon)
// - parallel reduce (coming soon)

const SEQ_THRESHOLD: usize = 256;

pub mod map;
pub mod pack;
pub mod prefix;
pub mod reduce;
mod utils;
