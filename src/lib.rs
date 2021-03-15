// implementation of common parallel algorithms in rust
// - parallel prefix (e.g. sum)
// - parallel pack (coming soon)

const SEQUENTIAL_THRESHOLD: usize = 256;

mod prefix;
mod pack;
