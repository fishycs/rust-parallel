use super::SEQ_THRESHOLD;
use crate::utils::{unwrap, unwrap_mut};
use num_cpus;
use std::mem::replace;
use std::ops::{IndexMut, RangeTo};

struct PrefixTree {
    sum: usize,
    left: Option<Box<PrefixTree>>,
    right: Option<Box<PrefixTree>>,
}

impl PrefixTree {
    fn new() -> Self {
        return PrefixTree {
            sum: 0,
            left: None,
            right: None,
        };
    }
}

struct BitArray {
    bits: Vec<u8>,
}

impl BitArray {
    pub fn new(len_bytes: usize) -> Self {
        return BitArray {
            bits: vec![0; len_bytes],
        };
    }

    pub fn as_slice(&self) -> BitSlice {
        return BitSlice { bits: &self.bits[..] };
    }

    pub fn as_slice_mut(&mut self) -> BitSliceMut {
        return BitSliceMut {
            bits: &mut self.bits[..],
        };
    }
}

struct BitSlice<'s> {
    bits: &'s [u8],
}

impl<'s> BitSlice<'s> {
    pub fn split_at_byte(&self, byte: usize) -> (BitSlice, BitSlice) {
        let (left_bits, right_bits) = self.bits.split_at(byte);
        return (BitSlice { bits: left_bits }, BitSlice { bits: right_bits });
    }

    pub fn get(&self, index: usize) -> bool {
        return self.bits[index / 8] >> (index % 8) & 1_u8 == 1_u8;
    }

    pub fn len(&self) -> usize {
        return self.bits.len();
    }
}

struct BitSliceMut<'s> {
    bits: &'s mut [u8],
}

impl<'s> BitSliceMut<'s> {
    pub fn split_at_byte_mut(&mut self, byte: usize) -> (BitSliceMut, BitSliceMut) {
        let (left_bits, right_bits) = self.bits.split_at_mut(byte);
        return (BitSliceMut { bits: left_bits }, BitSliceMut { bits: right_bits });
    }

    pub fn get(&self, index: usize) -> bool {
        return self.bits[index / 8] >> (index % 8) == 1_u8;
    }

    pub fn set(&mut self, index: usize, val: bool) {
        if val {
            self.bits[index / 8] |= 1 << (index % 8);
        } else {
            self.bits[index / 8] &= !(1 << (index % 8));
        }
    }

    pub fn len(&self) -> usize {
        return self.bits.len();
    }
}

pub fn alloc_copy<T, F, B, A>(src: &[T], filter_fn: &F, allocate_fn: &A, buffered: bool) -> (B, usize)
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
    B: IndexMut<RangeTo<usize>, Output = [T]>,
    A: Fn(usize) -> B,
{
    return alloc_copy_tune(src, filter_fn, allocate_fn, buffered, num_cpus::get(), SEQ_THRESHOLD);
}

pub fn alloc_copy_tune<T, F, B, A>(
    src: &[T],
    filter_fn: &F,
    allocate_fn: &A,
    buffered: bool,
    threads: usize,
    seq_threshold: usize,
) -> (B, usize)
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
    B: IndexMut<RangeTo<usize>, Output = [T]>,
    A: Fn(usize) -> B,
{
    if threads == 0 {
        panic!("threads cannot be zero!");
    }
    if seq_threshold == 0 {
        panic!("seq_threshold cannot be zero!");
    }

    if src.len() > 0 {
        if buffered {
            // buffered

            // generate partial sums.
            let mut tree: PrefixTree = PrefixTree::new();
            let mut buf: BitArray = BitArray::new((src.len() + 7 * threads) / 8);
            parallel_pack_sum_buf(src, buf.as_slice_mut(), &mut tree, filter_fn, threads, seq_threshold);

            // allocate destination
            let mut dest: B = allocate_fn(tree.sum);
            let dest_slice: &mut [T] = &mut dest[..tree.sum];
            if dest_slice.len() < tree.sum {
                panic!(
                    "allocated buffer length is too small! (required: {}, found: {})",
                    tree.sum,
                    dest_slice.len()
                );
            }

            // move elements
            parallel_pack_copy_buf(src, dest_slice, buf.as_slice(), &tree, threads, seq_threshold);

            return (dest, tree.sum);
        } else {
            // non-buffered

            // generate partial sums
            let mut tree: PrefixTree = PrefixTree::new();
            parallel_pack_sum_unbuf(src, &mut tree, filter_fn, threads, seq_threshold);

            // allocate destination
            let mut dest: B = allocate_fn(tree.sum);
            let dest_slice: &mut [T] = &mut dest[..tree.sum];
            if dest_slice.len() < tree.sum {
                panic!(
                    "allocated buffer length is too small! (required: {}, found: {})",
                    tree.sum,
                    dest_slice.len()
                );
            }

            // move elements
            parallel_pack_copy_unbuf(src, dest_slice, &tree, filter_fn, threads, seq_threshold);

            return (dest, tree.sum);
        }
    } else {
        return (allocate_fn(0), 0);
    }
}

pub fn alloc_move<T, F, B, A, D>(src: &mut [T], filter_fn: &F, allocate_fn: &A, default_fn: &D, buffered: bool) -> (B, usize)
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    B: IndexMut<RangeTo<usize>, Output = [T]>,
    A: Fn(usize) -> B,
    D: Fn(&T, usize) -> T + Sync,
{
    return alloc_move_tune(
        src,
        filter_fn,
        allocate_fn,
        default_fn,
        buffered,
        num_cpus::get(),
        SEQ_THRESHOLD,
    );
}

pub fn alloc_move_tune<T, F, B, A, D>(
    src: &mut [T],
    filter_fn: &F,
    allocate_fn: &A,
    default_fn: &D,
    buffered: bool,
    threads: usize,
    seq_threshold: usize,
) -> (B, usize)
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    B: IndexMut<RangeTo<usize>, Output = [T]>,
    A: Fn(usize) -> B,
    D: Fn(&T, usize) -> T + Sync,
{
    if threads == 0 {
        panic!("threads cannot be zero!");
    }
    if seq_threshold == 0 {
        panic!("seq_threshold cannot be zero!");
    }

    if src.len() > 0 {
        if buffered {
            // buffered

            // generate partial sums.
            let mut tree: PrefixTree = PrefixTree::new();
            let mut buf: BitArray = BitArray::new((src.len() + 7 * threads) / 8);
            parallel_pack_sum_buf(src, buf.as_slice_mut(), &mut tree, filter_fn, threads, seq_threshold);

            // allocate destination
            let mut dest: B = allocate_fn(tree.sum);
            let dest_slice: &mut [T] = &mut dest[..tree.sum];
            if dest_slice.len() < tree.sum {
                panic!(
                    "allocated buffer length is too small! (required: {}, found: {})",
                    tree.sum,
                    dest_slice.len()
                );
            }

            // move elements
            parallel_pack_move_buf(src, dest_slice, buf.as_slice(), &tree, default_fn, 0, threads, seq_threshold);

            return (dest, tree.sum);
        } else {
            // non-buffered

            // generate partial sums
            let mut tree: PrefixTree = PrefixTree::new();
            parallel_pack_sum_unbuf(src, &mut tree, filter_fn, threads, seq_threshold);

            // allocate destination
            let mut dest: B = allocate_fn(tree.sum);
            let dest_slice: &mut [T] = &mut dest[..tree.sum];
            if dest_slice.len() < tree.sum {
                panic!(
                    "allocated buffer length is too small! (required: {}, found: {})",
                    tree.sum,
                    dest_slice.len()
                );
            }

            // move elements
            parallel_pack_move_unbuf(src, dest_slice, &tree, filter_fn, default_fn, 0, threads, seq_threshold);

            return (dest, tree.sum);
        }
    } else {
        return (allocate_fn(0), 0);
    }
}

pub fn dest_copy<T, F>(src: &[T], dest: &mut [T], filter_fn: &F, buffered: bool) -> usize
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
{
    return dest_copy_tune(src, dest, filter_fn, buffered, num_cpus::get(), SEQ_THRESHOLD);
}

pub fn dest_copy_tune<T, F>(
    src: &[T],
    dest: &mut [T],
    filter_fn: &F,
    buffered: bool,
    threads: usize,
    seq_threshold: usize,
) -> usize
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
{
    if threads == 0 {
        panic!("threads cannot be zero!");
    }
    if seq_threshold == 0 {
        panic!("seq_threshold cannot be zero!");
    }

    if src.len() > 0 {
        if buffered {
            // buffered

            // generate partial sums.
            let mut tree: PrefixTree = PrefixTree::new();
            let mut buf: BitArray = BitArray::new((src.len() + 7 * threads) / 8);
            parallel_pack_sum_buf(src, buf.as_slice_mut(), &mut tree, filter_fn, threads, seq_threshold);

            // check destination buffer
            if dest.len() < tree.sum {
                panic!(
                    "provided buffer is too small! (required: {}, found: {})",
                    tree.sum,
                    dest.len()
                );
            }

            // move elements
            parallel_pack_copy_buf(src, dest, buf.as_slice(), &tree, threads, seq_threshold);

            return tree.sum;
        } else {
            // non-buffered

            // generate partial sums
            let mut tree: PrefixTree = PrefixTree::new();
            parallel_pack_sum_unbuf(src, &mut tree, filter_fn, threads, seq_threshold);

            // check destination buffer
            if dest.len() < tree.sum {
                panic!(
                    "provided destination buffer is too small! (required: {}, found: {})",
                    tree.sum,
                    dest.len()
                );
            }

            // move elements
            parallel_pack_copy_unbuf(src, dest, &tree, filter_fn, threads, seq_threshold);

            return tree.sum;
        }
    } else {
        return 0;
    }
}

pub fn dest_move<T, F, D>(src: &mut [T], dest: &mut [T], filter_fn: &F, default_fn: &D, buffered: bool) -> usize
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    D: Fn(&T, usize) -> T + Sync,
{
    return dest_move_tune(src, dest, filter_fn, default_fn, buffered, num_cpus::get(), SEQ_THRESHOLD);
}

pub fn dest_move_tune<T, F, D>(
    src: &mut [T],
    dest: &mut [T],
    filter_fn: &F,
    default_fn: &D,
    buffered: bool,
    threads: usize,
    seq_threshold: usize,
) -> usize
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    D: Fn(&T, usize) -> T + Sync,
{
    if threads == 0 {
        panic!("threads cannot be zero!");
    }
    if seq_threshold == 0 {
        panic!("seq_threshold cannot be zero!");
    }

    if src.len() > 0 {
        if buffered {
            // buffered

            // generate partial sums.
            let mut tree: PrefixTree = PrefixTree::new();
            let mut buf: BitArray = BitArray::new((src.len() + 7 * threads) / 8);
            parallel_pack_sum_buf(src, buf.as_slice_mut(), &mut tree, filter_fn, threads, seq_threshold);

            // check destination buffer
            if dest.len() < tree.sum {
                panic!(
                    "provided buffer is too small! (required: {}, found: {})",
                    tree.sum,
                    dest.len()
                );
            }

            // move elements
            parallel_pack_move_buf(src, dest, buf.as_slice(), &tree, default_fn, 0, threads, seq_threshold);

            return tree.sum;
        } else {
            // non-buffered

            // generate partial sums
            let mut tree: PrefixTree = PrefixTree::new();
            parallel_pack_sum_unbuf(src, &mut tree, filter_fn, threads, seq_threshold);

            // check destination buffer
            if dest.len() < tree.sum {
                panic!(
                    "provided destination buffer is too small! (required: {}, found: {})",
                    tree.sum,
                    dest.len()
                );
            }

            // move elements
            parallel_pack_move_unbuf(src, dest, &tree, filter_fn, default_fn, 0, threads, seq_threshold);

            return tree.sum;
        }
    } else {
        return 0;
    }
}

fn parallel_pack_sum_unbuf<T, F>(array: &[T], tree: &mut PrefixTree, filter_fn: &F, threads: usize, seq_threshold: usize)
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
{
    if threads == 1 || array.len() <= seq_threshold {
        // sequential

        for t in array.iter() {
            tree.sum += filter_fn(t) as usize;
        }
    } else {
        // parallel

        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        let (left_array, right_array) = array.split_at(array.len() / 2);

        tree.left = Some(Box::new(PrefixTree::new()));
        tree.right = Some(Box::new(PrefixTree::new()));
        let left_tree: &mut PrefixTree = unwrap_mut(&mut tree.left);
        let right_tree: &mut PrefixTree = unwrap_mut(&mut tree.right);

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_pack_sum_unbuf(left_array, left_tree, filter_fn, left_threads, seq_threshold);
            });
            parallel_pack_sum_unbuf(right_array, right_tree, filter_fn, right_threads, seq_threshold);
        })
        .unwrap();

        tree.sum = left_tree.sum + right_tree.sum;
    }
}

fn parallel_pack_copy_unbuf<T, F>(
    src: &[T],
    dest: &mut [T],
    tree: &PrefixTree,
    filter_fn: &F,
    threads: usize,
    seq_threshold: usize,
) where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
{
    if threads == 1 || src.len() <= seq_threshold {
        // sequential

        let mut i: usize = 0;
        for t in src.iter() {
            if filter_fn(t) {
                dest[i] = *t;
                i += 1;
            }
        }
    } else {
        // parallel

        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        let left_tree: &PrefixTree = unwrap(&tree.left);
        let right_tree: &PrefixTree = unwrap(&tree.right);

        let (src_left, src_right) = src.split_at(src.len() / 2);
        let (dest_left, dest_right) = dest.split_at_mut(left_tree.sum);

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_pack_copy_unbuf(src_left, dest_left, left_tree, filter_fn, left_threads, seq_threshold);
            });
            parallel_pack_copy_unbuf(src_right, dest_right, right_tree, filter_fn, right_threads, seq_threshold);
        })
        .unwrap();
    }
}

fn parallel_pack_move_unbuf<T, F, D>(
    src: &mut [T],
    dest: &mut [T],
    tree: &PrefixTree,
    filter_fn: &F,
    default_fn: &D,
    mut base: usize,
    threads: usize,
    seq_threshold: usize,
) where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    D: Fn(&T, usize) -> T + Sync,
{
    if threads == 1 || src.len() <= seq_threshold {
        // sequential

        let mut i: usize = 0;
        for t in src.iter_mut() {
            if filter_fn(t) {
                let new: T = default_fn(t, base);
                dest[i] = replace(t, new);
                i += 1;
            }
            base += 1;
        }
    } else {
        // parallel

        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        let left_tree: &PrefixTree = unwrap(&tree.left);
        let right_tree: &PrefixTree = unwrap(&tree.right);

        let (src_left, src_right) = src.split_at_mut(src.len() / 2);
        let (dest_left, dest_right) = dest.split_at_mut(left_tree.sum);

        let left_base: usize = base;
        let right_base: usize = base + src_left.len();

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_pack_move_unbuf(
                    src_left,
                    dest_left,
                    left_tree,
                    filter_fn,
                    default_fn,
                    left_base,
                    left_threads,
                    seq_threshold,
                );
            });
            parallel_pack_move_unbuf(
                src_right,
                dest_right,
                right_tree,
                filter_fn,
                default_fn,
                right_base,
                right_threads,
                seq_threshold,
            );
        })
        .unwrap();
    }
}

fn parallel_pack_sum_buf<T, F>(
    array: &[T],
    mut buf: BitSliceMut,
    tree: &mut PrefixTree,
    filter_fn: &F,
    threads: usize,
    seq_threshold: usize,
) where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
{
    if threads == 1 || array.len() <= seq_threshold {
        // sequential

        for i in 0..array.len() {
            let b: bool = filter_fn(&array[i]);
            tree.sum += b as usize;
            buf.set(i, b);
        }
    } else {
        // parallel

        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        let (left_array, right_array) = array.split_at(array.len() / 2);

        let (left_buf, right_buf) = buf.split_at_byte_mut(buf.len() / 2);

        tree.left = Some(Box::new(PrefixTree::new()));
        tree.right = Some(Box::new(PrefixTree::new()));
        let left_tree: &mut PrefixTree = unwrap_mut(&mut tree.left);
        let right_tree: &mut PrefixTree = unwrap_mut(&mut tree.right);

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_pack_sum_buf(left_array, left_buf, left_tree, filter_fn, left_threads, seq_threshold);
            });
            parallel_pack_sum_buf(right_array, right_buf, right_tree, filter_fn, right_threads, seq_threshold);
        })
        .unwrap();

        tree.sum = left_tree.sum + right_tree.sum;
    }
}

fn parallel_pack_copy_buf<T>(src: &[T], dest: &mut [T], buf: BitSlice, tree: &PrefixTree, threads: usize, seq_threshold: usize)
where
    T: Copy + Send + Sync,
{
    if threads == 1 || src.len() <= seq_threshold {
        // sequential

        let mut j: usize = 0;
        for i in 0..src.len() {
            if buf.get(i) {
                dest[j] = src[i];
                j += 1;
            }
        }
    } else {
        // parallel

        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        let left_tree: &PrefixTree = unwrap(&tree.left);
        let right_tree: &PrefixTree = unwrap(&tree.right);

        let (left_src, right_src) = src.split_at(src.len() / 2);
        let (left_dest, right_dest) = dest.split_at_mut(left_tree.sum);

        let (left_buf, right_buf) = buf.split_at_byte(buf.len() / 2);

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_pack_copy_buf(left_src, left_dest, left_buf, left_tree, left_threads, seq_threshold);
            });
            parallel_pack_copy_buf(right_src, right_dest, right_buf, right_tree, right_threads, seq_threshold);
        })
        .unwrap();
    }
}

fn parallel_pack_move_buf<T, D>(
    src: &mut [T],
    dest: &mut [T],
    buf: BitSlice,
    tree: &PrefixTree,
    default_fn: &D,
    base: usize,
    threads: usize,
    seq_threshold: usize,
) where
    T: Send + Sync,
    D: Fn(&T, usize) -> T + Sync,
{
    if threads == 1 || src.len() <= seq_threshold {
        // sequential

        let mut j: usize = 0;
        for i in 0..src.len() {
            if buf.get(i) {
                let new: T = default_fn(&src[i], base + i);
                dest[j] = replace(&mut src[i], new);
                j += 1;
            }
        }
    } else {
        // parallel

        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        let left_tree: &PrefixTree = unwrap(&tree.left);
        let right_tree: &PrefixTree = unwrap(&tree.right);

        let (left_src, right_src) = src.split_at_mut(src.len() / 2);
        let (left_dest, right_dest) = dest.split_at_mut(left_tree.sum);

        let (left_buf, right_buf) = buf.split_at_byte(buf.len() / 2);

        let left_base: usize = base;
        let right_base: usize = base + left_src.len();

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_pack_move_buf(
                    left_src,
                    left_dest,
                    left_buf,
                    left_tree,
                    default_fn,
                    left_base,
                    left_threads,
                    seq_threshold,
                );
            });
            parallel_pack_move_buf(
                right_src,
                right_dest,
                right_buf,
                right_tree,
                default_fn,
                right_base,
                right_threads,
                seq_threshold,
            );
        })
        .unwrap();
    }
}

// run some tests
#[cfg(test)]
mod tests {
    use crate::pack::*;
    use std::time::Instant;

    const N: usize = 10000000;
    const N_SMALL: usize = 35;

    // test parallel pack algorithm (unbuf)
    #[test]
    fn test_parallel_pack_small_unbuf() {
        let par: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 12];

        let (out, len) = alloc_copy_tune(
            &par[..],
            &|a: &u8| -> bool {
                return *a % 3 == 0;
            },
            &|a: usize| -> Vec<u8> {
                return vec![0; a];
            },
            false,
            12,
            1,
        );

        assert_eq!(out, vec![3, 6, 9, 12]);

        println!(">>> PACK [small unbuf]: out = {:?}, len = {}", out, len);
    }

    // test parallel pack algorithm (buf)
    #[test]
    fn test_parallel_pack_small_buf() {
        let par: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 12];

        let (out, len) = alloc_copy_tune(
            &par[..],
            &|a: &u8| -> bool {
                return *a % 3 == 0;
            },
            &|a: usize| -> Vec<u8> {
                return vec![0; a];
            },
            true,
            12,
            1,
        );

        assert_eq!(out, vec![3, 6, 9, 12]);

        println!(">>> PACK [small buf]: out = {:?}, len = {}", out, len);
    }

    // test parallel pack algorithm another time (buf)
    #[test]
    fn test_parallel_pack_small_2_buf() {
        let mut par: Vec<usize> = Vec::new();
        let mut seq: Vec<usize> = Vec::new();
        for i in 0..N_SMALL {
            par.push(i);
            if i % 3 == 0 {
                seq.push(i);
            }
        }

        let (out, len) = alloc_copy_tune(
            &par[..],
            &|a: &usize| -> bool {
                return *a % 3 == 0;
            },
            &|a: usize| -> Vec<usize> {
                return vec![0; a];
            },
            true,
            12,
            1,
        );

        assert_eq!(out, seq);

        println!(">>> PACK [small buf 2]: out = {:?}, len = {}", out, len);
    }

    // test parallel pack alloc copy algorithm large (unbuf)
    #[test]
    fn test_parallel_pack_alloc_copy_large_unbuf() {
        // generate some random data
        let mut array: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            array.push(64 + i * i - 8 * i + 5);
        }

        // time parallel algorithm
        let start_par = Instant::now();
        let (par, len) = alloc_copy_tune(
            &array[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            &|a: usize| -> Vec<u128> {
                return vec![0; a];
            },
            false,
            24,
            256,
        );
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        for i in array {
            if filter_fn(&i) {
                seq.push(i);
            }
        }
        let dur_seq = start_seq.elapsed();

        // check integrity of results
        assert_eq!(seq, par);

        // print results
        println!(
            ">>> PACK [alloc copy unbuf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }

    // test parallel pack alloc move algorithm large (unbuf)
    #[test]
    fn test_parallel_pack_alloc_move_large_unbuf() {
        // generate some random data
        let mut par_src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par_src.push(64 + i * i - 8 * i + 5);
        }
        let mut seq_src: Vec<u128> = par_src.clone();

        // time parallel algorithm
        let start_par = Instant::now();
        let (par_dest, len) = alloc_move_tune(
            &mut par_src[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            &|a: usize| -> Vec<u128> {
                return vec![0; a];
            },
            &|a: &u128, b: usize| -> u128 {
                return *a + (b as u128);
            },
            false,
            24,
            256,
        );
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq_dest: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        let mut j = 0;
        for i in seq_src.iter_mut() {
            if filter_fn(&i) {
                seq_dest.push(*i);
                *i += j;
            }
            j += 1;
        }
        let dur_seq = start_seq.elapsed();

        // check integrity of results
        assert_eq!(seq_src, par_src);
        assert_eq!(seq_dest, par_dest);

        // print results
        println!(
            ">>> PACK [alloc move unbuf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }

    // test parallel pack alloc move algorithm large (unbuf)
    #[test]
    fn test_parallel_pack_dest_copy_large_unbuf() {
        // generate some random data
        let mut src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            src.push(64 + i * i - 8 * i + 5);
        }

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq_dest: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        for i in src.iter() {
            if filter_fn(&i) {
                seq_dest.push(*i);
            }
        }
        let dur_seq = start_seq.elapsed();

        // time parallel algorithm
        let start_par = Instant::now();
        let mut par_dest = vec![0; seq_dest.len()];
        let len = dest_copy(
            &src[..],
            &mut par_dest[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            false,
        );
        let dur_par = start_par.elapsed();

        // check integrity of results
        assert_eq!(seq_dest, par_dest);

        // print results
        println!(
            ">>> PACK [dest copy unbuf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }

    // test parallel pack alloc move algorithm large (unbuf)
    #[test]
    fn test_parallel_pack_dest_move_large_unbuf() {
        // generate some random data
        let mut par_src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par_src.push(64 + i * i - 8 * i + 5);
        }
        let mut seq_src: Vec<u128> = par_src.clone();

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq_dest: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        let mut j = 0;
        for i in seq_src.iter_mut() {
            if filter_fn(&i) {
                seq_dest.push(*i);
                *i += j;
            }
            j += 1;
        }
        let dur_seq = start_seq.elapsed();

        // time parallel algorithm
        let start_par = Instant::now();
        let mut par_dest = vec![0; seq_dest.len()];
        let len = dest_move(
            &mut par_src[..],
            &mut par_dest[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            &|a: &u128, b: usize| -> u128 {
                return *a + (b as u128);
            },
            false,
        );
        let dur_par = start_par.elapsed();

        // check integrity of results
        assert_eq!(seq_src, par_src);
        assert_eq!(seq_dest, par_dest);

        // print results
        println!(
            ">>> PACK [dest move unbuf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }

    // test parallel pack alloc copy algorithm large (buf)
    #[test]
    fn test_parallel_pack_alloc_copy_large_buf() {
        // generate some random data
        let mut array: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            array.push(64 + i * i - 8 * i + 5);
        }

        // time parallel algorithm
        let start_par = Instant::now();
        let (par, len) = alloc_copy_tune(
            &array[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            &|a: usize| -> Vec<u128> {
                return vec![0; a];
            },
            true,
            24,
            256,
        );
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        for i in array {
            if filter_fn(&i) {
                seq.push(i);
            }
        }
        let dur_seq = start_seq.elapsed();

        // check integrity of results
        assert_eq!(seq, par);

        // print results
        println!(
            ">>> PACK [alloc copy buf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }

    // test parallel pack alloc move algorithm large (buf)
    #[test]
    fn test_parallel_pack_alloc_move_large_buf() {
        // generate some random data
        let mut par_src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par_src.push(64 + i * i - 8 * i + 5);
        }
        let mut seq_src: Vec<u128> = par_src.clone();

        // time parallel algorithm
        let start_par = Instant::now();
        let (par_dest, len) = alloc_move_tune(
            &mut par_src[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            &|a: usize| -> Vec<u128> {
                return vec![0; a];
            },
            &|a: &u128, b: usize| -> u128 {
                return *a + (b as u128);
            },
            true,
            24,
            256,
        );
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq_dest: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        let mut j = 0;
        for i in seq_src.iter_mut() {
            if filter_fn(&i) {
                seq_dest.push(*i);
                *i += j;
            }
            j += 1;
        }
        let dur_seq = start_seq.elapsed();

        // check integrity of results
        assert_eq!(seq_src, par_src);
        assert_eq!(seq_dest, par_dest);

        // print results
        println!(
            ">>> PACK [alloc move buf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }

    // test parallel pack alloc move algorithm large (buf)
    #[test]
    fn test_parallel_pack_dest_copy_large_buf() {
        // generate some random data
        let mut src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            src.push(64 + i * i - 8 * i + 5);
        }

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq_dest: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        for i in src.iter() {
            if filter_fn(&i) {
                seq_dest.push(*i);
            }
        }
        let dur_seq = start_seq.elapsed();

        // time parallel algorithm
        let start_par = Instant::now();
        let mut par_dest = vec![0; seq_dest.len()];
        let len = dest_copy(
            &src[..],
            &mut par_dest[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            true,
        );
        let dur_par = start_par.elapsed();

        // check integrity of results
        assert_eq!(seq_dest, par_dest);

        // print results
        println!(
            ">>> PACK [dest copy buf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }

    // test parallel pack alloc move algorithm large (buf)
    #[test]
    fn test_parallel_pack_dest_move_large_buf() {
        // generate some random data
        let mut par_src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par_src.push(64 + i * i - 8 * i + 5);
        }
        let mut seq_src: Vec<u128> = par_src.clone();

        // time sequential algorithm
        let start_seq = Instant::now();
        let mut seq_dest: Vec<u128> = Vec::new();
        let filter_fn = |a: &u128| -> bool {
            return *a % 2 == 0;
        };
        let mut j = 0;
        for i in seq_src.iter_mut() {
            if filter_fn(&i) {
                seq_dest.push(*i);
                *i += j;
            }
            j += 1;
        }
        let dur_seq = start_seq.elapsed();

        // time parallel algorithm
        let start_par = Instant::now();
        let mut par_dest = vec![0; seq_dest.len()];
        let len = dest_move(
            &mut par_src[..],
            &mut par_dest[..],
            &|a: &u128| -> bool {
                return *a % 2 == 0;
            },
            &|a: &u128, b: usize| -> u128 {
                return *a + (b as u128);
            },
            true,
        );
        let dur_par = start_par.elapsed();

        // check integrity of results
        assert_eq!(seq_src, par_src);
        assert_eq!(seq_dest, par_dest);

        // print results
        println!(
            ">>> PACK [dest move buf]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}",
            dur_par, dur_seq, N, len
        );
    }
}
