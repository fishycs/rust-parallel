use num_cpus;
use std::mem::replace;
use std::ops::{IndexMut, RangeTo};
use super::SEQUENTIAL_THRESHOLD;

struct PrefixTree {
    sum: usize,
    left: Option<Box<PrefixTree>>,
    right: Option<Box<PrefixTree>>,
}

impl PrefixTree {
    fn new() -> Self {
	return PrefixTree{sum: 0, left: None, right: None};
    }
}

pub fn alloc_copy<T, F, B, A>(array: &[T], filter_fn: &F, allocate_fn: &A, buffered: bool) -> (B, usize)
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
    B: IndexMut<RangeTo<usize>, Output = [T]>,
    A: Fn(usize) -> B,
{
    return alloc_copy_tune(array, filter_fn, allocate_fn, buffered, num_cpus::get(), SEQUENTIAL_THRESHOLD);
}

pub fn alloc_copy_tune<T, F, B, A>(array: &[T], filter_fn: &F, allocate_fn: &A, buffered: bool, threads: usize, seq_threshold: usize) -> (B, usize)
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
    
    if array.len() > 0 {
	if buffered {
	    // buffered

	    panic!("buffered version is not yet implemented!");
	} else {
	    // non-buffered

	    // generate partial sums
	    let mut tree: PrefixTree = PrefixTree::new();
	    parallel_pack_sum(array, &mut tree, filter_fn, threads, seq_threshold);

	    // allocate buffer
	    let mut buf: B = allocate_fn(tree.sum);
	    let mut slice: &mut[T] = &mut buf[..tree.sum];
	    if slice.len() < tree.sum {
		panic!("allocated buffer length is too small! (required: {}, found: {})", tree.sum, slice.len());
	    }

	    // move elements
	    parallel_pack_copy(array, slice, &tree, filter_fn, threads, seq_threshold);

	    return (buf, tree.sum);
	}
    } else {
	return (allocate_fn(0), 0);
    }
}

pub fn alloc_move<T, F, B, A, D>(array: &mut[T], filter_fn: &F, allocate_fn: &A, default_fn: &D, buffered: bool) -> (B, usize)
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    B: IndexMut<RangeTo<usize>, Output = [T]>,
    A: Fn(usize) -> B,
    D: Fn(&T, usize) -> T + Sync,
{
    return alloc_move_tune(array, filter_fn, allocate_fn, default_fn, buffered, num_cpus::get(), SEQUENTIAL_THRESHOLD);
}

pub fn alloc_move_tune<T, F, B, A, D>(array: &mut[T], filter_fn: &F, allocate_fn: &A, default_fn: &D, buffered: bool, threads: usize, seq_threshold: usize) -> (B, usize)
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
    
    if array.len() > 0 {
	if buffered {
	    // buffered

	    panic!("buffered version is not yet implemented!");
	} else {
	    // non-buffered

	    // generate partial sums
	    let mut tree: PrefixTree = PrefixTree::new();
	    parallel_pack_sum(array, &mut tree, filter_fn, threads, seq_threshold);

	    // allocate buffer
	    let mut buf: B = allocate_fn(tree.sum);
	    let mut slice: &mut[T] = &mut buf[..tree.sum];
	    if slice.len() < tree.sum {
		panic!("allocated buffer length is too small! (required: {}, found: {})", tree.sum, slice.len());
	    }

	    // move elements
	    parallel_pack_move(array, slice, &tree, filter_fn, default_fn, 0, threads, seq_threshold);

	    return (buf, tree.sum);
	}
    } else {
	return (allocate_fn(0), 0);
    }
}

pub fn dest_copy<T, F>(src: &[T], dest: &mut[T], filter_fn: &F, buffered: bool) -> usize
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
{
    return dest_copy_tune(src, dest, filter_fn, buffered, num_cpus::get(), SEQUENTIAL_THRESHOLD);
}

pub fn dest_copy_tune<T, F>(src: &[T], dest: &mut[T], filter_fn: &F, buffered: bool, threads: usize, seq_threshold: usize) -> usize
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

	    panic!("buffered version is not yet implemented!");
	} else {
	    // non-buffered

	    // generate partial sums
	    let mut tree: PrefixTree = PrefixTree::new();
	    parallel_pack_sum(src, &mut tree, filter_fn, threads, seq_threshold);

	    // check destination buffer
	    if dest.len() < tree.sum {
		panic!("provided destination buffer is too small! (required: {}, found: {})", tree.sum, dest.len());
	    }

	    // move elements
	    parallel_pack_copy(src, dest, &tree, filter_fn, threads, seq_threshold);

	    return tree.sum;
	}
    } else {
	return 0;
    }
}

pub fn dest_move<T, F, D>(src: &mut[T], dest: &mut[T], filter_fn: &F, default_fn: &D, buffered: bool) -> usize
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    D: Fn(&T, usize) -> T + Sync,
{
    return dest_move_tune(src, dest, filter_fn, default_fn, buffered, num_cpus::get(), SEQUENTIAL_THRESHOLD);
}

pub fn dest_move_tune<T, F, D>(src: &mut[T], dest: &mut[T], filter_fn: &F, default_fn: &D, buffered: bool, threads: usize, seq_threshold: usize) -> usize
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

	    panic!("buffered version is not yet implemented!");
	} else {
	    // non-buffered

	    // generate partial sums
	    let mut tree: PrefixTree = PrefixTree::new();
	    parallel_pack_sum(src, &mut tree, filter_fn, threads, seq_threshold);

	    // check destination buffer
	    if dest.len() < tree.sum {
		panic!("provided destination buffer is too small! (required: {}, found: {})", tree.sum, dest.len());
	    }

	    // move elements
	    parallel_pack_move(src, dest, &tree, filter_fn, default_fn, 0, threads, seq_threshold);

	    return tree.sum;
	}
    } else {
	return 0;
    }
}

fn parallel_pack_sum<T, F>(array: &[T], tree: &mut PrefixTree, filter_fn: &F, threads: usize, seq_threshold: usize)
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

        let (left_array, right_array) = array.split_at(array.len()/2);
	let mut left_tree = PrefixTree::new();
	let mut right_tree = PrefixTree::new();

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_pack_sum(left_array, &mut left_tree, filter_fn, left_threads, seq_threshold);
            });
            parallel_pack_sum(right_array, &mut right_tree, filter_fn, right_threads, seq_threshold);
        }).unwrap();

        tree.sum = left_tree.sum + right_tree.sum;
	tree.left = Some(Box::new(left_tree));
	tree.right = Some(Box::new(right_tree));
    }
}

fn parallel_pack_copy<T, F>(source: &[T], dest: &mut[T], tree: &PrefixTree, filter_fn: &F, threads: usize, seq_threshold: usize)
where
    T: Copy + Send + Sync,
    F: Fn(&T) -> bool + Sync,
{
    if threads == 1 || source.len() <= seq_threshold {
	// sequential

	let mut i: usize = 0;
	for t in source.iter() {
	    if filter_fn(t) {
		dest[i] = *t;
		i += 1;
	    }
	}
    } else {
	// parallel

	let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

	let left_tree: &PrefixTree = &**(&tree.left).as_ref().unwrap();
	let right_tree: &PrefixTree = &**(&tree.right).as_ref().unwrap();

	let (left_source, right_source) = source.split_at(source.len()/2);
	let (left_dest, right_dest) = dest.split_at_mut(left_tree.sum);

	crossbeam::scope(|scope| {
	    scope.spawn(|_| {
		parallel_pack_copy(left_source, left_dest, left_tree, filter_fn, left_threads, seq_threshold);
	    });
	    parallel_pack_copy(right_source, right_dest, right_tree, filter_fn, right_threads, seq_threshold);
	}).unwrap();
    }
}

fn parallel_pack_move<T, F, D>(source: &mut[T], dest: &mut[T], tree: &PrefixTree, filter_fn: &F, default_fn: &D, mut base: usize, threads: usize, seq_threshold: usize)
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Sync,
    D: Fn(&T, usize) -> T + Sync,
{
    if threads == 1 || source.len() <= seq_threshold {
	// sequential

	let mut i: usize = 0;
	for t in source.iter_mut() {
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

	let left_tree: &PrefixTree = &**(&tree.left).as_ref().unwrap();
	let right_tree: &PrefixTree = &**(&tree.right).as_ref().unwrap();

	let (left_source, right_source) = source.split_at_mut(source.len()/2);
	let (left_dest, right_dest) = dest.split_at_mut(left_tree.sum);

	let left_base: usize = base;
	let right_base: usize = base + left_source.len();

	crossbeam::scope(|scope| {
	    scope.spawn(|_| {
		parallel_pack_move(left_source, left_dest, left_tree, filter_fn, default_fn, left_base, left_threads, seq_threshold);
	    });
	    parallel_pack_move(right_source, right_dest, right_tree, filter_fn, default_fn, right_base, right_threads, seq_threshold);
	}).unwrap();
    }
}

// run some tests
#[cfg(test)]
mod tests {
    use crate::pack::{alloc_copy_tune, alloc_move_tune};
    use std::time::{Instant};

    const N: usize = 10000000;

    // test parallel pack algorithm
    #[test]
    fn test_parallel_pack_small() {
	let par: Vec<u8> = vec![1,2,3,4,5,6,7,8,9,12];

	let (out, size) = alloc_copy_tune(&par[..], &|a: &u8| -> bool {
	    return *a%3 == 0;
	}, &|a: usize| -> Vec<u8> {
	    return vec![0; a];
	}, false, 12, 1);

	println!(">>> PACK: out = {:?}", out);
    }

    // test parallel pack alloc copy algorithm large
    #[test]
    fn test_parallel_pack_alloc_copy_large() {
        // generate some random data
        let mut array: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            array.push(64 + i*i - 8*i + 5);
        }

        // time parallel algorithm
        let start_par = Instant::now();
	let (par, len) = alloc_copy_tune(&array[..], &|a: &u128| -> bool {
	    return *a%2 == 0;
	}, &|a: usize| -> Vec<u128> {
	    return vec![0; a];
	}, false, 24, 256);
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
	let mut seq: Vec<u128> = Vec::new();
	let filter_fn = |a: &u128| -> bool {
	    return *a%2 == 0;
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
        println!(">>> PACK [alloc copy]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}", dur_par, dur_seq, N, par.len());
    }

    // test parallel pack alloc move algorithm large
    #[test]
    fn test_parallel_pack_alloc_move_large() {
	// generate some random data
        let mut par_src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par_src.push(64 + i*i - 8*i + 5);
        }
	let mut seq_src: Vec<u128> = par_src.clone();

        // time parallel algorithm
        let start_par = Instant::now();
	let (par_dest, len) = alloc_move_tune(&mut par_src[..], &|a: &u128| -> bool {
	    return *a%2 == 0;
	}, &|a: usize| -> Vec<u128> {
	    return vec![0; a];
	}, &|a: &u128, b: usize| -> u128 {
	    return *a + (b as u128);
	}, false, 24, 256);
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
	let mut seq_dest: Vec<u128> = Vec::new();
	let filter_fn = |a: &u128| -> bool {
	    return *a%2 == 0;
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
        println!(">>> PACK [alloc move]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}", dur_par, dur_seq, N, par_dest.len());
    }

    // test parallel pack alloc move algorithm large
    #[test]
    fn test_parallel_pack_alloc_move_large() {
	// generate some random data
        let mut par_src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par_src.push(64 + i*i - 8*i + 5);
        }
	let mut seq_src: Vec<u128> = par_src.clone();

        // time parallel algorithm
        let start_par = Instant::now();
	let (par_dest, len) = alloc_move_tune(&mut par_src[..], &|a: &u128| -> bool {
	    return *a%2 == 0;
	}, &|a: usize| -> Vec<u128> {
	    return vec![0; a];
	}, &|a: &u128, b: usize| -> u128 {
	    return *a + (b as u128);
	}, false, 24, 256);
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
	let mut seq_dest: Vec<u128> = Vec::new();
	let filter_fn = |a: &u128| -> bool {
	    return *a%2 == 0;
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
        println!(">>> PACK [alloc move]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}", dur_par, dur_seq, N, par_dest.len());
    }

    // test parallel pack alloc move algorithm large
    #[test]
    fn test_parallel_pack_alloc_move_large() {
	// generate some random data
        let mut par_src: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par_src.push(64 + i*i - 8*i + 5);
        }
	let mut seq_src: Vec<u128> = par_src.clone();

        // time parallel algorithm
        let start_par = Instant::now();
	let (par_dest, len) = alloc_move_tune(&mut par_src[..], &|a: &u128| -> bool {
	    return *a%2 == 0;
	}, &|a: usize| -> Vec<u128> {
	    return vec![0; a];
	}, &|a: &u128, b: usize| -> u128 {
	    return *a + (b as u128);
	}, false, 24, 256);
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
	let mut seq_dest: Vec<u128> = Vec::new();
	let filter_fn = |a: &u128| -> bool {
	    return *a%2 == 0;
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
        println!(">>> PACK [alloc move]: parallel_pack = {:?}, sequential_pack = {:?}, source_len = {}, pack_len = {}", dur_par, dur_seq, N, par_dest.len());
    }
    
}
