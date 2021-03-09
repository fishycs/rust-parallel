// implementation of common parallel algorithms in rust (only 1 currently)
// - parallel prefix sum

use std::thread;
use std::ops::AddAssign;

extern crate num_traits;
use num_traits::identities::Zero;

extern crate num_cpus;

const SEQUENTIAL_THRESHOLD: usize = 16;

// struct to hold important information for the parallel prefix sum operations
struct PrefixTree<T> {
    low: usize,
    high: usize,
    sum: T,
    from_left: T,
    left: Option<Box<PrefixTree<T>>>,
    right: Option<Box<PrefixTree<T>>>,
}

// runs a parallel prefix sum in place on this mutable slice
// type of indices of slide must implement Zero and AddAssign
pub fn parallel_prefix_sum<T: Zero>(array: &mut[T]) where for<'t> T: AddAssign<&'t T> {
    if array.len() > 0 {
	let cores: usize = num_cpus::get();
	
	let array_ptr: *mut T = array.as_mut_ptr();
	
	let mut tree: PrefixTree<T> = PrefixTree::<T>{low: 0, high: array.len(), sum: T::zero(), from_left: T::zero(), left: None, right: None};
	
	// collect sums and build prefix tree
	pps_sum(array_ptr, &mut tree, cores);
	
	// finalize and distribute sums from prefix tree
	pps_distribute(array_ptr, &mut tree);
    }
}

// collect sums and build prefix tree from the array
fn pps_sum<T: Zero>(array_ptr: *mut T, tree_ptr: &mut PrefixTree<T>, cores: usize) where for<'t> T: AddAssign<&'t T> {
    let low: usize = tree_ptr.low;
    let high: usize = tree_ptr.high;

    if cores == 1 || high - low <= SEQUENTIAL_THRESHOLD {
	// sequential

	unsafe {
	    for i in low..high {
		tree_ptr.sum += &*array_ptr.add(i);
	    }
	}
    } else {
	// parallel

	let mid: usize = low + (high - low)/2;

	let left: PrefixTree<T> = PrefixTree::<T>{low, high: mid, sum: T::zero(), from_left: T::zero(), left: None, right: None};
	let right: PrefixTree<T> = PrefixTree::<T>{low: mid, high, sum: T::zero(), from_left: T::zero(), left: None, right: None};
	tree_ptr.left = Some(Box::new(left));
	tree_ptr.right = Some(Box::new(right));

	let left_ptr: &mut PrefixTree<T>;
	let right_ptr: &mut PrefixTree<T>;
	unsafe {
	    left_ptr = option_box_to_mut_ref(&mut tree_ptr.left);
	    right_ptr = option_box_to_mut_ref(&mut tree_ptr.right);
	}

	let left_cores: usize = cores/2;
	let right_cores: usize = cores - left_cores;

	let array_disguise: usize = array_ptr as usize;
	let left_disguise: usize = left_ptr as *mut PrefixTree<T> as usize;

	let handle: thread::JoinHandle<()>;
	unsafe {
	    handle = thread::spawn(move || {
		pps_sum(array_disguise as *mut T, (left_disguise as *mut PrefixTree<T>).as_mut().unwrap(), left_cores);
	    });
	}
	pps_sum(array_ptr, right_ptr, right_cores);
	handle.join().unwrap();

	tree_ptr.sum += &left_ptr.sum;
	tree_ptr.sum += &right_ptr.sum;
    }
}

// finalize and distribute sums from prefix tree to the array
fn pps_distribute<T>(array_ptr: *mut T, tree_ptr: &mut PrefixTree<T>) where for<'t> T: AddAssign<&'t T> {
    if tree_ptr.left.is_none() {
	// sequential
	
	let low: usize = tree_ptr.low;
	let high: usize = tree_ptr.high;
	
	let from_left: &T = &tree_ptr.from_left;

	unsafe {
	    for i in low + 1..high {
		*array_ptr.add(i) += &*array_ptr.add(i - 1);
		*array_ptr.add(i - 1) += from_left;
	    }
	    *array_ptr.add(high - 1) += from_left;
	}
    } else {
	// parallel

	let left_ptr: &mut PrefixTree<T>;
	let right_ptr: &mut PrefixTree<T>;
	unsafe {
	    left_ptr = option_box_to_mut_ref(&mut tree_ptr.left);
	    right_ptr = option_box_to_mut_ref(&mut tree_ptr.right);
	}
	
	left_ptr.from_left += &tree_ptr.from_left;
	right_ptr.from_left += &tree_ptr.from_left;
	right_ptr.from_left += &left_ptr.sum;
	
	let array_disguise: usize = array_ptr as usize;
	let left_disguise: usize = left_ptr as *mut PrefixTree<T> as usize;

	let handle: thread::JoinHandle<()>;
	unsafe {
	    handle = thread::spawn(move || {
		pps_distribute(array_disguise as *mut T, (left_disguise as *mut PrefixTree<T>).as_mut().unwrap());
	    });
	}
	pps_distribute(array_ptr, right_ptr);
	handle.join().unwrap();
    }
}

// convert a mutable reference to an option of a box into a mutable reference to the contents of the box
// unsafe since this method will crash if the option is None
unsafe fn option_box_to_mut_ref<T>(var: &mut Option<Box<T>>) -> &mut T {
    return (&mut**var.as_mut().unwrap() as *mut T).as_mut().unwrap();
}

// run some tests
#[cfg(test)]
mod tests {
    use std::time::{Instant};

    const N: usize = 10000000;

    // test parallel prefix sum algorithm
    #[test]
    fn test_parallel_prefix_sum() {
	// generate some random data
	let mut par: Vec<u128> = Vec::new();
	for i in 0..N {
	    let i: u128 = i as u128;
	    par.push(64+i*i-(8*i) + 5);
	}
	let mut seq: Vec<u128> = par.clone();

	// time parallel algorithm
	let start_par = Instant::now();
	crate::parallel_prefix_sum::<u128>(&mut par[..]);
	let dur_par = start_par.elapsed();

	// time sequential algorithm
	let start_seq = Instant::now();
	for i in 1..seq.len() {
	    seq[i] += seq[i - 1];
	}
	let dur_seq = start_seq.elapsed();
	
	// check integrity of results
	for i in 0..par.len() {
	    assert_eq!(seq[i], par[i]);
	}

	// print results
	println!("parallel = {:?}, sequential = {:?}", dur_par, dur_seq);
    }
}
