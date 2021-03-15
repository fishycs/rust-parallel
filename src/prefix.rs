use super::SEQUENTIAL_THRESHOLD;
use num_cpus;
use std::ops::AddAssign;

// parallel prefix sum (in place)
pub fn parallel_prefix_sum<T: Send>(array: &mut [T])
where
    for<'t> T: AddAssign<&'t T>,
{
    parallel_prefix_sum_tune(array, num_cpus::get(), SEQUENTIAL_THRESHOLD);
}

// parallel prefix sum (in place)
pub fn parallel_prefix_sum_tune<T: Send>(array: &mut [T], threads: usize, seq_threshold: usize)
where
    for<'t> T: AddAssign<&'t T>,
{
    parallel_prefix_tune(
        array,
        &|a: &mut T, b: &T| {
            *a += b;
        },
        threads,
        seq_threshold,
    );
}

// parallel prefix (in place)
pub fn parallel_prefix<T: Send, F: Fn(&mut T, &T) + Sync>(array: &mut [T], accumulate_fn: &F) {
    parallel_prefix_tune(array, accumulate_fn, num_cpus::get(), SEQUENTIAL_THRESHOLD);
}

// parallel prefix (in place)
pub fn parallel_prefix_tune<T: Send, F: Fn(&mut T, &T) + Sync>(
    array: &mut [T],
    accumulate_fn: &F,
    threads: usize,
    seq_threshold: usize,
) {
    if threads == 0 {
        panic!("threads cannot be zero!");
    }
    if seq_threshold == 0 {
        panic!("seq_threshold cannot be zero!");
    }

    // collect and build sums
    parallel_prefix_accumulate(array, accumulate_fn, threads, seq_threshold);

    // finalize and distribute sums
    let len: usize = array.len() - 1;
    parallel_prefix_distribute(
        &mut array[..len],
        accumulate_fn,
        threads,
        seq_threshold,
        true,
    );
}

// collect and build sums
fn parallel_prefix_accumulate<T: Send, F: Fn(&mut T, &T) + Sync>(
    array: &mut [T],
    accumulate_fn: &F,
    threads: usize,
    seq_threshold: usize,
) {
    if threads == 1 || array.len() <= seq_threshold {
        // sequential
        let (array, end) = array.split_at_mut(array.len() - 1);
        let sum: &mut T = &mut end[0];
        for t in array {
            accumulate_fn(sum, t);
        }
    } else {
        // parallel
        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        // have to split, since each thread requires unique access...
        let (left, right) = array.split_at_mut(array.len() / 2);

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_prefix_accumulate(left, accumulate_fn, left_threads, seq_threshold);
            });
            parallel_prefix_accumulate(right, accumulate_fn, right_threads, seq_threshold);
        })
        .unwrap();

        accumulate_fn(&mut right[right.len() - 1], &left[left.len() - 1]); // sum accumulation
    }
}

// finalize and distribute sums
fn parallel_prefix_distribute<T: Send, F: Fn(&mut T, &T) + Sync>(
    array: &mut [T],
    accumulate_fn: &F,
    threads: usize,
    seq_threshold: usize,
    start: bool,
) {
    if threads == 1 || array.len() <= seq_threshold {
        // sequential
        for i in 1..array.len() {
            let (prev, now) = array.split_at_mut(i);
            accumulate_fn(&mut now[0], &prev[i - 1]);
        }
    } else {
        // parallel
        let left_threads: usize = threads / 2;
        let right_threads: usize = threads - left_threads;

        let left;
        let right;

        if start {
            let (left_array, right_array) = array.split_at_mut((array.len() + 1) / 2 - 1);
            left = left_array;
            right = right_array;
        } else {
            let (left_array, right_array) = array.split_at_mut(array.len() / 2);
            left = left_array;
            right = right_array;

            accumulate_fn(&mut right[0], &left[0]); // from left accumulation
        }

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_prefix_distribute(left, accumulate_fn, left_threads, seq_threshold, start);
            });
            parallel_prefix_distribute(right, accumulate_fn, right_threads, seq_threshold, false);
        })
        .unwrap();
    }
}

// run some tests
#[cfg(test)]
mod tests {
    use crate::prefix::*;
    use std::time::Instant;

    const N: usize = 10000000;
    // test parallel prefix sum algorithm with a large set of random data
    #[test]
    fn test_parallel_prefix_sum() {
        // generate some random data
        let mut par: Vec<u128> = Vec::new();
        for i in 0..N {
            let i: u128 = i as u128;
            par.push(64 + i * i - 8 * i + 5);
        }
        let mut seq: Vec<u128> = par.clone();

        // time parallel algorithm
        let start_par = Instant::now();
        parallel_prefix_sum(&mut par[..]);
        let dur_par = start_par.elapsed();

        // time sequential algorithm
        let start_seq = Instant::now();
        for i in 1..seq.len() {
            seq[i] += seq[i - 1];
        }
        let dur_seq = start_seq.elapsed();

        // check integrity of results
        assert_eq!(seq, par);

        // print results
        println!("parallel = {:?}, sequential = {:?}", dur_par, dur_seq);
    }

    const K: usize = 100;
    // test a fully sequential version of parallel prefix sum
    #[test]
    fn test_parallel_prefix_sum_full_seq() {
        let mut arr = [1usize; K];
        parallel_prefix_sum_tune(&mut arr, 1, usize::MAX);
        let mut expected = [0; K];
        for i in 0..K {
            expected[i] = i + 1;
        }
        assert_eq!(expected, arr);
    }

    // test a fully parallel version of parallel prefix sum
    #[test]
    fn test_parallel_prefix_sum_full_par() {
        let mut arr = [1usize; K];
        parallel_prefix_sum_tune(&mut arr, usize::MAX, 1);
        let mut expected = [0; K];
        for i in 0..K {
            expected[i] = i + 1;
        }
        assert_eq!(expected, arr);
    }

    // tests to ensure parallel prefix panics if given bad tuning args
    #[test]
    #[should_panic]
    fn test_parallel_prefix_sum_bad_args_1() {
        let mut arr = [0; 0];
        parallel_prefix_sum_tune(&mut arr, 0, 1);
    }
    #[test]
    #[should_panic]
    fn test_parallel_prefix_sum_bad_args_2() {
        let mut arr = [0; 0];
        parallel_prefix_sum_tune(&mut arr, 0, 0);
    }
    #[test]
    #[should_panic]
    fn test_parallel_prefix_sum_bad_args_3() {
        let mut arr = [0; 0];
        parallel_prefix_sum_tune(&mut arr, 1, 0);
    }

    const J: usize = 20;
    // test with product operator
    #[test]
    fn test_parallel_prefix_product() {
        let mut arr = [2usize; J];
        let accumulate_fn = &|a: &mut usize, b: &usize| {
            *a *= b;
        };
        parallel_prefix_tune(&mut arr, accumulate_fn, 10, 2);
        let mut expected = [0; J];
        let mut product = 1;
        for i in 0..J {
            product *= 2;
            expected[i] = product;
        }
        assert_eq!(expected, arr);
    }
}
