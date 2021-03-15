use num_cpus;
use std::ops::AddAssign;
use super::SEQUENTIAL_THRESHOLD;

// parallel prefix sum (in place)
pub fn parallel_prefix_sum<T: Send>(array: &mut [T])
where
    for<'t> T: AddAssign<&'t T>,
{
    parallel_prefix_sum_tune(array, num_cpus::get(), SEQUENTIAL_THRESHOLD);
}

// parallel prefix sum (in place)
pub fn parallel_prefix_sum_tune<T: Send>(array: &mut [T], cores: usize, threshold: usize)
where
    for<'t> T: AddAssign<&'t T>,
{
    parallel_prefix_tune(
        array,
        &|a: &mut T, b: &T| {
            *a += b;
        },
        cores,
        threshold,
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
    mut cores: usize,
    mut threshold: usize,
) {
    if cores == 0 {
	cores = 1;
    }
    if threshold == 0 {
	threshold = 1;
    }
    
    // collect and build sums
    parallel_prefix_accumulate(array, accumulate_fn, cores, threshold);

    // finalize and distribute sums
    let len: usize = array.len() - 1;
    parallel_prefix_distribute(&mut array[..len], accumulate_fn, cores, threshold, true);
}

// collect and build sums
fn parallel_prefix_accumulate<T: Send, F: Fn(&mut T, &T) + Sync>(
    array: &mut [T],
    accumulate_fn: &F,
    cores: usize,
    threshold: usize,
) {
    if cores == 1 || array.len() <= threshold {
        // sequential
        let (array, end) = array.split_at_mut(array.len() - 1);
        let sum: &mut T = &mut end[0];
        for t in array {
            accumulate_fn(sum, t);
        }
    } else {
        // parallel
        let left_cores: usize = cores / 2;
        let right_cores: usize = cores - left_cores;

        // have to split, since each thread requires unique access...
        let (left, right) = array.split_at_mut(array.len() / 2);

        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                parallel_prefix_accumulate(left, accumulate_fn, left_cores, threshold);
            });
            parallel_prefix_accumulate(right, accumulate_fn, right_cores, threshold);
        })
        .unwrap(); // probably should handle this error somehow...

        accumulate_fn(&mut right[right.len() - 1], &left[left.len() - 1]); // sum accumulation
    }
}

// finalize and distribute sums
fn parallel_prefix_distribute<T: Send, F: Fn(&mut T, &T) + Sync>(
    array: &mut [T],
    accumulate_fn: &F,
    cores: usize,
    threshold: usize,
    start: bool,
) {
    if cores == 1 || array.len() <= threshold {
        // sequential
        for i in 1..array.len() {
            let (prev, now) = array.split_at_mut(i);
            accumulate_fn(&mut now[0], &prev[i - 1]);
        }
    } else {
        // parallel
        let left_cores: usize = cores / 2;
        let right_cores: usize = cores - left_cores;

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
                parallel_prefix_distribute(left, accumulate_fn, left_cores, threshold, start);
            });
            parallel_prefix_distribute(right, accumulate_fn, right_cores, threshold, false);
        })
        .unwrap();
    }
}

// run some tests
#[cfg(test)]
mod tests {
    use crate::prefix::parallel_prefix_sum;
    use std::time::Instant;

    const N: usize = 10000000;
    const K: usize = 20;

    #[test]
    fn test_parallel_prefix_sum_small() {
        let mut arr = [1; K];
        parallel_prefix_sum::<u128>(&mut arr);
        let mut expected = [0; K];
        for i in 0..K {
            expected[i] = (i as u128) + 1;
        }
        assert_eq!(expected, arr);
    }

    // test parallel prefix sum algorithm
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
        parallel_prefix_sum::<u128>(&mut par[..]);
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
}
