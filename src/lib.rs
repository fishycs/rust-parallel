// implementation of common parallel algorithms in rust (only 1 currently)
// - parallel prefix sum

use num_cpus;
use num_traits::identities::Zero;
use std::{ops::AddAssign};

const SEQUENTIAL_THRESHOLD: usize = 16;

// struct to hold important information for the parallel prefix sum operations
struct PrefixTree<T>
where
    for<'t> T: AddAssign<&'t T>,
    T: Send,
    T: Zero,
{
    sum: T,
    from_left: T,
    left: Option<Box<PrefixTree<T>>>,
    right: Option<Box<PrefixTree<T>>>,
}

impl<T> PrefixTree<T>
where
    for<'t> T: AddAssign<&'t T>,
    T: Send,
    T: Zero,
{
    pub fn default() -> Self {
        PrefixTree::<T> {
            sum: T::zero(),
            from_left: T::zero(),
            left: None,
            right: None,
        }
    }
}

// runs a parallel prefix sum in place on this mutable slice
// type of indices of slide must implement Zero and AddAssign
pub fn parallel_prefix_sum<T>(array: &mut [T])
where
    for<'t> T: AddAssign<&'t T>,
    T: Send,
    T: Zero,
{
    if array.len() > 0 {
        let cores: usize = num_cpus::get();

        let mut tree = PrefixTree::<T>::default();

        // collect sums and build prefix tree
        pps_sum(array, &mut tree, cores);

        // finalize and distribute sums from prefix tree
        pps_distribute(array, &mut tree);
    }
}

// collect sums and build prefix tree from the array
fn pps_sum<T>(array: &mut [T], tree: &mut PrefixTree<T>, cores: usize)
where
    for<'t> T: AddAssign<&'t T>,
    T: Send,
    T: Zero,
{
    if cores == 1 || array.len() <= SEQUENTIAL_THRESHOLD {
        // sequential
        for v in array {
            tree.sum += v
        }
    } else {
        // parallel
        let mut left = PrefixTree::<T>::default();
        let mut right = PrefixTree::<T>::default();

        let left_cores: usize = cores / 2;
        let right_cores: usize = cores - left_cores;

        // have to split, since each thread requires unique access...
        let (left_arr, right_arr) = array.split_at_mut(array.len() / 2);
        crossbeam::scope(|scope| {
            scope.spawn(|_| {
                pps_sum(left_arr, &mut left, left_cores);
            });
            pps_sum(right_arr, &mut right, right_cores);
        })
        .unwrap(); // probably should handle this error somehow...

        tree.sum += &left.sum;
        tree.sum += &right.sum;
        tree.left = Some(Box::new(left));
        tree.right = Some(Box::new(right));
    }
}

// finalize and distribute sums from prefix tree to the array
fn pps_distribute<T>(array: &mut [T], tree: &mut PrefixTree<T>)
where
    for<'t> T: AddAssign<&'t T>,
    T: Send,
    T: Zero,
{
    match (tree.left.as_mut(), tree.right.as_mut()) {
        (Some(left), Some(right)) => {
            left.from_left += &tree.from_left;
            right.from_left += &tree.from_left;
            right.from_left += &left.sum;

            let (left_arr, right_arr) = array.split_at_mut(array.len() / 2);
            crossbeam::scope(|scope| {
                scope.spawn(|_| pps_distribute(left_arr, left));
                pps_distribute(right_arr, right);
            })
            .unwrap(); // probably should handle this error somehow...
        }
        _ => {
            let from_left: &T = &tree.from_left;
            for i in 1..array.len() {
                let (a, b) = array.split_at_mut(i);
                b[0] += &a[i - 1];
                array[i - 1] += from_left;
            }
            array[array.len() - 1] += from_left;
        }
    }
}

// run some tests
#[cfg(test)]
mod tests {
    use crate::parallel_prefix_sum;
    use std::time::Instant;

    const N: usize = 10000000;

    #[test]
    fn test_parallel_prefix_sum_small() {
        let mut arr = [1; 20];
        parallel_prefix_sum::<u128>(&mut arr);
        let mut expected = [0; 20];
        for i in 0..20 {
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
            par.push(64 + i * i - (8 * i) + 5);
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
        for i in 0..par.len() {
            assert_eq!(seq[i], par[i]);
        }

        // print results
        println!("parallel = {:?}, sequential = {:?}", dur_par, dur_seq);
    }
}
