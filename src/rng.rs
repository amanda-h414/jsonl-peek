//! A small seedable random number generator and a reservoir sampler built on
//! top of it.
//!
//! Both exist so `sample` can produce a reproducible, uniform subset of a
//! file without knowing its length in advance, using nothing outside the
//! standard library.
//!
//! ```
//! use jsonl_peek::rng::{Reservoir, SplitMix64};
//!
//! let mut rng = SplitMix64::new(42);
//! assert!(rng.below(10) < 10);
//!
//! let mut reservoir = Reservoir::new(2, 42);
//! for line in 0..100 {
//!     reservoir.add(line);
//! }
//! assert_eq!(reservoir.as_slice().len(), 2);
//! assert_eq!(reservoir.seen(), 100);
//! ```

/// A SplitMix64 generator.
///
/// SplitMix64 is not cryptographically secure, but it is fast, has a full
/// 64-bit period, and - unlike the standard library's default hasher - is
/// stable across Rust versions and platforms, which matters because
/// `--seed` promises the same sample every time it is run.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seeds the generator. Any seed, including 0, produces a full-period
    /// stream; the first output does not depend on the seed in an obvious
    /// way, so 0 is not a weak seed here.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Returns the next 64 bits of output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a value drawn uniformly from `0..bound`.
    ///
    /// Uses Lemire's rejection method rather than `next_u64() % bound`,
    /// which is biased towards small values whenever `bound` does not
    /// evenly divide 2^64 - exactly the case for reservoir sizes that
    /// matter here.
    ///
    /// Panics if `bound` is 0.
    pub fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "below: bound must be positive");
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = (self.next_u64() as u128) * (bound as u128);
            if (product as u64) >= threshold {
                return (product >> 64) as u64;
            }
        }
    }
}

/// A fixed-capacity, uniform random sample of a stream of unknown length.
///
/// Implements Algorithm R (Vitter, 1985): the first `capacity` items offered
/// are kept outright, and each item after that replaces a uniformly chosen
/// slot with probability `capacity / seen`. After any number of calls to
/// [`add`](Reservoir::add), every item offered so far has had an equal
/// `capacity / seen` chance of surviving into the sample.
pub struct Reservoir<T> {
    rng: SplitMix64,
    capacity: usize,
    seen: u64,
    items: Vec<T>,
}

impl<T> Reservoir<T> {
    /// Creates a reservoir that holds at most `capacity` items, seeded for
    /// reproducibility.
    pub fn new(capacity: usize, seed: u64) -> Self {
        Reservoir {
            rng: SplitMix64::new(seed),
            capacity,
            seen: 0,
            items: Vec::with_capacity(capacity),
        }
    }

    /// Offers `item` to the sample.
    pub fn add(&mut self, item: T) {
        self.seen += 1;
        if self.capacity == 0 {
            return;
        }
        if self.items.len() < self.capacity {
            self.items.push(item);
            return;
        }
        let slot = self.rng.below(self.seen);
        if slot < self.capacity as u64 {
            self.items[slot as usize] = item;
        }
    }

    /// The number of items offered so far, including any not kept.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Borrows the current sample, in reservoir order (not input order).
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    /// Consumes the reservoir, returning the sample in reservoir order.
    pub fn into_vec(self) -> Vec<T> {
        self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_stream() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let a_vals: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let b_vals: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(a_vals, b_vals);
    }

    #[test]
    fn seed_zero_is_not_degenerate() {
        let mut rng = SplitMix64::new(0);
        let vals: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
        assert!(vals.iter().any(|v| *v != vals[0]));
    }

    #[test]
    fn below_never_reaches_bound() {
        let mut rng = SplitMix64::new(123);
        for _ in 0..10_000 {
            assert!(rng.below(7) < 7);
        }
        // A bound of 1 has exactly one possible outcome.
        for _ in 0..100 {
            assert_eq!(rng.below(1), 0);
        }
    }

    #[test]
    fn below_is_roughly_uniform() {
        let mut rng = SplitMix64::new(99);
        let mut counts = [0u32; 5];
        let draws = 50_000;
        for _ in 0..draws {
            counts[rng.below(5) as usize] += 1;
        }
        let expected = draws as f64 / counts.len() as f64;
        for count in counts {
            let relative_error = (count as f64 - expected).abs() / expected;
            assert!(relative_error < 0.05, "count {count} too far from {expected}");
        }
    }

    #[test]
    fn reservoir_keeps_everything_under_capacity() {
        let mut r = Reservoir::new(10, 1);
        for i in 0..5 {
            r.add(i);
        }
        assert_eq!(r.seen(), 5);
        assert_eq!(r.into_vec(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn reservoir_never_exceeds_capacity() {
        let mut r = Reservoir::new(3, 1);
        for i in 0..1000 {
            r.add(i);
        }
        assert_eq!(r.seen(), 1000);
        assert_eq!(r.as_slice().len(), 3);
    }

    #[test]
    fn zero_capacity_reservoir_stays_empty() {
        let mut r = Reservoir::new(0, 1);
        for i in 0..10 {
            r.add(i);
        }
        assert_eq!(r.seen(), 10);
        assert!(r.into_vec().is_empty());
    }

    #[test]
    fn reservoir_sample_is_roughly_uniform() {
        // 200 trials of sampling 3 items out of 10; each item's long run
        // inclusion rate should land near 3/10 regardless of position.
        const N: u64 = 10;
        const K: usize = 3;
        const TRIALS: u64 = 4000;
        let mut hits = [0u32; N as usize];
        for seed in 0..TRIALS {
            let mut r = Reservoir::new(K, seed);
            for i in 0..N {
                r.add(i);
            }
            for item in r.into_vec() {
                hits[item as usize] += 1;
            }
        }
        let expected = TRIALS as f64 * K as f64 / N as f64;
        for hit in hits {
            let relative_error = (hit as f64 - expected).abs() / expected;
            assert!(relative_error < 0.1, "hit count {hit} too far from {expected}");
        }
    }
}
