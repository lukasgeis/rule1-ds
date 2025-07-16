use num::{FromPrimitive, ToPrimitive};

use crate::graph::Node;

/// Trait for all possible types that can be used as a key for a RadixHeap
pub trait RadixKey: Copy + Default + PartialOrd {
    /// Number of bits of Self
    const NUM_BITS: usize;

    /// Similarity to another instance of Self
    fn radix_similarity(&self, other: &Self) -> usize;
}

macro_rules! radix_key_impl_float {
    ($($t:ty),*) => {
        $(
            impl RadixKey for $t {
                const NUM_BITS: usize = (std::mem::size_of::<$t>() * 8);

                fn radix_similarity(&self, other: &Self) -> usize {
                    (self.to_bits() ^ other.to_bits()).leading_zeros() as usize
                }
            }
        )*
    };
}

macro_rules! radix_key_impl_int {
    ($($t:ty),*) => {
        $(
            impl RadixKey for $t {
                const NUM_BITS: usize = (std::mem::size_of::<$t>() * 8);

                fn radix_similarity(&self, other: &Self) -> usize {
                    (self ^ other).leading_zeros() as usize
                }
            }
        )*
    };
}

radix_key_impl_float!(f32, f64);
radix_key_impl_int!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/// Inverted radix similarity (=> how far away is other from self)
///
/// Not part of trait to not expose to user
fn radix_distance<K: RadixKey>(lhs: &K, rhs: &K) -> usize {
    let dist = K::NUM_BITS - lhs.radix_similarity(rhs);
    debug_assert!(dist <= K::NUM_BITS);
    dist
}

/// Marker trait for types that can be used as values in IndexedRadixHeap (=> have to be convertible to usize)
pub trait RadixValue: FromPrimitive + ToPrimitive + Default + Copy {}
impl<T: FromPrimitive + ToPrimitive + Default + Copy> RadixValue for T {}

/// Stores all key-value-pairs of the same similarity
type Bucket<K, V> = Vec<(K, V)>;

const NOT_IN_HEAP: usize = usize::MAX;

/// # IndexedRadixHeap: A Radix-MinHeap
///
/// Allows fast insertions/deletions/queries into elements sorted by associated RadixKey-type.
/// Elements have to be convertible to a usize smaller than a given value to allow for fast
/// existence queries.
///
/// ### IMPORTANT
/// NUM_BUCKETS must be equal to K::NUM_BITS + 1! This will be checked when creating the heap.
#[derive(Debug)]
pub struct IndexedRadixHeap<K: RadixKey, V: RadixValue, const NUM_BUCKETS: usize> {
    /// Number of elements in the heap
    len: usize,
    /// Current top element (last element removed)
    top: K,
    /// All buckets
    buckets: [Bucket<K, V>; NUM_BUCKETS],
    /// A pointer for each element to where it is in the heap
    pointer: Vec<(usize, usize)>,
}

/// A heap with nodes as keys and values
pub type NodeHeap = IndexedRadixHeap<Node, Node, 33>;

impl<K: RadixKey, V: RadixValue, const NUM_BUCKETS: usize> IndexedRadixHeap<K, V, NUM_BUCKETS> {
    /// Creates a new heap with a given top element and a maximum number of elements
    pub fn new(n: usize, top: K) -> Self {
        // Only accept heaps with enough buckets
        assert!(NUM_BUCKETS > K::NUM_BITS);
        Self {
            len: 0,
            top,
            buckets: array_init::array_init(|_| Vec::new()),
            pointer: vec![(NOT_IN_HEAP, NOT_IN_HEAP); n],
        }
    }

    /// Force pushes a (key, value)-pair on the heap. The caller is responsible for ensuring bounds
    /// and that value is not already on the heap.
    pub fn push(&mut self, key: K, value: V) {
        debug_assert!(self.pointer[value.to_usize().unwrap()].0 == NOT_IN_HEAP);

        let bucket = radix_distance(&key, &self.top);
        self.pointer[value.to_usize().unwrap()] = (bucket, self.buckets[bucket].len());
        self.buckets[bucket].push((key, value));
        self.len += 1;
    }

    /// Force removes a (key, value)-pair identified by the value from the heap.
    /// Panics if no value was found.
    pub fn remove(&mut self, value: V) -> K {
        let value = value.to_usize().unwrap();
        debug_assert!(value < self.pointer.len());

        let (bucket, position) = self.pointer[value];
        debug_assert_ne!(bucket, NOT_IN_HEAP);

        self._remove_inner(value, bucket, position)
    }

    /// Private function to remove an element from a specific position in a bucket and update the
    /// pointer. Returns the element. Panics if no entry was found.
    fn _remove_inner(&mut self, value: usize, bucket: usize, position: usize) -> K {
        let res = self.buckets[bucket].swap_remove(position);
        if self.buckets[bucket].len() > position {
            self.pointer[self.buckets[bucket][position].1.to_usize().unwrap()].1 = position;
        }
        self.len -= 1;

        self.pointer[value].0 = NOT_IN_HEAP;

        res.0
    }

    /// Updates the heap to find the new smallest element and re-order buckets accordingly
    fn update_buckets(&mut self) {
        let (buckets, repush) = match self.buckets.iter().position(|bucket| !bucket.is_empty()) {
            None | Some(0) => return,
            Some(index) => {
                let (buckets, rest) = self.buckets.split_at_mut(index);
                (buckets, &mut rest[0])
            }
        };

        self.top = repush
            .iter()
            .min_by(|(k1, _), (k2, _)| k1.partial_cmp(k2).unwrap())
            .unwrap()
            .0;

        repush.drain(..).for_each(|(key, value)| {
            let bucket = radix_distance(&key, &self.top);
            self.pointer[value.to_usize().unwrap()] = (bucket, buckets[bucket].len());
            buckets[bucket].push((key, value));
        });
    }

    /// Pops the smallest element from the heap (no tiebreaker).
    /// Returns *None* if no element is left on the heap.
    pub fn pop(&mut self) -> Option<(K, V)> {
        let (key, val) = self.buckets[0].pop().or_else(|| {
            self.update_buckets();
            self.buckets[0].pop()
        })?;

        self.len -= 1;
        self.pointer[val.to_usize().unwrap()].0 = NOT_IN_HEAP;

        Some((key, val))
    }
}
