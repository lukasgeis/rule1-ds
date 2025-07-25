use std::marker::PhantomData;

use stream_bitset::PrimIndex;

use crate::graph::Node;

#[derive(Clone, Default)]
pub struct Marker<T: Clone + Eq, I: PrimIndex> {
    data: Vec<T>,
    default: T,
    _index: PhantomData<I>,
}

pub type NodeMarker = Marker<Node, Node>;

impl<T: Clone + Eq, I: PrimIndex> Marker<T, I> {
    #[inline(always)]
    pub fn new(n: I, default: T) -> Self {
        Marker {
            data: vec![default.clone(); n.to_usize().unwrap()],
            default,
            _index: Default::default(),
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.data.iter_mut().for_each(|i| *i = self.default.clone());
    }

    #[inline(always)]
    pub fn get_mark(&self, index: I) -> T {
        self.data[index.to_usize().unwrap()].clone()
    }

    #[inline(always)]
    pub fn is_marked_with(&self, index: I, marker: T) -> bool {
        self.data[index.to_usize().unwrap()] == marker
    }

    #[inline(always)]
    pub fn mark_with(&mut self, index: I, marker: T) {
        self.data[index.to_usize().unwrap()] = marker;
    }

    #[inline(always)]
    pub fn mark_all_with<Iter: Iterator<Item = I>>(&mut self, indices: Iter, marker: T) {
        for i in indices {
            self.mark_with(i, marker.clone());
        }
    }
}
