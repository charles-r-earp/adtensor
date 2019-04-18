use std::mem;
use arrayvec::{ArrayVec};
use smallvec::{SmallVec};

pub struct Array<A>
  where A: arrayvec::Array {
  a: ArrayVec<A>
}

unsafe impl<A> smallvec::Array for Array<A>
  where A: arrayvec::Array {
  type Item = <A as arrayvec::Array>::Item;
  fn size() -> usize {
    mem::size_of::<A>() / mem::size_of::<<A as arrayvec::Array>::Item>()
  }
  fn ptr(&self) -> *const Self::Item {
    self.a.as_slice().as_ptr()
  }
  fn ptr_mut(&mut self) -> *mut Self::Item {
    self.a.as_mut_slice().as_mut_ptr()
  }
}

pub type SmallArrayVec<A> = SmallVec<Array<A>>;
