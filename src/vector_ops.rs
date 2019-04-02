use crate::{Vector};
use std::ops::{Deref, Add, AddAssign, Mul};
use typenum::{Unsigned};
use generic_array::{ArrayLength};

pub fn dot<'a, 'b, T, N>(v1: &'a [T], v2: &'b [T]) -> T
  where T: Default + Copy + Add<T, Output=T> + AddAssign<T> + Mul<T, Output=T>,
        N: Unsigned + ArrayLength<T> {
    let a = v1[..N::to_usize()].as_ptr();
    let b = v2[..N::to_usize()].as_ptr();
    let mut s = [T::default(); 8];
    let mut i = 0;
    while i + 8 < N::to_usize() {
      s[0] = unsafe { a.add(i).read() * b.add(i).read() };
      s[1] = unsafe { a.add(i+1).read() * b.add(i+1).read() };
      s[2] = unsafe { a.add(i+2).read() * b.add(i+2).read() };
      s[3] = unsafe { a.add(i+3).read() * b.add(i+3).read() };
      s[4] = unsafe { a.add(i+4).read() * b.add(i+4).read() };
      s[5] = unsafe { a.add(i+5).read() * b.add(i+5).read() };
      s[6] = unsafe { a.add(i+6).read() * b.add(i+6).read() };
      s[7] = unsafe { a.add(i+7).read() * b.add(i+7).read() };
      i += 8;
    }
    let mut t = T::default();
    t += s[0] + s[4];
    t += s[1] + s[5];
    t += s[2] + s[6];
    t += s[3] + s[7];
    for u in i..N::to_usize() {
      t += unsafe { a.add(u).read() * b.add(u).read() };
    }
    t
  }
