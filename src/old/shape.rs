use std::ops::{Deref, DerefMut};
use std::iter::{Iterator};
use std::fmt::{Debug, Formatter, Result};
use arrayvec::{ArrayVec};

type Dims = ArrayVec<[usize; 8]>;

#[derive(Default, Clone, Eq, PartialEq)]
pub struct Shape {
  d: Dims
}

impl Shape {
  pub fn with_len(n: usize) -> Self {
    let mut d = Dims::default();
    debug_assert!(n <= d.capacity());
    unsafe { d.set_len(n); }
    Self{d}
  }
  pub fn new<D>(d: D) -> Self
    where D: AsRef<[usize]> {
    let d = d.as_ref();
    let mut s = Self::with_len(d.len());
    s.deref_mut().copy_from_slice(d);
    s
  }
  pub fn broadcast<A, B>(a: A, b: B) -> Self
    where A: AsRef<[usize]>,
          B: AsRef<[usize]> {
    let a = a.as_ref();
    let b = b.as_ref();
    let mut ait = a.into_iter().skip_while(|&&i| i == 1);
    let mut bit = b.into_iter().skip_while(|&&i| i == 1);
    debug_assert!({
      let atail = &a[a.len()-ait.by_ref().count()..];
      let btail = &b[b.len()-bit.by_ref().count()..];
      let msg = format!(
        "Can't broadcast {:?} and {:?}, tails {:?} and {:?} do not match!", 
        &a, &b, &atail, &btail
      );
      assert!(atail == btail || atail.len() == 0 || btail.len() == 0, msg);
      true
    });
    let s = Self::new(if ait.count() >= bit.count() {a} else {b});
    s
  }
     
  pub fn len(&self) -> usize {
    self.d.len()
  }
  pub fn size(&self) -> usize {
    self.iter().product::<usize>()
  }
}

impl Debug for Shape {
  fn fmt(&self, f: &mut Formatter) -> Result {
    self.deref().fmt(f)
  }
}

impl Deref for Shape {
  type Target = [usize];
  fn deref(&self) -> &Self::Target {
    &self.d
  }
}

impl DerefMut for Shape {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.d
  }
}

impl AsRef<[usize]> for Shape {
  fn as_ref(&self) -> &[usize] {
    self.d.as_ref()
  }
}
