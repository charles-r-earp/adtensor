use std::cell::RefCell;
use std::iter::{Iterator, FromIterator, IntoIterator, ExactSizeIterator};
use std::ops::{Deref, DerefMut, Add, Mul, AddAssign};
use num_traits::{Zero, One};


#[derive(Default, Clone, Copy, Debug)]
pub struct Node<T> {
  p: [usize; 2],
  d: [T; 2]
}

#[derive(Default, Debug)]
pub struct Expr<T> {
  n: RefCell<Vec<Node<T>>>
}  

#[derive(Debug, Clone)]
pub struct Tensor<'e, T, V> {
  e: &'e Expr<T>,
  k: usize, // k == 0 for constants
  s: Vec<usize>,
  v: V
}

pub type TensorOwned<'e, T> = Tensor<'e, T, Vec<T>>;
pub type TensorView<'a, 'e, T> = Tensor<'e, T, &'a [T]>;


pub trait IntoTensorData {
  type Item;
  fn data(self) -> Vec<Self::Item>;
}

pub fn broadcast<'a, 'b>(a: &'a Vec<usize>, b: &'b Vec<usize>) -> Vec<usize>
  where {
  debug_assert_eq!(a, b);
  a.clone()
}

impl<T> Expr<T> {
  pub fn new() -> Self
    where T: Default {
    Self::default()
  }
  pub fn var<'e>(&'e self, s: Vec<usize>, v: Vec<T>) -> TensorOwned<'e, T>
    where T: One + Copy {
    let mut n = self.n.borrow_mut();
    let p = s.iter().product::<usize>();
    debug_assert_eq!(p, v.len());
    let l = n.len();
    n.resize(l + p, Node{p: [0; 2], d: [T::one(); 2]}); 
    Tensor{e: &self, k: l+1, s, v} 
  }
  pub fn constant<'e>(&'e self, s: Vec<usize>, v: Vec<T>) -> TensorOwned<'e, T>
    where T: Zero + Copy {
    let mut n = self.n.borrow_mut();
    let p = s.iter().product::<usize>();
    debug_assert_eq!(p, v.len());
    Tensor{e: &self, k: 0, s, v} 
  }
  fn push<'e, Z>(&'e self, ks: [usize; 2], d: Z) -> usize
    where T: Zero + Copy,
          Z: ExactSizeIterator<Item=(T, T)> {
    let mut n = self.n.borrow_mut();
    let l = n.len();  
    let mut ps = [0, 0];
    n.resize(l + d.len(), Node{p: [0; 2], d: [T::zero(); 2]});
    n[l..].iter_mut()
          .zip(d)
          .enumerate()
          .for_each(|(i, (n, (d1, d2)))| {
      if ks[0] != 0 {
        n.p[0] = ks[0] + i;
        n.d[0] = d1;
      }
      if ks[1] != 0 {
        n.p[1] = ks[1] + i;
        n.d[1] = d2;
      }
    });
    l + 1
  }
}

impl<'e, T, V> Tensor<'e, T, V> {
  pub fn grad(&self) -> TensorOwned<'e, T>
    where T: Copy + Zero + One + AddAssign,
          V: Deref<Target=[T]> {
    let mut n = self.e.n.borrow_mut();
    let mut g = TensorOwned{e: self.e, k: 0, s: vec![n.len()], v: vec![T::zero(); n.len()]};
    if self.k == 0 {
      return g;
    } 
    let i = self.k - 1;
    let l = self.v.deref().len();
    g.iter_mut()
     .skip(i)
     .take(l)
     .for_each(|d| *d = T::one());
        
    n.iter()
     .enumerate()
     .rev()
     .for_each(|(i, n)| {
      let d = g[i];
      if n.p[0] != 0 {
        g[n.p[0]-1] += n.d[0] * d;
      }
      if n.p[1] != 0 {
        g[n.p[1]-1] += n.d[1] * d;
      } 
    });
    g
  }
}

impl<'e, T, V> Deref for Tensor<'e, T, V>
  where V: Deref<Target=[T]> {
  type Target = V;
  fn deref(&self) -> &Self::Target {
    &self.v
  }
}

impl<'e, T, V> DerefMut for Tensor<'e, T, V>
  where V: DerefMut<Target=[T]> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.v
  }
}

impl<'a, 'b, 'e, T, A, B> Mul<&'b Tensor<'e, T, B>> for &'a Tensor<'e, T, A>
  where T: Add + Mul<Output=T> + Copy + Zero,
        A: Deref<Target=[T]>,
        B: Deref<Target=[T]> {
  type Output = TensorOwned<'e, T>;
  fn mul(self, rhs: &'b Tensor<'e, T, B>) -> Self::Output {
    debug_assert_eq!(self.e as *const _, rhs.e as *const _);
    let k: usize;
    if self.k != 0 || rhs.k != 0 {
      k = self.e.push([self.k, rhs.k], self.iter()
                                           .zip(rhs.iter())
                                           .map(|(&a, &b)| (b, a))
      );
    }
    else {
      k = 0;
    }
    TensorOwned{e: self.e, k, s: broadcast(&self.s, &rhs.s), 
                v: Vec::from_iter(self.iter()
                                      .zip(rhs.iter())
                                      .map(|(&a, &b)| a * b)
    )}
  }
}
