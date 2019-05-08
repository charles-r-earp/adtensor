use crate::adscalar::ADScalar;
use crate::shape::Shape;
use crate::tensor::{Tensor, Matmul};
use crate::adtensor::{ADTensor};
use std::iter::{Iterator, empty};
use std::marker::PhantomData;
use std::mem;
use std::cmp::PartialOrd;
use rand::distributions::{Distribution, Normal};
use num_traits::{Float, Zero, One};
use num_traits::cast::{NumCast, ToPrimitive};

pub trait Function<T> {
  fn forward<'b, 'g>(&self, x: &'b ADTensor<'g, T>) -> ADTensor<'g, T>;
  fn eval<'b>(&self, x: &'b Tensor<T>) -> Tensor<T>;
}

pub trait Layer<T>: Function<T> {
  fn collect_params<'b, 'p>(&self, p: &'b mut Vec<&'p Tensor<T>>) {}
  fn params<'a>(&'a self) -> Vec<&'a Tensor<T>> {
    let mut p = Vec::new();
    self.collect_params(&mut p);
    p.sort_unstable_by_key(|p| p.param_key());
    p
  }
  fn collect_params_mut<'b, 'p>(&mut self, p: &'b mut Vec<&'p mut Tensor<T>>) {}
  fn params_mut<'a>(&'a mut self) -> Vec<&'a mut Tensor<T>> {
    let mut p = Vec::new();
    self.collect_params_mut(&mut p);
    p.sort_unstable_by_key(|p| p.param_key());
    p
  }
  fn build<'b>(&mut self, s: &'b Shape) -> Shape;
}

pub trait Initializer<T> {
  fn init(&self, s: Shape) -> Tensor<T>;
}

pub struct BasicInit<T> {
  _p: PhantomData<T>
}

impl<T> BasicInit<T> {
  pub fn new() -> Self {
    Self{_p: PhantomData::default()}
  }
}

impl<T> Initializer<T> for BasicInit<T>
  where T: Float {
  fn init(&self, s: Shape) -> Tensor<T> {
    let d0 = 
      if s.len() > 0 {s[0]}
      else {1};
    let d1 = 
      if s.len() > 1 {s[1]}
      else {1};
    // https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    let n = Normal::new(0., (2./(d0 + d1).to_f64().unwrap()).sqrt());
    Tensor::shape(s).init_fn(|| T::from(n.sample(&mut rand::thread_rng())).unwrap())
  }
}

pub trait Activation<T>: Function<T> + Initializer<T> {
  fn is_identity() -> bool { false }
}

pub struct Step<T> {
  i: BasicInit<T>
}

impl<T> Step<T> {
  pub fn new() -> Self {
    Self{i: BasicInit::new()}
  }
}

impl<T> Function<T> for Step<T>
  where T: Zero + One + Copy + PartialOrd {
  fn forward<'b, 'g>(&self, x: &'b ADTensor<'g, T>) -> ADTensor<'g, T> {
    x.map(|x| if x.v() > T::zero() {ADScalar::new(T::one())}
              else {ADScalar::zero()})
  } 
  fn eval<'b>(&self, x: &'b Tensor<T>) -> Tensor<T> {
    x.map(|x| if x > T::zero() {T::one()} 
              else {T::zero()})
  }
}

impl<T> Initializer<T> for Step<T>
  where T: Float {
  fn init(&self, s: Shape) -> Tensor<T> {
    self.i.init(s)
  }
}

impl<T> Activation<T> for Step<T>
  where T: Float {}

pub struct Identity<T> {
  i: BasicInit<T>
}

impl<T> Identity<T> {
  fn new() -> Self {
    Self{i: BasicInit::new()}
  }
}

impl<T> Initializer<T> for Identity<T>
  where T: Float {
  fn init(&self, s: Shape) -> Tensor<T> {
    self.i.init(s)
  }
}

impl<T> Function<T> for Identity<T> {
  fn forward<'b, 'g>(&self, x: &'b ADTensor<'g, T>) -> ADTensor<'g, T> {
    panic!();
  }
  fn eval<'b>(&self, x: &'b Tensor<T>) -> Tensor<T> {
    panic!();
  }
}

pub struct Linear<T, A> {
  w: Tensor<T>,
  b: Option<Tensor<T>>,
  a: A
}

impl<T> Linear<T, Identity<T>> {
  pub fn c(c: usize) -> Self {
    Self{w: Tensor::shape(vec![0, c]),
         b: None,
         a: Identity::new()}
  }
  pub fn act<A>(self, a: A) -> Linear<T, A>
    where A: Activation<T> {
    Linear{w: self.w,
           b: self.b,
           a: a}
  }
}

impl<T, A> Linear<T, A> { 
  pub fn bias(self) -> Self {
    let c = self.w.s[0];
    Self{w: self.w,
         b: Some(Tensor::shape(vec![1, c])),
         a: self.a}
  }
}

macro_rules! impl_linear_function {
  ($t:ty) => {
  impl<A> Function<$t> for Linear<$t, A>
    where A: Activation<$t> {
    fn forward<'b, 'g>(&self, x: &'b ADTensor<'g, $t>) -> ADTensor<'g, $t> {
      panic!();
    }
    fn eval<'b>(&self, x: &'b Tensor<$t>) -> Tensor<$t> {
      let y: Tensor<$t>;
      if let Some(ref b) = self.b {
        y = &x.mm(&self.w) + b
      }
      else {
        y = x.mm(&self.w)
      }
      if A::is_identity() {
        self.a.eval(&y)
      }
      else {
        y
      }   
    }
  }
  }
}

impl_linear_function!(f32);
impl_linear_function!(f64);

impl<T, A> Layer<T> for Linear<T, A>
  where Self: Function<T>,
        T: Zero,
        A: Activation<T> {
  fn build<'b>(&mut self, s: &'b Shape) -> Shape {
    debug_assert!(s.len() > 1, "Shape for Linear.build must have at least 2 dimensions!");
    self.w.s[1] = s[0];
    self.w = self.a.init(self.w.s.clone());
    if let Some(ref mut b) = self.b {
      *b = Tensor::shape(b.s.clone()).init_fn(|| T::zero())
    }
    s.broadcast_mm(&self.w.s)
  }
}





