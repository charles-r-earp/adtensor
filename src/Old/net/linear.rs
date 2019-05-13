use crate::core::{Shape, Shaped, Tensor, Matmul};
use crate::net::{GetParams, Function};

pub struct Weight<T> {
  pub w: Tensor<T>,
  f: Box<Fn(&Shape)->T>
}

impl<T> Weight<T> {
  #[inline]
  pub fn shape_fn<S, F>(s: S, f: F) -> Self
    where Shape: From<S>,
          F: 'static + Fn(&Shape)->T {
    Self{w: Tensor::shape(s), f: Box::new(f)}
  }
  #[inline]
  pub fn c<F>(c: usize, f: F) -> Self
    where F: 'static + Fn(&Shape)->T {
    Self::shape_fn(vec![0, c], f)
  }
}

impl<'p, T> GetParams<'p, T> for Weight<T> {
  fn params(&'p self, p: &mut Vec<&'p Tensor<T>>) {
    p.push(&self.w)
  }
  fn params_mut(&'p mut self, p: &mut Vec<&'p mut Tensor<T>>) {
    p.push(&mut self.w)
  }
}

impl<'p, 'x, T, X> Function<'p, T, &'x X> for Weight<T>
  where T: 'p,
        X: Shaped,
        &'x X: Matmul<&'p Tensor<T>>, {
  type Output = <&'x X as Matmul<&'p Tensor<T>>>::Output;
  fn build(&'p mut self, x: &'x X, rebuild: bool) -> Self::Output {
    if rebuild || self.w.len() == 0 {
      self.w = Tensor::shape_fn(vec![x.s()[0], self.w.s()[0]], |s| (*self.f)(s));
    }
    self.eval(x)
  }
  fn eval(&'p self, x: &'x X) -> Self::Output {
    x.mm(&self.w)
  }
}

#[cfg(test)]
mod tests {
  use crate::core::{Tensor, Matmul, Shaped};
  use crate::net::{Function, Weight};
  #[test]
  fn test_weight() {
    let x = Tensor::<f32>::ones(vec![1, 2]);
    let mut fc = Weight::<f32>::c(4, |_| 0.1);
    (&mut fc).eval(&x);
    let y = fc.eval(&x);
  }
}
