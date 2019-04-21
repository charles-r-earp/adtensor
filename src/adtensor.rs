use crate::tensor::{Tensor, Matmul};
use num_traits::{Zero, One};

#[derive(Default, Debug, Clone)]
pub struct ADTensor<T> {
  v: Tensor<T>,
  d: Tensor<T>
}

impl<T> ADTensor<T> {
  pub fn new(v: Tensor<T>) -> Self
    where T: One + Copy {
    let d = Tensor::with_elem(v.s().clone(), T::one()); 
    Self{v, d}
  }
  pub fn v<'a>(&'a self) -> &'a Tensor<T> {
    &self.v
  }
}

impl<T> From<Tensor<T>> for ADTensor<T>
  where T: Zero + Copy {
  fn from(v: Tensor<T>) -> Self {
    let d = Tensor::with_shape(v.s().clone());
    Self{v, d}
  }
} 

/*impl<'a, 'b, T> Matmul<&'b Tensor<T>> for &'a ADTensor<T> 
  where &'a Tensor<T>: Matmul<&'b Tensor<T>> {
  type Output = ADTensor<T>;
  fn mm(self, rhs: &'b Tensor<T>) -> Self::Output {
    if self.d.is_empty() {
      self.v().mm(rhs).into()
    }
    else {
      ADTensor::<T>{v: self.v().mm(rhs),  */
