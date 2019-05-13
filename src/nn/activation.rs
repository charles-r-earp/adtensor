use crate::{Tensor, TensorMap, autograd::Forward};
use std::marker::PhantomData;
use num_traits::Float;

#[derive(Debug, Default)] 
pub struct Sigmoid<T> {
  _p: PhantomData<T>
}

impl<'a, X, T> Forward<'a, X> for Sigmoid<T>
  where X: TensorMap<T, T, Output=Tensor<T>>,
        T: Float {
  type Y = Tensor<T>;
  fn forward(&'a mut self, x: X) -> Tensor<T> {
    x.map(|x| (T::one() + x.neg().exp()).recip())
  }
} 
