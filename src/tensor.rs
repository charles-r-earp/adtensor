use std::marker::PhantomData;
pub use typenum::{TArr, ATerm, tarr};
use generic_array::{GenericArray, ArrayLength};

pub struct Grad<P> {
  p: PhantomData<P>
} 

pub trait TensorShape<T> {
  type Storage;
}
 
pub type TensorStorage<T, S> = <S as TensorShape<T>>::Storage; 

pub struct Tensor<T, S>
  where S: TensorShape<T> {
  s: TensorStorage<T, S>
}

impl<T, N> TensorShape<T> for tarr![N]
  where N: ArrayLength<T> {
  type Storage = GenericArray<T, N>;
}
  
impl<T, N, P> TensorShape<T> for tarr![N, Grad<P>]
  where tarr![N]: TensorShape<T>,
        tarr![P, N]: TensorShape<T> {
  type Storage = (Tensor<T, tarr![N]>, Tensor<T, tarr![P, N]>);
}
  

