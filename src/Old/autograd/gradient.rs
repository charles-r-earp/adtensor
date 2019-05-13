

#[derive(Debug)]
pub struct Gradient<'p, T> {
  p: Vec<&'p Tensor<T>>,
  v: Vec<Tensor<T>>
}

impl<T> Gradient {
  
