pub trait Optimizer<T> {
  fn step(&mut self, x: T) -> T;
}

impl<F, T> Optimizer<T> for F 
  where F: FnMut(T)->T {
  fn step(&mut self, x: T) -> T {
    (self)(x)
  }
}
