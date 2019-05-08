

pub trait Optimizer<T> {
  fn step(&mut self, x: T) -> T;
}
