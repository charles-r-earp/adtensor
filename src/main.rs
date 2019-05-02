use adtensor::shape::Shape;
use adtensor::tensor::Tensor;
use adtensor::graph::{Graph, Param, Optimizer};
use std::iter::IntoIterator;

fn main() { 
  let g = Graph::new();
  let x = g.var(Tensor::from(vec![3f32]));
  let mut w = Param::from(Tensor::from(vec![4f32]));
  x.register(vec![&mut w]);
  let y = &x * &w;
}
