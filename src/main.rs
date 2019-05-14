use adtensor::{Shape, Tensor};
use adtensor::autograd::{Parameter, Forward};
use adtensor::nn::init::{Zeros, Xavier};
use adtensor::nn::{Linear};

fn main() {
  let mut fc = Linear::<f32, ()>::new(
    Parameter::shape_init(vec![1], Xavier::default()),
    Some(Parameter::shape_init(vec![], Zeros::default()))
  );
  println!("fc: {:#?}", &fc);
  let x = Tensor::<f32>::ones(vec![10, 2]);
  let y = fc.forward(x);
  println!("fc: {:#?}", &fc);
  println!("y: {:?}", &y);
}
