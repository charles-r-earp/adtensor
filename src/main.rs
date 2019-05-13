use adtensor::Tensor;
use adtensor::autograd::Forward;
use adtensor::nn::{ElemInit, HeInit, Weight, Bias, Sigmoid, Seq};
use adtensor::optim::SGD;
use adtensor::seq;

fn main() {
  let mut net = seq![
    Weight::c(1).bias(),
    Sigmoid::default()
  ];
  let x = Tensor::<f32>::ones(vec![10, 2]);
  let y = net.forward(x); 
  println!("net: {:#?}", &net);
  println!("y: {:?}", &y);
}
