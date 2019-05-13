use adtensor::Tensor;
use adtensor::autograd::Forward;
use adtensor::nn::{BasicInit, Weight, Sigmoid, Seq};
use adtensor::optim::SGD;
use adtensor::seq;

fn main() {
  let mut net = seq![
    Weight::new(
      vec![1], 
      BasicInit::new(), 
      SGD::new(0.01)
    ),
    Sigmoid::default()
  ];
  let x = Tensor::<f32>::ones(vec![1, 2]);
  let y = net.forward(x); 
  println!("y: {:?}", &y);
}
