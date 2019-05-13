use adtensor::core::{Tensor, Matmul};
use adtensor::net::{GetParams, Function, Weight};
use adtensor::autograd::{Graph, Loss};
use adtensor::optim::{Optimizer, SGD};

fn main() {
  let mut net = Weight::c(1, |_| 0.1f32);
  net.build(&Tensor::zeros(vec![1, 2]), true);
  let x = vec![vec![Tensor::<f32>::ones(vec![1, 2]); 3]];
  let mut opt = SGD::new(0.01);
  x.into_iter().for_each(|x| {
    let loss =
    {
      let mut p = Vec::new();
      net.params(&mut p);
      p.sort_unstable_by_key(|p| p.param_key());
      x.into_iter().map(|x| { 
        let mut g = Graph::params(&p);
        let x = g.input(x);
        let y = net.eval(&x);
        Loss::from(y)
      }).fold(Loss::new(), |a, l| &a + &l)
    };
    println!("loss: {:?}", &loss);
    let mut p = Vec::new();
    net.params_mut(&mut p);
    p.sort_unstable_by_key(|p| p.param_key());  
    opt.step(p, loss);     
  }); 
}
