//use adtensor::tensor::Tensor;
use criterion::*;
use matrixmultiply::sgemm;

type T = f32;
const K: usize = 28*28;
const M: usize = 64;
const N: usize = 10;

fn criterion_benchmark(c: &mut Criterion) {
  c.bench_function(&format!("sgemm [{} {}] x [{} {}]", M, K, K, N), |b| {
    let x = [0f32; M*K].to_vec();
    let y = [0f32; K*N].to_vec();
    let mut z = [0f32; M*N].to_vec();
    b.iter(|| {
      unsafe{ sgemm(
        M,
        K, 
        N,
        1.0,
        x.as_ptr(),
        1,
        1,
        y.as_ptr(),
        1,
        1,
        0.0,
        z.as_mut_ptr(),
        1,
        1
      ) };
      z[0]
    })  
  });
}





criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
