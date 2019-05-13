pub mod initializer;
pub use initializer::{Initializer, BasicInit, XavierInit};
pub mod activation;
pub use activation::{Sigmoid};
pub mod linear;
pub use linear::Weight;
pub mod sequential;
pub use sequential::Seq;
