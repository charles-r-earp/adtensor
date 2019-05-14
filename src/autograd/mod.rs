pub mod parameter;
pub use parameter::{Initializer, Optimizer, Parameter};
pub mod graph;
pub use graph::{Forward, Backward, Evaluate};

