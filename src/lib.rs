//! Typed local Laya inference through Burn.

pub mod inference;
pub use inference::{Agent, Prediction, Usage, input, model, output};
pub use input::{Question, Request};
pub use output::{Action, Answer, AnswerValue};
