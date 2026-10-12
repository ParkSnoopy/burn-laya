#[allow(unused_variables, clippy::all)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/model/laya.rs"));
}
pub use generated::Model;
use burn::tensor::{
    Bool,
    Device,
    Int,
    Tensor,
};
use anyhow::{
    Result,
    ensure,
};

use crate::input::Batch;

fn matrix<T: burn::tensor::Element>(rows: &[Vec<T>]) -> burn::tensor::TensorData {
    burn::tensor::TensorData::new(
        rows.iter().flatten().cloned().collect(),
        [rows.len(), rows[0].len()],
    )
}

pub type ModelOutput = (Vec<Vec<f32>>, Vec<Vec<f32>>);

pub fn infer(model: &Model, batch: &Batch, device: &Device) -> Result<ModelOutput> {
    let rows = batch.input_ids.len();
    ensure!(
        (1..=64).contains(&rows),
        "batch needs between 1 and 64 rows"
    );
    let length = batch.input_ids[0].len();
    ensure!(
        (1..=512).contains(&length),
        "sequence length must be between 1 and 512"
    );
    ensure!(
        batch.attention_mask.len() == rows
            && batch.marker_pos.len() == rows
            && batch.marker_mask.len() == rows
            && batch.qtype.len() == rows,
        "batch row counts must match"
    );
    let width = batch.marker_pos[0].len();
    ensure!(
        (2..=256).contains(&width),
        "marker width must be between 2 and 256"
    );
    for row in 0..rows {
        ensure!(
            batch.input_ids[row].len() == length
                && batch.input_ids[row]
                    .iter()
                    .all(|id| (0..50368).contains(id)),
            "invalid token row"
        );
        ensure!(
            batch.attention_mask[row].len() == length
                && batch.attention_mask[row][0] == 1
                && batch.attention_mask[row]
                    .iter()
                    .all(|mask| (0..=1).contains(mask)),
            "invalid attention mask"
        );
        ensure!(
            batch.marker_pos[row].len() == width
                && batch.marker_pos[row]
                    .iter()
                    .all(|position| (0..length as i64).contains(position)),
            "invalid marker positions"
        );
        ensure!(
            batch.marker_mask[row].len() == width
                && batch.marker_mask[row].iter().any(|mask| *mask),
            "invalid marker mask"
        );
        ensure!((0..=2).contains(&batch.qtype[row]), "invalid question type");
    }
    let (logits, actions) = model.forward(
        Tensor::<2, Int>::from_data(matrix(&batch.input_ids), (device, burn::tensor::DType::I64)),
        Tensor::<2, Int>::from_data(
            matrix(&batch.attention_mask),
            (device, burn::tensor::DType::I64),
        ),
        Tensor::<2, Int>::from_data(
            matrix(&batch.marker_pos),
            (device, burn::tensor::DType::I64),
        ),
        Tensor::<2, Bool>::from_data(matrix(&batch.marker_mask), device),
        Tensor::<1, Int>::from_data(
            burn::tensor::TensorData::new(batch.qtype.clone(), [rows]),
            (device, burn::tensor::DType::I64),
        ),
    );
    ensure!(
        logits.shape().as_slice() == [rows, width] && actions.shape().as_slice() == [rows, 2],
        "unexpected model output shape"
    );
    let collect = |tensor: Tensor<2>, width| -> Result<Vec<Vec<f32>>> {
        let values = tensor.into_data().try_to_vec::<f32>()?;
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "model returned nonfinite values"
        );
        Ok(values.chunks_exact(width).map(<[f32]>::to_vec).collect())
    };
    Ok((collect(logits, width)?, collect(actions, 2)?))
}
