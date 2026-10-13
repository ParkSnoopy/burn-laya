//! Exact query-tiled attention without an oversized dense score allocation.

use burn::tensor::{Tensor, ops::AttentionModuleOptions};

pub fn forward(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    bias: Option<Tensor<4>>,
    scale: Option<f64>,
) -> Tensor<4> {
    // This is a scratch-allocation budget, not an input, batch, or option limit.
    const SCORE_BYTES: usize = 32 * 1024 * 1024;
    let [batches, heads, queries, _] = query.dims();
    let keys = key.dims()[2];
    let options = AttentionModuleOptions {
        scale,
        softcap: None,
        is_causal: false,
    };
    let hidden = bias
        .as_ref()
        .map(|bias| bias.clone().max_dim(3).equal_elem(f32::NEG_INFINITY));
    let output = if batches
        .saturating_mul(heads)
        .saturating_mul(queries)
        .saturating_mul(keys)
        .saturating_mul(size_of::<f32>())
        <= SCORE_BYTES
    {
        burn::tensor::module::attention(query, key, value, None, bias, options)
    } else {
        let tile = (SCORE_BYTES / heads / keys / size_of::<f32>())
            .max(1)
            .min(queries);
        let mut rows = Vec::with_capacity(batches);
        for batch in 0..batches {
            let q = query.clone().narrow(0, batch, 1);
            let k = key.clone().narrow(0, batch, 1);
            let v = value.clone().narrow(0, batch, 1);
            let mut tiles = Vec::with_capacity(queries.div_ceil(tile));
            for start in (0..queries).step_by(tile) {
                let length = tile.min(queries - start);
                let tile_bias = bias.as_ref().map(|bias| {
                    let [bias_batches, _, bias_queries, _] = bias.dims();
                    let bias = if bias_batches == 1 {
                        bias.clone()
                    } else {
                        bias.clone().narrow(0, batch, 1)
                    };
                    if bias_queries == 1 {
                        bias
                    } else {
                        bias.narrow(2, start, length)
                    }
                });
                tiles.push(burn::tensor::module::attention(
                    q.clone().narrow(2, start, length),
                    k.clone(),
                    v.clone(),
                    None,
                    tile_bias,
                    options,
                ));
            }
            rows.push(Tensor::cat(tiles, 2));
        }
        Tensor::cat(rows, 0)
    };
    if let Some(hidden) = hidden {
        let dims = output.dims();
        output.mask_fill(hidden.expand(dims), 0.0)
    } else {
        output
    }
}
