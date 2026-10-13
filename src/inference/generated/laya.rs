// Generated from the verified English Laya ONNX graph by burn-onnx 0.22.
extern crate alloc;
use burn::prelude::*;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Submodule1 {
    constant1: burn::module::Param<Tensor<2>>,
    layernormalization1: LayerNorm,
    constant90: burn::module::Param<Tensor<1, Bool>>,
    constant87: burn::module::Param<Tensor<1>>,
    constant86: burn::module::Param<Tensor<1>>,
    linear1: Linear,
    linear2: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant1: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([50368, 1024], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [50368, 1024].into(),
        );
        let layernormalization1 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let constant90: burn::module::Param<Tensor<1, Bool>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Bool,
            >::from_data(
                burn::tensor::TensorData::from([true]),
                (device, burn::tensor::DType::Bool(burn::tensor::BoolStore::Native)),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant87: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([32], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [32].into(),
        );
        let constant86: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([32], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [32].into(),
        );
        let linear1 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear2 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            constant1,
            layernormalization1,
            constant90,
            constant87,
            constant86,
            linear1,
            linear2,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        attention_mask: Tensor<2, Int>,
        qtype: Tensor<1, Int>,
        input_ids: Tensor<2, Int>,
    ) -> (
        Tensor<3>,
        [i64; 5],
        Tensor<3>,
        Tensor<3>,
        Tensor<4>,
        [i64; 3],
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<2, Bool>,
        f32,
        [i64; 1],
        [i64; 1],
        [i64; 1],
    ) {
        let shape1_out1: [i64; 1] = {
            let axes = &attention_mask.clone().dims()[1..2];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let squeeze1_out1 = shape1_out1[0] as i64;
        let shape2_out1: [i64; 1] = {
            let axes = &qtype.dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let squeeze2_out1 = shape2_out1[0] as i64;
        let range1_out1 = {
            let start = 0i64;
            let delta = 1i64;
            assert!(delta != 0);
            let n = (((squeeze1_out1 as i128 - start as i128) as f64) / delta as f64)
                .ceil()
                .max(0.0) as i64;
            Tensor::arange(0..n, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(delta)
                .add_scalar(start)
        };
        let unsqueeze1_out1: Tensor<2, Int> = range1_out1
            .clone()
            .unsqueeze_dims::<2>(&[0]);
        let constant1_out1 = self.constant1.val();
        let gather1_out1 = {
            let axis_size = constant1_out1.dims()[0] as i64;
            let negative = input_ids.clone().lower_elem(0i64);
            let corrected = input_ids.clone() + axis_size;
            let indices = input_ids.mask_where(negative, corrected);
            constant1_out1.take::<2, 3>(0, indices)
        };
        let layernormalization1_out1 = {
            let dtype = gather1_out1.dtype();
            self.layernormalization1
                .forward(gather1_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let cast1_out1 = attention_mask.bool();
        let range2_out1 = {
            let start = 0i64;
            let delta = 1i64;
            assert!(delta != 0);
            let n = (((squeeze2_out1 as i128 - start as i128) as f64) / delta as f64)
                .ceil()
                .max(0.0) as i64;
            Tensor::arange(0..n, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(delta)
                .add_scalar(start)
        };
        let unsqueeze2_out1: Tensor<3, Int> = range2_out1.unsqueeze_dims::<3>(&[1, 2]);
        let unsqueeze3_out1: Tensor<4, Int> = unsqueeze2_out1.unsqueeze_dims::<4>(&[3]);
        let unsqueeze4_out1: Tensor<3, Int> = range1_out1.unsqueeze_dims::<3>(&[0, 1]);
        let unsqueeze5_out1: Tensor<4, Int> = unsqueeze4_out1
            .clone()
            .unsqueeze_dims::<4>(&[3]);
        let unsqueeze6_out1: Tensor<4, Int> = unsqueeze4_out1.unsqueeze_dims::<4>(&[2]);
        let constant88_out1 = 0i64;
        let greaterorequal1_out1 = unsqueeze5_out1
            .clone()
            .greater_equal_elem(constant88_out1);
        let constant90_out1 = self.constant90.val();
        let and1_out1 = (constant90_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .bool_and(greaterorequal1_out1.clone());
        let max1_out1 = unsqueeze3_out1.clone().max_pair(unsqueeze6_out1.clone());
        let shape3_out1: [i64; 4] = {
            let axes = &max1_out1.dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let expand1_out1 = {
            let onnx_shape: [i64; 4usize] = shape3_out1;
            let input_dims = unsqueeze3_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze3_out1.expand(shape)
        };
        let unsqueeze7_out1: Tensor<5, Int> = expand1_out1.unsqueeze_dims::<5>(&[-1]);
        let expand2_out1 = {
            let onnx_shape: [i64; 4usize] = shape3_out1;
            let input_dims = unsqueeze6_out1.clone().dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze6_out1.clone().expand(shape)
        };
        let unsqueeze8_out1: Tensor<5, Int> = expand2_out1.unsqueeze_dims::<5>(&[-1]);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [unsqueeze7_out1, unsqueeze8_out1].into(),
            4,
        );
        let gathernd1_out1 = {
            let data_dims = cast1_out1.clone().dims();
            let indices_i64 = concat1_out1.cast(burn::tensor::DType::I64);
            let idx_dims = indices_i64.dims();
            let k = idx_dims[5 - 1];
            let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
            for i in 0..k {
                dim_sizes.push(data_dims[0 + i] as i64);
            }
            let mut bcast_shape = [1usize; 5];
            bcast_shape[5 - 1] = k;
            let dims_tensor = Tensor::<
                1,
                Int,
            >::from_data(
                    burn::tensor::TensorData::from(dim_sizes.as_slice()),
                    (&self.device, burn::tensor::DType::I64),
                )
                .reshape(bcast_shape);
            let negative = indices_i64.clone().lower_elem(0i64);
            let corrected = indices_i64.clone() + dims_tensor;
            let indices_norm = indices_i64.mask_where(negative, corrected);
            let aug_indices = indices_norm;
            cast1_out1
                .clone()
                .int()
                .cast(burn::tensor::DType::I64)
                .gather_nd(aug_indices)
                .bool()
        };
        let and2_out1 = and1_out1.bool_and(gathernd1_out1.clone());
        let constant91_out1 = 0f32;
        let constant226_out1 = -340282350000000000000000000000000000000f32;
        let where1_out1 = {
            let cond = and2_out1;
            Tensor::<
                1,
            >::from_data(
                    burn::tensor::TensorData::from([constant226_out1 as f64]),
                    (&self.device, burn::tensor::DType::F32),
                )
                .reshape([1, 1, 1, 1])
                .expand(cond.dims())
                .mask_fill(cond, constant91_out1)
        };
        let sub1_out1 = unsqueeze5_out1.sub(unsqueeze6_out1);
        let abs1_out1 = sub1_out1.abs();
        let constant227_out1 = 64i64;
        let lessorequal1_out1 = abs1_out1.lower_equal_elem(constant227_out1);
        let and3_out1 = (constant90_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .bool_and(lessorequal1_out1);
        let and4_out1 = and3_out1.bool_and(greaterorequal1_out1);
        let and5_out1 = (constant90_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .bool_and(and4_out1);
        let and6_out1 = and5_out1.bool_and(gathernd1_out1);
        let where2_out1 = {
            let cond = and6_out1;
            Tensor::<
                1,
            >::from_data(
                    burn::tensor::TensorData::from([constant226_out1 as f64]),
                    (&self.device, burn::tensor::DType::F32),
                )
                .reshape([1, 1, 1, 1])
                .expand(cond.dims())
                .mask_fill(cond, constant91_out1)
        };
        let unsqueeze9_out1: Tensor<3, Int> = unsqueeze1_out1.unsqueeze_dims::<3>(&[2]);
        let cast2_out1 = unsqueeze9_out1.float().cast(burn::tensor::DType::F32);
        let constant87_out1 = self.constant87.val();
        let mul1_out1 = cast2_out1
            .clone()
            .mul((constant87_out1).unsqueeze_dims(&[0isize, 1isize]));
        let concat2_out1 = burn::tensor::Tensor::cat(
            [mul1_out1.clone(), mul1_out1].into(),
            2,
        );
        let cos1_out1 = concat2_out1.clone().cos();
        let sin1_out1 = concat2_out1.sin();
        let constant86_out1 = self.constant86.val();
        let mul2_out1 = cast2_out1
            .mul((constant86_out1).unsqueeze_dims(&[0isize, 1isize]));
        let concat3_out1 = burn::tensor::Tensor::cat(
            [mul2_out1.clone(), mul2_out1].into(),
            2,
        );
        let cos2_out1 = concat3_out1.clone().cos();
        let sin2_out1 = concat3_out1.sin();
        let linear1_out1 = self.linear1.forward(layernormalization1_out1.clone());
        let constant95_out1: [i64; 1] = [3i64];
        let constant225_out1: [i64; 1] = [-1i64];
        let constant228_out1: [i64; 1] = [64i64];
        let concat4_out1: [i64; 5usize] = [
            &shape2_out1[..],
            &shape1_out1[..],
            &constant95_out1[..],
            &constant225_out1[..],
            &constant228_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape1_out1 = linear1_out1.reshape(concat4_out1);
        let slice1_out1 = reshape1_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze3_out1 = slice1_out1.squeeze_dims::<4>(&[-3]);
        let slice2_out1 = reshape1_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze4_out1 = slice2_out1.squeeze_dims::<4>(&[-3]);
        let slice3_out1 = reshape1_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze5_out1 = slice3_out1.squeeze_dims::<4>(&[-3]);
        let transpose1_out1 = squeeze3_out1.permute([0, 2, 1, 3]);
        let transpose2_out1 = squeeze4_out1.permute([0, 2, 1, 3]);
        let transpose3_out1 = squeeze5_out1.permute([0, 2, 1, 3]);
        let unsqueeze10_out1: Tensor<4> = cos2_out1.unsqueeze_dims::<4>(&[1]);
        let unsqueeze11_out1: Tensor<4> = sin2_out1.unsqueeze_dims::<4>(&[1]);
        let mul3_out1 = transpose1_out1.clone().mul(unsqueeze10_out1.clone());
        let slice4_out1 = transpose1_out1.clone().slice(s![.., .., .., 0..32]);
        let slice5_out1 = transpose1_out1.slice(s![.., .., .., 32..]);
        let neg1_out1 = slice5_out1.neg();
        let concat5_out1 = burn::tensor::Tensor::cat([neg1_out1, slice4_out1].into(), 3);
        let mul4_out1 = concat5_out1.mul(unsqueeze11_out1.clone());
        let add1_out1 = mul3_out1.add(mul4_out1);
        let mul5_out1 = transpose2_out1.clone().mul(unsqueeze10_out1.clone());
        let slice6_out1 = transpose2_out1.clone().slice(s![.., .., .., 0..32]);
        let slice7_out1 = transpose2_out1.slice(s![.., .., .., 32..]);
        let neg2_out1 = slice7_out1.neg();
        let concat6_out1 = burn::tensor::Tensor::cat([neg2_out1, slice6_out1].into(), 3);
        let mul6_out1 = concat6_out1.mul(unsqueeze11_out1.clone());
        let add2_out1 = mul5_out1.add(mul6_out1);
        let matmul3_out1 = crate::inference::attention::forward(
            add1_out1,
            add2_out1,
            transpose3_out1,
            Some(where1_out1.clone()),
            Some(0.125f64),
        );
        let transpose5_out1 = matmul3_out1.permute([0, 2, 1, 3]);
        let concat7_out1: [i64; 3usize] = [
            &shape2_out1[..],
            &shape1_out1[..],
            &constant225_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape2_out1 = transpose5_out1.reshape(concat7_out1);
        let linear2_out1 = self.linear2.forward(reshape2_out1);
        let add4_out1 = layernormalization1_out1.add(linear2_out1);
        (
            add4_out1,
            concat4_out1,
            cos1_out1,
            sin1_out1,
            where2_out1,
            concat7_out1,
            unsqueeze10_out1,
            unsqueeze11_out1,
            where1_out1,
            cast1_out1,
            constant91_out1,
            shape1_out1,
            shape2_out1,
            constant228_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule2 {
    layernormalization2: LayerNorm,
    linear3: Linear,
    linear4: Linear,
    layernormalization3: LayerNorm,
    linear5: Linear,
    linear6: Linear,
    layernormalization4: LayerNorm,
    linear7: Linear,
    linear8: Linear,
    layernormalization5: LayerNorm,
    linear9: Linear,
    linear10: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule2 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization2 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear3 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear4 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization3 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear5 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear6 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization4 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear7 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear8 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization5 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear9 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear10 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization2,
            linear3,
            linear4,
            layernormalization3,
            linear5,
            linear6,
            layernormalization4,
            linear7,
            linear8,
            layernormalization5,
            linear9,
            linear10,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add4_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        cos1_out1: Tensor<3>,
        sin1_out1: Tensor<3>,
        where2_out1: Tensor<4>,
        concat7_out1: [i64; 3],
    ) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
        let layernormalization2_out1 = {
            let dtype = add4_out1.clone().dtype();
            self.layernormalization2
                .forward(add4_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear3_out1 = self.linear3.forward(layernormalization2_out1);
        let split_tensors = linear3_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split1_out1, split1_out2] = split_tensors.try_into().unwrap();
        let gelu1_out1 = burn::tensor::activation::gelu(split1_out1);
        let mul8_out1 = gelu1_out1.mul(split1_out2);
        let linear4_out1 = self.linear4.forward(mul8_out1);
        let add5_out1 = add4_out1.add(linear4_out1);
        let layernormalization3_out1 = {
            let dtype = add5_out1.clone().dtype();
            self.layernormalization3
                .forward(add5_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear5_out1 = self.linear5.forward(layernormalization3_out1);
        let reshape3_out1 = linear5_out1.reshape(concat4_out1);
        let slice8_out1 = reshape3_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze6_out1 = slice8_out1.squeeze_dims::<4>(&[-3]);
        let slice9_out1 = reshape3_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze7_out1 = slice9_out1.squeeze_dims::<4>(&[-3]);
        let slice10_out1 = reshape3_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze8_out1 = slice10_out1.squeeze_dims::<4>(&[-3]);
        let transpose6_out1 = squeeze6_out1.permute([0, 2, 1, 3]);
        let transpose7_out1 = squeeze7_out1.permute([0, 2, 1, 3]);
        let transpose8_out1 = squeeze8_out1.permute([0, 2, 1, 3]);
        let unsqueeze12_out1: Tensor<4> = cos1_out1.unsqueeze_dims::<4>(&[1]);
        let unsqueeze13_out1: Tensor<4> = sin1_out1.unsqueeze_dims::<4>(&[1]);
        let mul9_out1 = transpose6_out1.clone().mul(unsqueeze12_out1.clone());
        let slice11_out1 = transpose6_out1.clone().slice(s![.., .., .., 0..32]);
        let slice12_out1 = transpose6_out1.slice(s![.., .., .., 32..]);
        let neg3_out1 = slice12_out1.neg();
        let concat8_out1 = burn::tensor::Tensor::cat(
            [neg3_out1, slice11_out1].into(),
            3,
        );
        let mul10_out1 = concat8_out1.mul(unsqueeze13_out1.clone());
        let add6_out1 = mul9_out1.add(mul10_out1);
        let mul11_out1 = transpose7_out1.clone().mul(unsqueeze12_out1.clone());
        let slice13_out1 = transpose7_out1.clone().slice(s![.., .., .., 0..32]);
        let slice14_out1 = transpose7_out1.slice(s![.., .., .., 32..]);
        let neg4_out1 = slice14_out1.neg();
        let concat9_out1 = burn::tensor::Tensor::cat(
            [neg4_out1, slice13_out1].into(),
            3,
        );
        let mul12_out1 = concat9_out1.mul(unsqueeze13_out1.clone());
        let add7_out1 = mul11_out1.add(mul12_out1);
        let matmul9_out1 = crate::inference::attention::forward(
            add6_out1,
            add7_out1,
            transpose8_out1,
            Some(where2_out1.clone()),
            Some(0.125f64),
        );
        let transpose10_out1 = matmul9_out1.permute([0, 2, 1, 3]);
        let reshape4_out1 = transpose10_out1.reshape(concat7_out1);
        let linear6_out1 = self.linear6.forward(reshape4_out1);
        let add9_out1 = add5_out1.add(linear6_out1);
        let layernormalization4_out1 = {
            let dtype = add9_out1.clone().dtype();
            self.layernormalization4
                .forward(add9_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear7_out1 = self.linear7.forward(layernormalization4_out1);
        let split_tensors = linear7_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split2_out1, split2_out2] = split_tensors.try_into().unwrap();
        let gelu2_out1 = burn::tensor::activation::gelu(split2_out1);
        let mul14_out1 = gelu2_out1.mul(split2_out2);
        let linear8_out1 = self.linear8.forward(mul14_out1);
        let add10_out1 = add9_out1.add(linear8_out1);
        let layernormalization5_out1 = {
            let dtype = add10_out1.clone().dtype();
            self.layernormalization5
                .forward(add10_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear9_out1 = self.linear9.forward(layernormalization5_out1);
        let reshape5_out1 = linear9_out1.reshape(concat4_out1);
        let slice15_out1 = reshape5_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze9_out1 = slice15_out1.squeeze_dims::<4>(&[-3]);
        let slice16_out1 = reshape5_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze10_out1 = slice16_out1.squeeze_dims::<4>(&[-3]);
        let slice17_out1 = reshape5_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze11_out1 = slice17_out1.squeeze_dims::<4>(&[-3]);
        let transpose11_out1 = squeeze9_out1.permute([0, 2, 1, 3]);
        let transpose12_out1 = squeeze10_out1.permute([0, 2, 1, 3]);
        let transpose13_out1 = squeeze11_out1.permute([0, 2, 1, 3]);
        let mul15_out1 = transpose11_out1.clone().mul(unsqueeze12_out1.clone());
        let slice18_out1 = transpose11_out1.clone().slice(s![.., .., .., 0..32]);
        let slice19_out1 = transpose11_out1.slice(s![.., .., .., 32..]);
        let neg5_out1 = slice19_out1.neg();
        let concat10_out1 = burn::tensor::Tensor::cat(
            [neg5_out1, slice18_out1].into(),
            3,
        );
        let mul16_out1 = concat10_out1.mul(unsqueeze13_out1.clone());
        let add11_out1 = mul15_out1.add(mul16_out1);
        let mul17_out1 = transpose12_out1.clone().mul(unsqueeze12_out1.clone());
        let slice20_out1 = transpose12_out1.clone().slice(s![.., .., .., 0..32]);
        let slice21_out1 = transpose12_out1.slice(s![.., .., .., 32..]);
        let neg6_out1 = slice21_out1.neg();
        let concat11_out1 = burn::tensor::Tensor::cat(
            [neg6_out1, slice20_out1].into(),
            3,
        );
        let mul18_out1 = concat11_out1.mul(unsqueeze13_out1.clone());
        let add12_out1 = mul17_out1.add(mul18_out1);
        let matmul15_out1 = crate::inference::attention::forward(
            add11_out1,
            add12_out1,
            transpose13_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose15_out1 = matmul15_out1.permute([0, 2, 1, 3]);
        let reshape6_out1 = transpose15_out1.reshape(concat7_out1);
        let linear10_out1 = self.linear10.forward(reshape6_out1);
        let add14_out1 = add10_out1.add(linear10_out1);
        (add14_out1, unsqueeze12_out1, unsqueeze13_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule3 {
    layernormalization6: LayerNorm,
    linear11: Linear,
    linear12: Linear,
    layernormalization7: LayerNorm,
    linear13: Linear,
    linear14: Linear,
    layernormalization8: LayerNorm,
    linear15: Linear,
    linear16: Linear,
    layernormalization9: LayerNorm,
    linear17: Linear,
    linear18: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule3 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization6 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear11 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear12 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization7 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear13 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear14 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization8 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear15 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear16 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization9 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear17 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear18 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization6,
            linear11,
            linear12,
            layernormalization7,
            linear13,
            linear14,
            layernormalization8,
            linear15,
            linear16,
            layernormalization9,
            linear17,
            linear18,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add14_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
    ) -> Tensor<3> {
        let layernormalization6_out1 = {
            let dtype = add14_out1.clone().dtype();
            self.layernormalization6
                .forward(add14_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear11_out1 = self.linear11.forward(layernormalization6_out1);
        let split_tensors = linear11_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split3_out1, split3_out2] = split_tensors.try_into().unwrap();
        let gelu3_out1 = burn::tensor::activation::gelu(split3_out1);
        let mul20_out1 = gelu3_out1.mul(split3_out2);
        let linear12_out1 = self.linear12.forward(mul20_out1);
        let add15_out1 = add14_out1.add(linear12_out1);
        let layernormalization7_out1 = {
            let dtype = add15_out1.clone().dtype();
            self.layernormalization7
                .forward(add15_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear13_out1 = self.linear13.forward(layernormalization7_out1);
        let reshape7_out1 = linear13_out1.reshape(concat4_out1);
        let slice22_out1 = reshape7_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze12_out1 = slice22_out1.squeeze_dims::<4>(&[-3]);
        let slice23_out1 = reshape7_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze13_out1 = slice23_out1.squeeze_dims::<4>(&[-3]);
        let slice24_out1 = reshape7_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze14_out1 = slice24_out1.squeeze_dims::<4>(&[-3]);
        let transpose16_out1 = squeeze12_out1.permute([0, 2, 1, 3]);
        let transpose17_out1 = squeeze13_out1.permute([0, 2, 1, 3]);
        let transpose18_out1 = squeeze14_out1.permute([0, 2, 1, 3]);
        let mul21_out1 = transpose16_out1.clone().mul(unsqueeze10_out1.clone());
        let slice25_out1 = transpose16_out1.clone().slice(s![.., .., .., 0..32]);
        let slice26_out1 = transpose16_out1.slice(s![.., .., .., 32..]);
        let neg7_out1 = slice26_out1.neg();
        let concat12_out1 = burn::tensor::Tensor::cat(
            [neg7_out1, slice25_out1].into(),
            3,
        );
        let mul22_out1 = concat12_out1.mul(unsqueeze11_out1.clone());
        let add16_out1 = mul21_out1.add(mul22_out1);
        let mul23_out1 = transpose17_out1.clone().mul(unsqueeze10_out1);
        let slice27_out1 = transpose17_out1.clone().slice(s![.., .., .., 0..32]);
        let slice28_out1 = transpose17_out1.slice(s![.., .., .., 32..]);
        let neg8_out1 = slice28_out1.neg();
        let concat13_out1 = burn::tensor::Tensor::cat(
            [neg8_out1, slice27_out1].into(),
            3,
        );
        let mul24_out1 = concat13_out1.mul(unsqueeze11_out1);
        let add17_out1 = mul23_out1.add(mul24_out1);
        let matmul21_out1 = crate::inference::attention::forward(
            add16_out1,
            add17_out1,
            transpose18_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose20_out1 = matmul21_out1.permute([0, 2, 1, 3]);
        let reshape8_out1 = transpose20_out1.reshape(concat7_out1);
        let linear14_out1 = self.linear14.forward(reshape8_out1);
        let add19_out1 = add15_out1.add(linear14_out1);
        let layernormalization8_out1 = {
            let dtype = add19_out1.clone().dtype();
            self.layernormalization8
                .forward(add19_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear15_out1 = self.linear15.forward(layernormalization8_out1);
        let split_tensors = linear15_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split4_out1, split4_out2] = split_tensors.try_into().unwrap();
        let gelu4_out1 = burn::tensor::activation::gelu(split4_out1);
        let mul26_out1 = gelu4_out1.mul(split4_out2);
        let linear16_out1 = self.linear16.forward(mul26_out1);
        let add20_out1 = add19_out1.add(linear16_out1);
        let layernormalization9_out1 = {
            let dtype = add20_out1.clone().dtype();
            self.layernormalization9
                .forward(add20_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear17_out1 = self.linear17.forward(layernormalization9_out1);
        let reshape9_out1 = linear17_out1.reshape(concat4_out1);
        let slice29_out1 = reshape9_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze15_out1 = slice29_out1.squeeze_dims::<4>(&[-3]);
        let slice30_out1 = reshape9_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze16_out1 = slice30_out1.squeeze_dims::<4>(&[-3]);
        let slice31_out1 = reshape9_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze17_out1 = slice31_out1.squeeze_dims::<4>(&[-3]);
        let transpose21_out1 = squeeze15_out1.permute([0, 2, 1, 3]);
        let transpose22_out1 = squeeze16_out1.permute([0, 2, 1, 3]);
        let transpose23_out1 = squeeze17_out1.permute([0, 2, 1, 3]);
        let mul27_out1 = transpose21_out1.clone().mul(unsqueeze12_out1.clone());
        let slice32_out1 = transpose21_out1.clone().slice(s![.., .., .., 0..32]);
        let slice33_out1 = transpose21_out1.slice(s![.., .., .., 32..]);
        let neg9_out1 = slice33_out1.neg();
        let concat14_out1 = burn::tensor::Tensor::cat(
            [neg9_out1, slice32_out1].into(),
            3,
        );
        let mul28_out1 = concat14_out1.mul(unsqueeze13_out1.clone());
        let add21_out1 = mul27_out1.add(mul28_out1);
        let mul29_out1 = transpose22_out1.clone().mul(unsqueeze12_out1);
        let slice34_out1 = transpose22_out1.clone().slice(s![.., .., .., 0..32]);
        let slice35_out1 = transpose22_out1.slice(s![.., .., .., 32..]);
        let neg10_out1 = slice35_out1.neg();
        let concat15_out1 = burn::tensor::Tensor::cat(
            [neg10_out1, slice34_out1].into(),
            3,
        );
        let mul30_out1 = concat15_out1.mul(unsqueeze13_out1);
        let add22_out1 = mul29_out1.add(mul30_out1);
        let matmul27_out1 = crate::inference::attention::forward(
            add21_out1,
            add22_out1,
            transpose23_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose25_out1 = matmul27_out1.permute([0, 2, 1, 3]);
        let reshape10_out1 = transpose25_out1.reshape(concat7_out1);
        let linear18_out1 = self.linear18.forward(reshape10_out1);
        let add24_out1 = add20_out1.add(linear18_out1);
        add24_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4 {
    layernormalization10: LayerNorm,
    linear19: Linear,
    linear20: Linear,
    layernormalization11: LayerNorm,
    linear21: Linear,
    linear22: Linear,
    layernormalization12: LayerNorm,
    linear23: Linear,
    linear24: Linear,
    layernormalization13: LayerNorm,
    linear25: Linear,
    linear26: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule4 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization10 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear19 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear20 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization11 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear21 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear22 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization12 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear23 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear24 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization13 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear25 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear26 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization10,
            linear19,
            linear20,
            layernormalization11,
            linear21,
            linear22,
            layernormalization12,
            linear23,
            linear24,
            layernormalization13,
            linear25,
            linear26,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add24_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
    ) -> Tensor<3> {
        let layernormalization10_out1 = {
            let dtype = add24_out1.clone().dtype();
            self.layernormalization10
                .forward(add24_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear19_out1 = self.linear19.forward(layernormalization10_out1);
        let split_tensors = linear19_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split5_out1, split5_out2] = split_tensors.try_into().unwrap();
        let gelu5_out1 = burn::tensor::activation::gelu(split5_out1);
        let mul32_out1 = gelu5_out1.mul(split5_out2);
        let linear20_out1 = self.linear20.forward(mul32_out1);
        let add25_out1 = add24_out1.add(linear20_out1);
        let layernormalization11_out1 = {
            let dtype = add25_out1.clone().dtype();
            self.layernormalization11
                .forward(add25_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear21_out1 = self.linear21.forward(layernormalization11_out1);
        let reshape11_out1 = linear21_out1.reshape(concat4_out1);
        let slice36_out1 = reshape11_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze18_out1 = slice36_out1.squeeze_dims::<4>(&[-3]);
        let slice37_out1 = reshape11_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze19_out1 = slice37_out1.squeeze_dims::<4>(&[-3]);
        let slice38_out1 = reshape11_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze20_out1 = slice38_out1.squeeze_dims::<4>(&[-3]);
        let transpose26_out1 = squeeze18_out1.permute([0, 2, 1, 3]);
        let transpose27_out1 = squeeze19_out1.permute([0, 2, 1, 3]);
        let transpose28_out1 = squeeze20_out1.permute([0, 2, 1, 3]);
        let mul33_out1 = transpose26_out1.clone().mul(unsqueeze12_out1.clone());
        let slice39_out1 = transpose26_out1.clone().slice(s![.., .., .., 0..32]);
        let slice40_out1 = transpose26_out1.slice(s![.., .., .., 32..]);
        let neg11_out1 = slice40_out1.neg();
        let concat16_out1 = burn::tensor::Tensor::cat(
            [neg11_out1, slice39_out1].into(),
            3,
        );
        let mul34_out1 = concat16_out1.mul(unsqueeze13_out1.clone());
        let add26_out1 = mul33_out1.add(mul34_out1);
        let mul35_out1 = transpose27_out1.clone().mul(unsqueeze12_out1);
        let slice41_out1 = transpose27_out1.clone().slice(s![.., .., .., 0..32]);
        let slice42_out1 = transpose27_out1.slice(s![.., .., .., 32..]);
        let neg12_out1 = slice42_out1.neg();
        let concat17_out1 = burn::tensor::Tensor::cat(
            [neg12_out1, slice41_out1].into(),
            3,
        );
        let mul36_out1 = concat17_out1.mul(unsqueeze13_out1);
        let add27_out1 = mul35_out1.add(mul36_out1);
        let matmul33_out1 = crate::inference::attention::forward(
            add26_out1,
            add27_out1,
            transpose28_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose30_out1 = matmul33_out1.permute([0, 2, 1, 3]);
        let reshape12_out1 = transpose30_out1.reshape(concat7_out1);
        let linear22_out1 = self.linear22.forward(reshape12_out1);
        let add29_out1 = add25_out1.add(linear22_out1);
        let layernormalization12_out1 = {
            let dtype = add29_out1.clone().dtype();
            self.layernormalization12
                .forward(add29_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear23_out1 = self.linear23.forward(layernormalization12_out1);
        let split_tensors = linear23_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split6_out1, split6_out2] = split_tensors.try_into().unwrap();
        let gelu6_out1 = burn::tensor::activation::gelu(split6_out1);
        let mul38_out1 = gelu6_out1.mul(split6_out2);
        let linear24_out1 = self.linear24.forward(mul38_out1);
        let add30_out1 = add29_out1.add(linear24_out1);
        let layernormalization13_out1 = {
            let dtype = add30_out1.clone().dtype();
            self.layernormalization13
                .forward(add30_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear25_out1 = self.linear25.forward(layernormalization13_out1);
        let reshape13_out1 = linear25_out1.reshape(concat4_out1);
        let slice43_out1 = reshape13_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze21_out1 = slice43_out1.squeeze_dims::<4>(&[-3]);
        let slice44_out1 = reshape13_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze22_out1 = slice44_out1.squeeze_dims::<4>(&[-3]);
        let slice45_out1 = reshape13_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze23_out1 = slice45_out1.squeeze_dims::<4>(&[-3]);
        let transpose31_out1 = squeeze21_out1.permute([0, 2, 1, 3]);
        let transpose32_out1 = squeeze22_out1.permute([0, 2, 1, 3]);
        let transpose33_out1 = squeeze23_out1.permute([0, 2, 1, 3]);
        let mul39_out1 = transpose31_out1.clone().mul(unsqueeze10_out1.clone());
        let slice46_out1 = transpose31_out1.clone().slice(s![.., .., .., 0..32]);
        let slice47_out1 = transpose31_out1.slice(s![.., .., .., 32..]);
        let neg13_out1 = slice47_out1.neg();
        let concat18_out1 = burn::tensor::Tensor::cat(
            [neg13_out1, slice46_out1].into(),
            3,
        );
        let mul40_out1 = concat18_out1.mul(unsqueeze11_out1.clone());
        let add31_out1 = mul39_out1.add(mul40_out1);
        let mul41_out1 = transpose32_out1.clone().mul(unsqueeze10_out1);
        let slice48_out1 = transpose32_out1.clone().slice(s![.., .., .., 0..32]);
        let slice49_out1 = transpose32_out1.slice(s![.., .., .., 32..]);
        let neg14_out1 = slice49_out1.neg();
        let concat19_out1 = burn::tensor::Tensor::cat(
            [neg14_out1, slice48_out1].into(),
            3,
        );
        let mul42_out1 = concat19_out1.mul(unsqueeze11_out1);
        let add32_out1 = mul41_out1.add(mul42_out1);
        let matmul39_out1 = crate::inference::attention::forward(
            add31_out1,
            add32_out1,
            transpose33_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose35_out1 = matmul39_out1.permute([0, 2, 1, 3]);
        let reshape14_out1 = transpose35_out1.reshape(concat7_out1);
        let linear26_out1 = self.linear26.forward(reshape14_out1);
        let add34_out1 = add30_out1.add(linear26_out1);
        add34_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5 {
    layernormalization14: LayerNorm,
    linear27: Linear,
    linear28: Linear,
    layernormalization15: LayerNorm,
    linear29: Linear,
    linear30: Linear,
    layernormalization16: LayerNorm,
    linear31: Linear,
    linear32: Linear,
    layernormalization17: LayerNorm,
    linear33: Linear,
    linear34: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule5 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization14 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear27 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear28 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization15 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear29 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear30 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization16 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear31 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear32 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization17 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear33 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear34 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization14,
            linear27,
            linear28,
            layernormalization15,
            linear29,
            linear30,
            layernormalization16,
            linear31,
            linear32,
            layernormalization17,
            linear33,
            linear34,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add34_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
        concat7_out1: [i64; 3],
    ) -> Tensor<3> {
        let layernormalization14_out1 = {
            let dtype = add34_out1.clone().dtype();
            self.layernormalization14
                .forward(add34_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear27_out1 = self.linear27.forward(layernormalization14_out1);
        let split_tensors = linear27_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split7_out1, split7_out2] = split_tensors.try_into().unwrap();
        let gelu7_out1 = burn::tensor::activation::gelu(split7_out1);
        let mul44_out1 = gelu7_out1.mul(split7_out2);
        let linear28_out1 = self.linear28.forward(mul44_out1);
        let add35_out1 = add34_out1.add(linear28_out1);
        let layernormalization15_out1 = {
            let dtype = add35_out1.clone().dtype();
            self.layernormalization15
                .forward(add35_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear29_out1 = self.linear29.forward(layernormalization15_out1);
        let reshape15_out1 = linear29_out1.reshape(concat4_out1);
        let slice50_out1 = reshape15_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze24_out1 = slice50_out1.squeeze_dims::<4>(&[-3]);
        let slice51_out1 = reshape15_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze25_out1 = slice51_out1.squeeze_dims::<4>(&[-3]);
        let slice52_out1 = reshape15_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze26_out1 = slice52_out1.squeeze_dims::<4>(&[-3]);
        let transpose36_out1 = squeeze24_out1.permute([0, 2, 1, 3]);
        let transpose37_out1 = squeeze25_out1.permute([0, 2, 1, 3]);
        let transpose38_out1 = squeeze26_out1.permute([0, 2, 1, 3]);
        let mul45_out1 = transpose36_out1.clone().mul(unsqueeze12_out1.clone());
        let slice53_out1 = transpose36_out1.clone().slice(s![.., .., .., 0..32]);
        let slice54_out1 = transpose36_out1.slice(s![.., .., .., 32..]);
        let neg15_out1 = slice54_out1.neg();
        let concat20_out1 = burn::tensor::Tensor::cat(
            [neg15_out1, slice53_out1].into(),
            3,
        );
        let mul46_out1 = concat20_out1.mul(unsqueeze13_out1.clone());
        let add36_out1 = mul45_out1.add(mul46_out1);
        let mul47_out1 = transpose37_out1.clone().mul(unsqueeze12_out1.clone());
        let slice55_out1 = transpose37_out1.clone().slice(s![.., .., .., 0..32]);
        let slice56_out1 = transpose37_out1.slice(s![.., .., .., 32..]);
        let neg16_out1 = slice56_out1.neg();
        let concat21_out1 = burn::tensor::Tensor::cat(
            [neg16_out1, slice55_out1].into(),
            3,
        );
        let mul48_out1 = concat21_out1.mul(unsqueeze13_out1.clone());
        let add37_out1 = mul47_out1.add(mul48_out1);
        let matmul45_out1 = crate::inference::attention::forward(
            add36_out1,
            add37_out1,
            transpose38_out1,
            Some(where2_out1.clone()),
            Some(0.125f64),
        );
        let transpose40_out1 = matmul45_out1.permute([0, 2, 1, 3]);
        let reshape16_out1 = transpose40_out1.reshape(concat7_out1);
        let linear30_out1 = self.linear30.forward(reshape16_out1);
        let add39_out1 = add35_out1.add(linear30_out1);
        let layernormalization16_out1 = {
            let dtype = add39_out1.clone().dtype();
            self.layernormalization16
                .forward(add39_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear31_out1 = self.linear31.forward(layernormalization16_out1);
        let split_tensors = linear31_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split8_out1, split8_out2] = split_tensors.try_into().unwrap();
        let gelu8_out1 = burn::tensor::activation::gelu(split8_out1);
        let mul50_out1 = gelu8_out1.mul(split8_out2);
        let linear32_out1 = self.linear32.forward(mul50_out1);
        let add40_out1 = add39_out1.add(linear32_out1);
        let layernormalization17_out1 = {
            let dtype = add40_out1.clone().dtype();
            self.layernormalization17
                .forward(add40_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear33_out1 = self.linear33.forward(layernormalization17_out1);
        let reshape17_out1 = linear33_out1.reshape(concat4_out1);
        let slice57_out1 = reshape17_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze27_out1 = slice57_out1.squeeze_dims::<4>(&[-3]);
        let slice58_out1 = reshape17_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze28_out1 = slice58_out1.squeeze_dims::<4>(&[-3]);
        let slice59_out1 = reshape17_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze29_out1 = slice59_out1.squeeze_dims::<4>(&[-3]);
        let transpose41_out1 = squeeze27_out1.permute([0, 2, 1, 3]);
        let transpose42_out1 = squeeze28_out1.permute([0, 2, 1, 3]);
        let transpose43_out1 = squeeze29_out1.permute([0, 2, 1, 3]);
        let mul51_out1 = transpose41_out1.clone().mul(unsqueeze12_out1.clone());
        let slice60_out1 = transpose41_out1.clone().slice(s![.., .., .., 0..32]);
        let slice61_out1 = transpose41_out1.slice(s![.., .., .., 32..]);
        let neg17_out1 = slice61_out1.neg();
        let concat22_out1 = burn::tensor::Tensor::cat(
            [neg17_out1, slice60_out1].into(),
            3,
        );
        let mul52_out1 = concat22_out1.mul(unsqueeze13_out1.clone());
        let add41_out1 = mul51_out1.add(mul52_out1);
        let mul53_out1 = transpose42_out1.clone().mul(unsqueeze12_out1);
        let slice62_out1 = transpose42_out1.clone().slice(s![.., .., .., 0..32]);
        let slice63_out1 = transpose42_out1.slice(s![.., .., .., 32..]);
        let neg18_out1 = slice63_out1.neg();
        let concat23_out1 = burn::tensor::Tensor::cat(
            [neg18_out1, slice62_out1].into(),
            3,
        );
        let mul54_out1 = concat23_out1.mul(unsqueeze13_out1);
        let add42_out1 = mul53_out1.add(mul54_out1);
        let matmul51_out1 = crate::inference::attention::forward(
            add41_out1,
            add42_out1,
            transpose43_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose45_out1 = matmul51_out1.permute([0, 2, 1, 3]);
        let reshape18_out1 = transpose45_out1.reshape(concat7_out1);
        let linear34_out1 = self.linear34.forward(reshape18_out1);
        let add44_out1 = add40_out1.add(linear34_out1);
        add44_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6 {
    layernormalization18: LayerNorm,
    linear35: Linear,
    linear36: Linear,
    layernormalization19: LayerNorm,
    linear37: Linear,
    linear38: Linear,
    layernormalization20: LayerNorm,
    linear39: Linear,
    linear40: Linear,
    layernormalization21: LayerNorm,
    linear41: Linear,
    linear42: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule6 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization18 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear35 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear36 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization19 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear37 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear38 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization20 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear39 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear40 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization21 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear41 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear42 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization18,
            linear35,
            linear36,
            layernormalization19,
            linear37,
            linear38,
            layernormalization20,
            linear39,
            linear40,
            layernormalization21,
            linear41,
            linear42,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add44_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
    ) -> Tensor<3> {
        let layernormalization18_out1 = {
            let dtype = add44_out1.clone().dtype();
            self.layernormalization18
                .forward(add44_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear35_out1 = self.linear35.forward(layernormalization18_out1);
        let split_tensors = linear35_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split9_out1, split9_out2] = split_tensors.try_into().unwrap();
        let gelu9_out1 = burn::tensor::activation::gelu(split9_out1);
        let mul56_out1 = gelu9_out1.mul(split9_out2);
        let linear36_out1 = self.linear36.forward(mul56_out1);
        let add45_out1 = add44_out1.add(linear36_out1);
        let layernormalization19_out1 = {
            let dtype = add45_out1.clone().dtype();
            self.layernormalization19
                .forward(add45_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear37_out1 = self.linear37.forward(layernormalization19_out1);
        let reshape19_out1 = linear37_out1.reshape(concat4_out1);
        let slice64_out1 = reshape19_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze30_out1 = slice64_out1.squeeze_dims::<4>(&[-3]);
        let slice65_out1 = reshape19_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze31_out1 = slice65_out1.squeeze_dims::<4>(&[-3]);
        let slice66_out1 = reshape19_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze32_out1 = slice66_out1.squeeze_dims::<4>(&[-3]);
        let transpose46_out1 = squeeze30_out1.permute([0, 2, 1, 3]);
        let transpose47_out1 = squeeze31_out1.permute([0, 2, 1, 3]);
        let transpose48_out1 = squeeze32_out1.permute([0, 2, 1, 3]);
        let mul57_out1 = transpose46_out1.clone().mul(unsqueeze10_out1.clone());
        let slice67_out1 = transpose46_out1.clone().slice(s![.., .., .., 0..32]);
        let slice68_out1 = transpose46_out1.slice(s![.., .., .., 32..]);
        let neg19_out1 = slice68_out1.neg();
        let concat24_out1 = burn::tensor::Tensor::cat(
            [neg19_out1, slice67_out1].into(),
            3,
        );
        let mul58_out1 = concat24_out1.mul(unsqueeze11_out1.clone());
        let add46_out1 = mul57_out1.add(mul58_out1);
        let mul59_out1 = transpose47_out1.clone().mul(unsqueeze10_out1);
        let slice69_out1 = transpose47_out1.clone().slice(s![.., .., .., 0..32]);
        let slice70_out1 = transpose47_out1.slice(s![.., .., .., 32..]);
        let neg20_out1 = slice70_out1.neg();
        let concat25_out1 = burn::tensor::Tensor::cat(
            [neg20_out1, slice69_out1].into(),
            3,
        );
        let mul60_out1 = concat25_out1.mul(unsqueeze11_out1);
        let add47_out1 = mul59_out1.add(mul60_out1);
        let matmul57_out1 = crate::inference::attention::forward(
            add46_out1,
            add47_out1,
            transpose48_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose50_out1 = matmul57_out1.permute([0, 2, 1, 3]);
        let reshape20_out1 = transpose50_out1.reshape(concat7_out1);
        let linear38_out1 = self.linear38.forward(reshape20_out1);
        let add49_out1 = add45_out1.add(linear38_out1);
        let layernormalization20_out1 = {
            let dtype = add49_out1.clone().dtype();
            self.layernormalization20
                .forward(add49_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear39_out1 = self.linear39.forward(layernormalization20_out1);
        let split_tensors = linear39_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split10_out1, split10_out2] = split_tensors.try_into().unwrap();
        let gelu10_out1 = burn::tensor::activation::gelu(split10_out1);
        let mul62_out1 = gelu10_out1.mul(split10_out2);
        let linear40_out1 = self.linear40.forward(mul62_out1);
        let add50_out1 = add49_out1.add(linear40_out1);
        let layernormalization21_out1 = {
            let dtype = add50_out1.clone().dtype();
            self.layernormalization21
                .forward(add50_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear41_out1 = self.linear41.forward(layernormalization21_out1);
        let reshape21_out1 = linear41_out1.reshape(concat4_out1);
        let slice71_out1 = reshape21_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze33_out1 = slice71_out1.squeeze_dims::<4>(&[-3]);
        let slice72_out1 = reshape21_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze34_out1 = slice72_out1.squeeze_dims::<4>(&[-3]);
        let slice73_out1 = reshape21_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze35_out1 = slice73_out1.squeeze_dims::<4>(&[-3]);
        let transpose51_out1 = squeeze33_out1.permute([0, 2, 1, 3]);
        let transpose52_out1 = squeeze34_out1.permute([0, 2, 1, 3]);
        let transpose53_out1 = squeeze35_out1.permute([0, 2, 1, 3]);
        let mul63_out1 = transpose51_out1.clone().mul(unsqueeze12_out1.clone());
        let slice74_out1 = transpose51_out1.clone().slice(s![.., .., .., 0..32]);
        let slice75_out1 = transpose51_out1.slice(s![.., .., .., 32..]);
        let neg21_out1 = slice75_out1.neg();
        let concat26_out1 = burn::tensor::Tensor::cat(
            [neg21_out1, slice74_out1].into(),
            3,
        );
        let mul64_out1 = concat26_out1.mul(unsqueeze13_out1.clone());
        let add51_out1 = mul63_out1.add(mul64_out1);
        let mul65_out1 = transpose52_out1.clone().mul(unsqueeze12_out1);
        let slice76_out1 = transpose52_out1.clone().slice(s![.., .., .., 0..32]);
        let slice77_out1 = transpose52_out1.slice(s![.., .., .., 32..]);
        let neg22_out1 = slice77_out1.neg();
        let concat27_out1 = burn::tensor::Tensor::cat(
            [neg22_out1, slice76_out1].into(),
            3,
        );
        let mul66_out1 = concat27_out1.mul(unsqueeze13_out1);
        let add52_out1 = mul65_out1.add(mul66_out1);
        let matmul63_out1 = crate::inference::attention::forward(
            add51_out1,
            add52_out1,
            transpose53_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose55_out1 = matmul63_out1.permute([0, 2, 1, 3]);
        let reshape22_out1 = transpose55_out1.reshape(concat7_out1);
        let linear42_out1 = self.linear42.forward(reshape22_out1);
        let add54_out1 = add50_out1.add(linear42_out1);
        add54_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule7 {
    layernormalization22: LayerNorm,
    linear43: Linear,
    linear44: Linear,
    layernormalization23: LayerNorm,
    linear45: Linear,
    linear46: Linear,
    layernormalization24: LayerNorm,
    linear47: Linear,
    linear48: Linear,
    layernormalization25: LayerNorm,
    linear49: Linear,
    linear50: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule7 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization22 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear43 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear44 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization23 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear45 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear46 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization24 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear47 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear48 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization25 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear49 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear50 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization22,
            linear43,
            linear44,
            layernormalization23,
            linear45,
            linear46,
            layernormalization24,
            linear47,
            linear48,
            layernormalization25,
            linear49,
            linear50,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add54_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
    ) -> Tensor<3> {
        let layernormalization22_out1 = {
            let dtype = add54_out1.clone().dtype();
            self.layernormalization22
                .forward(add54_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear43_out1 = self.linear43.forward(layernormalization22_out1);
        let split_tensors = linear43_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split11_out1, split11_out2] = split_tensors.try_into().unwrap();
        let gelu11_out1 = burn::tensor::activation::gelu(split11_out1);
        let mul68_out1 = gelu11_out1.mul(split11_out2);
        let linear44_out1 = self.linear44.forward(mul68_out1);
        let add55_out1 = add54_out1.add(linear44_out1);
        let layernormalization23_out1 = {
            let dtype = add55_out1.clone().dtype();
            self.layernormalization23
                .forward(add55_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear45_out1 = self.linear45.forward(layernormalization23_out1);
        let reshape23_out1 = linear45_out1.reshape(concat4_out1);
        let slice78_out1 = reshape23_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze36_out1 = slice78_out1.squeeze_dims::<4>(&[-3]);
        let slice79_out1 = reshape23_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze37_out1 = slice79_out1.squeeze_dims::<4>(&[-3]);
        let slice80_out1 = reshape23_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze38_out1 = slice80_out1.squeeze_dims::<4>(&[-3]);
        let transpose56_out1 = squeeze36_out1.permute([0, 2, 1, 3]);
        let transpose57_out1 = squeeze37_out1.permute([0, 2, 1, 3]);
        let transpose58_out1 = squeeze38_out1.permute([0, 2, 1, 3]);
        let mul69_out1 = transpose56_out1.clone().mul(unsqueeze12_out1.clone());
        let slice81_out1 = transpose56_out1.clone().slice(s![.., .., .., 0..32]);
        let slice82_out1 = transpose56_out1.slice(s![.., .., .., 32..]);
        let neg23_out1 = slice82_out1.neg();
        let concat28_out1 = burn::tensor::Tensor::cat(
            [neg23_out1, slice81_out1].into(),
            3,
        );
        let mul70_out1 = concat28_out1.mul(unsqueeze13_out1.clone());
        let add56_out1 = mul69_out1.add(mul70_out1);
        let mul71_out1 = transpose57_out1.clone().mul(unsqueeze12_out1);
        let slice83_out1 = transpose57_out1.clone().slice(s![.., .., .., 0..32]);
        let slice84_out1 = transpose57_out1.slice(s![.., .., .., 32..]);
        let neg24_out1 = slice84_out1.neg();
        let concat29_out1 = burn::tensor::Tensor::cat(
            [neg24_out1, slice83_out1].into(),
            3,
        );
        let mul72_out1 = concat29_out1.mul(unsqueeze13_out1);
        let add57_out1 = mul71_out1.add(mul72_out1);
        let matmul69_out1 = crate::inference::attention::forward(
            add56_out1,
            add57_out1,
            transpose58_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose60_out1 = matmul69_out1.permute([0, 2, 1, 3]);
        let reshape24_out1 = transpose60_out1.reshape(concat7_out1);
        let linear46_out1 = self.linear46.forward(reshape24_out1);
        let add59_out1 = add55_out1.add(linear46_out1);
        let layernormalization24_out1 = {
            let dtype = add59_out1.clone().dtype();
            self.layernormalization24
                .forward(add59_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear47_out1 = self.linear47.forward(layernormalization24_out1);
        let split_tensors = linear47_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split12_out1, split12_out2] = split_tensors.try_into().unwrap();
        let gelu12_out1 = burn::tensor::activation::gelu(split12_out1);
        let mul74_out1 = gelu12_out1.mul(split12_out2);
        let linear48_out1 = self.linear48.forward(mul74_out1);
        let add60_out1 = add59_out1.add(linear48_out1);
        let layernormalization25_out1 = {
            let dtype = add60_out1.clone().dtype();
            self.layernormalization25
                .forward(add60_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear49_out1 = self.linear49.forward(layernormalization25_out1);
        let reshape25_out1 = linear49_out1.reshape(concat4_out1);
        let slice85_out1 = reshape25_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze39_out1 = slice85_out1.squeeze_dims::<4>(&[-3]);
        let slice86_out1 = reshape25_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze40_out1 = slice86_out1.squeeze_dims::<4>(&[-3]);
        let slice87_out1 = reshape25_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze41_out1 = slice87_out1.squeeze_dims::<4>(&[-3]);
        let transpose61_out1 = squeeze39_out1.permute([0, 2, 1, 3]);
        let transpose62_out1 = squeeze40_out1.permute([0, 2, 1, 3]);
        let transpose63_out1 = squeeze41_out1.permute([0, 2, 1, 3]);
        let mul75_out1 = transpose61_out1.clone().mul(unsqueeze10_out1.clone());
        let slice88_out1 = transpose61_out1.clone().slice(s![.., .., .., 0..32]);
        let slice89_out1 = transpose61_out1.slice(s![.., .., .., 32..]);
        let neg25_out1 = slice89_out1.neg();
        let concat30_out1 = burn::tensor::Tensor::cat(
            [neg25_out1, slice88_out1].into(),
            3,
        );
        let mul76_out1 = concat30_out1.mul(unsqueeze11_out1.clone());
        let add61_out1 = mul75_out1.add(mul76_out1);
        let mul77_out1 = transpose62_out1.clone().mul(unsqueeze10_out1);
        let slice90_out1 = transpose62_out1.clone().slice(s![.., .., .., 0..32]);
        let slice91_out1 = transpose62_out1.slice(s![.., .., .., 32..]);
        let neg26_out1 = slice91_out1.neg();
        let concat31_out1 = burn::tensor::Tensor::cat(
            [neg26_out1, slice90_out1].into(),
            3,
        );
        let mul78_out1 = concat31_out1.mul(unsqueeze11_out1);
        let add62_out1 = mul77_out1.add(mul78_out1);
        let matmul75_out1 = crate::inference::attention::forward(
            add61_out1,
            add62_out1,
            transpose63_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose65_out1 = matmul75_out1.permute([0, 2, 1, 3]);
        let reshape26_out1 = transpose65_out1.reshape(concat7_out1);
        let linear50_out1 = self.linear50.forward(reshape26_out1);
        let add64_out1 = add60_out1.add(linear50_out1);
        add64_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule8 {
    layernormalization26: LayerNorm,
    linear51: Linear,
    linear52: Linear,
    layernormalization27: LayerNorm,
    linear53: Linear,
    linear54: Linear,
    layernormalization28: LayerNorm,
    linear55: Linear,
    linear56: Linear,
    layernormalization29: LayerNorm,
    linear57: Linear,
    linear58: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule8 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization26 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear51 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear52 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization27 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear53 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear54 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization28 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear55 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear56 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization29 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear57 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear58 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization26,
            linear51,
            linear52,
            layernormalization27,
            linear53,
            linear54,
            layernormalization28,
            linear55,
            linear56,
            layernormalization29,
            linear57,
            linear58,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add64_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
        concat7_out1: [i64; 3],
    ) -> Tensor<3> {
        let layernormalization26_out1 = {
            let dtype = add64_out1.clone().dtype();
            self.layernormalization26
                .forward(add64_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear51_out1 = self.linear51.forward(layernormalization26_out1);
        let split_tensors = linear51_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split13_out1, split13_out2] = split_tensors.try_into().unwrap();
        let gelu13_out1 = burn::tensor::activation::gelu(split13_out1);
        let mul80_out1 = gelu13_out1.mul(split13_out2);
        let linear52_out1 = self.linear52.forward(mul80_out1);
        let add65_out1 = add64_out1.add(linear52_out1);
        let layernormalization27_out1 = {
            let dtype = add65_out1.clone().dtype();
            self.layernormalization27
                .forward(add65_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear53_out1 = self.linear53.forward(layernormalization27_out1);
        let reshape27_out1 = linear53_out1.reshape(concat4_out1);
        let slice92_out1 = reshape27_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze42_out1 = slice92_out1.squeeze_dims::<4>(&[-3]);
        let slice93_out1 = reshape27_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze43_out1 = slice93_out1.squeeze_dims::<4>(&[-3]);
        let slice94_out1 = reshape27_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze44_out1 = slice94_out1.squeeze_dims::<4>(&[-3]);
        let transpose66_out1 = squeeze42_out1.permute([0, 2, 1, 3]);
        let transpose67_out1 = squeeze43_out1.permute([0, 2, 1, 3]);
        let transpose68_out1 = squeeze44_out1.permute([0, 2, 1, 3]);
        let mul81_out1 = transpose66_out1.clone().mul(unsqueeze12_out1.clone());
        let slice95_out1 = transpose66_out1.clone().slice(s![.., .., .., 0..32]);
        let slice96_out1 = transpose66_out1.slice(s![.., .., .., 32..]);
        let neg27_out1 = slice96_out1.neg();
        let concat32_out1 = burn::tensor::Tensor::cat(
            [neg27_out1, slice95_out1].into(),
            3,
        );
        let mul82_out1 = concat32_out1.mul(unsqueeze13_out1.clone());
        let add66_out1 = mul81_out1.add(mul82_out1);
        let mul83_out1 = transpose67_out1.clone().mul(unsqueeze12_out1.clone());
        let slice97_out1 = transpose67_out1.clone().slice(s![.., .., .., 0..32]);
        let slice98_out1 = transpose67_out1.slice(s![.., .., .., 32..]);
        let neg28_out1 = slice98_out1.neg();
        let concat33_out1 = burn::tensor::Tensor::cat(
            [neg28_out1, slice97_out1].into(),
            3,
        );
        let mul84_out1 = concat33_out1.mul(unsqueeze13_out1.clone());
        let add67_out1 = mul83_out1.add(mul84_out1);
        let matmul81_out1 = crate::inference::attention::forward(
            add66_out1,
            add67_out1,
            transpose68_out1,
            Some(where2_out1.clone()),
            Some(0.125f64),
        );
        let transpose70_out1 = matmul81_out1.permute([0, 2, 1, 3]);
        let reshape28_out1 = transpose70_out1.reshape(concat7_out1);
        let linear54_out1 = self.linear54.forward(reshape28_out1);
        let add69_out1 = add65_out1.add(linear54_out1);
        let layernormalization28_out1 = {
            let dtype = add69_out1.clone().dtype();
            self.layernormalization28
                .forward(add69_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear55_out1 = self.linear55.forward(layernormalization28_out1);
        let split_tensors = linear55_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split14_out1, split14_out2] = split_tensors.try_into().unwrap();
        let gelu14_out1 = burn::tensor::activation::gelu(split14_out1);
        let mul86_out1 = gelu14_out1.mul(split14_out2);
        let linear56_out1 = self.linear56.forward(mul86_out1);
        let add70_out1 = add69_out1.add(linear56_out1);
        let layernormalization29_out1 = {
            let dtype = add70_out1.clone().dtype();
            self.layernormalization29
                .forward(add70_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear57_out1 = self.linear57.forward(layernormalization29_out1);
        let reshape29_out1 = linear57_out1.reshape(concat4_out1);
        let slice99_out1 = reshape29_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze45_out1 = slice99_out1.squeeze_dims::<4>(&[-3]);
        let slice100_out1 = reshape29_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze46_out1 = slice100_out1.squeeze_dims::<4>(&[-3]);
        let slice101_out1 = reshape29_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze47_out1 = slice101_out1.squeeze_dims::<4>(&[-3]);
        let transpose71_out1 = squeeze45_out1.permute([0, 2, 1, 3]);
        let transpose72_out1 = squeeze46_out1.permute([0, 2, 1, 3]);
        let transpose73_out1 = squeeze47_out1.permute([0, 2, 1, 3]);
        let mul87_out1 = transpose71_out1.clone().mul(unsqueeze12_out1.clone());
        let slice102_out1 = transpose71_out1.clone().slice(s![.., .., .., 0..32]);
        let slice103_out1 = transpose71_out1.slice(s![.., .., .., 32..]);
        let neg29_out1 = slice103_out1.neg();
        let concat34_out1 = burn::tensor::Tensor::cat(
            [neg29_out1, slice102_out1].into(),
            3,
        );
        let mul88_out1 = concat34_out1.mul(unsqueeze13_out1.clone());
        let add71_out1 = mul87_out1.add(mul88_out1);
        let mul89_out1 = transpose72_out1.clone().mul(unsqueeze12_out1);
        let slice104_out1 = transpose72_out1.clone().slice(s![.., .., .., 0..32]);
        let slice105_out1 = transpose72_out1.slice(s![.., .., .., 32..]);
        let neg30_out1 = slice105_out1.neg();
        let concat35_out1 = burn::tensor::Tensor::cat(
            [neg30_out1, slice104_out1].into(),
            3,
        );
        let mul90_out1 = concat35_out1.mul(unsqueeze13_out1);
        let add72_out1 = mul89_out1.add(mul90_out1);
        let matmul87_out1 = crate::inference::attention::forward(
            add71_out1,
            add72_out1,
            transpose73_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose75_out1 = matmul87_out1.permute([0, 2, 1, 3]);
        let reshape30_out1 = transpose75_out1.reshape(concat7_out1);
        let linear58_out1 = self.linear58.forward(reshape30_out1);
        let add74_out1 = add70_out1.add(linear58_out1);
        add74_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule9 {
    layernormalization30: LayerNorm,
    linear59: Linear,
    linear60: Linear,
    layernormalization31: LayerNorm,
    linear61: Linear,
    linear62: Linear,
    layernormalization32: LayerNorm,
    linear63: Linear,
    linear64: Linear,
    layernormalization33: LayerNorm,
    linear65: Linear,
    linear66: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule9 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization30 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear59 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear60 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization31 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear61 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear62 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization32 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear63 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear64 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization33 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear65 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear66 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization30,
            linear59,
            linear60,
            layernormalization31,
            linear61,
            linear62,
            layernormalization32,
            linear63,
            linear64,
            layernormalization33,
            linear65,
            linear66,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add74_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
    ) -> Tensor<3> {
        let layernormalization30_out1 = {
            let dtype = add74_out1.clone().dtype();
            self.layernormalization30
                .forward(add74_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear59_out1 = self.linear59.forward(layernormalization30_out1);
        let split_tensors = linear59_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split15_out1, split15_out2] = split_tensors.try_into().unwrap();
        let gelu15_out1 = burn::tensor::activation::gelu(split15_out1);
        let mul92_out1 = gelu15_out1.mul(split15_out2);
        let linear60_out1 = self.linear60.forward(mul92_out1);
        let add75_out1 = add74_out1.add(linear60_out1);
        let layernormalization31_out1 = {
            let dtype = add75_out1.clone().dtype();
            self.layernormalization31
                .forward(add75_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear61_out1 = self.linear61.forward(layernormalization31_out1);
        let reshape31_out1 = linear61_out1.reshape(concat4_out1);
        let slice106_out1 = reshape31_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze48_out1 = slice106_out1.squeeze_dims::<4>(&[-3]);
        let slice107_out1 = reshape31_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze49_out1 = slice107_out1.squeeze_dims::<4>(&[-3]);
        let slice108_out1 = reshape31_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze50_out1 = slice108_out1.squeeze_dims::<4>(&[-3]);
        let transpose76_out1 = squeeze48_out1.permute([0, 2, 1, 3]);
        let transpose77_out1 = squeeze49_out1.permute([0, 2, 1, 3]);
        let transpose78_out1 = squeeze50_out1.permute([0, 2, 1, 3]);
        let mul93_out1 = transpose76_out1.clone().mul(unsqueeze10_out1.clone());
        let slice109_out1 = transpose76_out1.clone().slice(s![.., .., .., 0..32]);
        let slice110_out1 = transpose76_out1.slice(s![.., .., .., 32..]);
        let neg31_out1 = slice110_out1.neg();
        let concat36_out1 = burn::tensor::Tensor::cat(
            [neg31_out1, slice109_out1].into(),
            3,
        );
        let mul94_out1 = concat36_out1.mul(unsqueeze11_out1.clone());
        let add76_out1 = mul93_out1.add(mul94_out1);
        let mul95_out1 = transpose77_out1.clone().mul(unsqueeze10_out1);
        let slice111_out1 = transpose77_out1.clone().slice(s![.., .., .., 0..32]);
        let slice112_out1 = transpose77_out1.slice(s![.., .., .., 32..]);
        let neg32_out1 = slice112_out1.neg();
        let concat37_out1 = burn::tensor::Tensor::cat(
            [neg32_out1, slice111_out1].into(),
            3,
        );
        let mul96_out1 = concat37_out1.mul(unsqueeze11_out1);
        let add77_out1 = mul95_out1.add(mul96_out1);
        let matmul93_out1 = crate::inference::attention::forward(
            add76_out1,
            add77_out1,
            transpose78_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose80_out1 = matmul93_out1.permute([0, 2, 1, 3]);
        let reshape32_out1 = transpose80_out1.reshape(concat7_out1);
        let linear62_out1 = self.linear62.forward(reshape32_out1);
        let add79_out1 = add75_out1.add(linear62_out1);
        let layernormalization32_out1 = {
            let dtype = add79_out1.clone().dtype();
            self.layernormalization32
                .forward(add79_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear63_out1 = self.linear63.forward(layernormalization32_out1);
        let split_tensors = linear63_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split16_out1, split16_out2] = split_tensors.try_into().unwrap();
        let gelu16_out1 = burn::tensor::activation::gelu(split16_out1);
        let mul98_out1 = gelu16_out1.mul(split16_out2);
        let linear64_out1 = self.linear64.forward(mul98_out1);
        let add80_out1 = add79_out1.add(linear64_out1);
        let layernormalization33_out1 = {
            let dtype = add80_out1.clone().dtype();
            self.layernormalization33
                .forward(add80_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear65_out1 = self.linear65.forward(layernormalization33_out1);
        let reshape33_out1 = linear65_out1.reshape(concat4_out1);
        let slice113_out1 = reshape33_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze51_out1 = slice113_out1.squeeze_dims::<4>(&[-3]);
        let slice114_out1 = reshape33_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze52_out1 = slice114_out1.squeeze_dims::<4>(&[-3]);
        let slice115_out1 = reshape33_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze53_out1 = slice115_out1.squeeze_dims::<4>(&[-3]);
        let transpose81_out1 = squeeze51_out1.permute([0, 2, 1, 3]);
        let transpose82_out1 = squeeze52_out1.permute([0, 2, 1, 3]);
        let transpose83_out1 = squeeze53_out1.permute([0, 2, 1, 3]);
        let mul99_out1 = transpose81_out1.clone().mul(unsqueeze12_out1.clone());
        let slice116_out1 = transpose81_out1.clone().slice(s![.., .., .., 0..32]);
        let slice117_out1 = transpose81_out1.slice(s![.., .., .., 32..]);
        let neg33_out1 = slice117_out1.neg();
        let concat38_out1 = burn::tensor::Tensor::cat(
            [neg33_out1, slice116_out1].into(),
            3,
        );
        let mul100_out1 = concat38_out1.mul(unsqueeze13_out1.clone());
        let add81_out1 = mul99_out1.add(mul100_out1);
        let mul101_out1 = transpose82_out1.clone().mul(unsqueeze12_out1);
        let slice118_out1 = transpose82_out1.clone().slice(s![.., .., .., 0..32]);
        let slice119_out1 = transpose82_out1.slice(s![.., .., .., 32..]);
        let neg34_out1 = slice119_out1.neg();
        let concat39_out1 = burn::tensor::Tensor::cat(
            [neg34_out1, slice118_out1].into(),
            3,
        );
        let mul102_out1 = concat39_out1.mul(unsqueeze13_out1);
        let add82_out1 = mul101_out1.add(mul102_out1);
        let matmul99_out1 = crate::inference::attention::forward(
            add81_out1,
            add82_out1,
            transpose83_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose85_out1 = matmul99_out1.permute([0, 2, 1, 3]);
        let reshape34_out1 = transpose85_out1.reshape(concat7_out1);
        let linear66_out1 = self.linear66.forward(reshape34_out1);
        let add84_out1 = add80_out1.add(linear66_out1);
        add84_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule10 {
    layernormalization34: LayerNorm,
    linear67: Linear,
    linear68: Linear,
    layernormalization35: LayerNorm,
    linear69: Linear,
    linear70: Linear,
    layernormalization36: LayerNorm,
    linear71: Linear,
    linear72: Linear,
    layernormalization37: LayerNorm,
    linear73: Linear,
    linear74: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule10 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization34 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear67 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear68 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization35 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear69 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear70 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization36 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear71 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear72 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization37 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear73 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear74 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization34,
            linear67,
            linear68,
            layernormalization35,
            linear69,
            linear70,
            layernormalization36,
            linear71,
            linear72,
            layernormalization37,
            linear73,
            linear74,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add84_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
    ) -> Tensor<3> {
        let layernormalization34_out1 = {
            let dtype = add84_out1.clone().dtype();
            self.layernormalization34
                .forward(add84_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear67_out1 = self.linear67.forward(layernormalization34_out1);
        let split_tensors = linear67_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split17_out1, split17_out2] = split_tensors.try_into().unwrap();
        let gelu17_out1 = burn::tensor::activation::gelu(split17_out1);
        let mul104_out1 = gelu17_out1.mul(split17_out2);
        let linear68_out1 = self.linear68.forward(mul104_out1);
        let add85_out1 = add84_out1.add(linear68_out1);
        let layernormalization35_out1 = {
            let dtype = add85_out1.clone().dtype();
            self.layernormalization35
                .forward(add85_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear69_out1 = self.linear69.forward(layernormalization35_out1);
        let reshape35_out1 = linear69_out1.reshape(concat4_out1);
        let slice120_out1 = reshape35_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze54_out1 = slice120_out1.squeeze_dims::<4>(&[-3]);
        let slice121_out1 = reshape35_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze55_out1 = slice121_out1.squeeze_dims::<4>(&[-3]);
        let slice122_out1 = reshape35_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze56_out1 = slice122_out1.squeeze_dims::<4>(&[-3]);
        let transpose86_out1 = squeeze54_out1.permute([0, 2, 1, 3]);
        let transpose87_out1 = squeeze55_out1.permute([0, 2, 1, 3]);
        let transpose88_out1 = squeeze56_out1.permute([0, 2, 1, 3]);
        let mul105_out1 = transpose86_out1.clone().mul(unsqueeze12_out1.clone());
        let slice123_out1 = transpose86_out1.clone().slice(s![.., .., .., 0..32]);
        let slice124_out1 = transpose86_out1.slice(s![.., .., .., 32..]);
        let neg35_out1 = slice124_out1.neg();
        let concat40_out1 = burn::tensor::Tensor::cat(
            [neg35_out1, slice123_out1].into(),
            3,
        );
        let mul106_out1 = concat40_out1.mul(unsqueeze13_out1.clone());
        let add86_out1 = mul105_out1.add(mul106_out1);
        let mul107_out1 = transpose87_out1.clone().mul(unsqueeze12_out1);
        let slice125_out1 = transpose87_out1.clone().slice(s![.., .., .., 0..32]);
        let slice126_out1 = transpose87_out1.slice(s![.., .., .., 32..]);
        let neg36_out1 = slice126_out1.neg();
        let concat41_out1 = burn::tensor::Tensor::cat(
            [neg36_out1, slice125_out1].into(),
            3,
        );
        let mul108_out1 = concat41_out1.mul(unsqueeze13_out1);
        let add87_out1 = mul107_out1.add(mul108_out1);
        let matmul105_out1 = crate::inference::attention::forward(
            add86_out1,
            add87_out1,
            transpose88_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose90_out1 = matmul105_out1.permute([0, 2, 1, 3]);
        let reshape36_out1 = transpose90_out1.reshape(concat7_out1);
        let linear70_out1 = self.linear70.forward(reshape36_out1);
        let add89_out1 = add85_out1.add(linear70_out1);
        let layernormalization36_out1 = {
            let dtype = add89_out1.clone().dtype();
            self.layernormalization36
                .forward(add89_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear71_out1 = self.linear71.forward(layernormalization36_out1);
        let split_tensors = linear71_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split18_out1, split18_out2] = split_tensors.try_into().unwrap();
        let gelu18_out1 = burn::tensor::activation::gelu(split18_out1);
        let mul110_out1 = gelu18_out1.mul(split18_out2);
        let linear72_out1 = self.linear72.forward(mul110_out1);
        let add90_out1 = add89_out1.add(linear72_out1);
        let layernormalization37_out1 = {
            let dtype = add90_out1.clone().dtype();
            self.layernormalization37
                .forward(add90_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear73_out1 = self.linear73.forward(layernormalization37_out1);
        let reshape37_out1 = linear73_out1.reshape(concat4_out1);
        let slice127_out1 = reshape37_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze57_out1 = slice127_out1.squeeze_dims::<4>(&[-3]);
        let slice128_out1 = reshape37_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze58_out1 = slice128_out1.squeeze_dims::<4>(&[-3]);
        let slice129_out1 = reshape37_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze59_out1 = slice129_out1.squeeze_dims::<4>(&[-3]);
        let transpose91_out1 = squeeze57_out1.permute([0, 2, 1, 3]);
        let transpose92_out1 = squeeze58_out1.permute([0, 2, 1, 3]);
        let transpose93_out1 = squeeze59_out1.permute([0, 2, 1, 3]);
        let mul111_out1 = transpose91_out1.clone().mul(unsqueeze10_out1.clone());
        let slice130_out1 = transpose91_out1.clone().slice(s![.., .., .., 0..32]);
        let slice131_out1 = transpose91_out1.slice(s![.., .., .., 32..]);
        let neg37_out1 = slice131_out1.neg();
        let concat42_out1 = burn::tensor::Tensor::cat(
            [neg37_out1, slice130_out1].into(),
            3,
        );
        let mul112_out1 = concat42_out1.mul(unsqueeze11_out1.clone());
        let add91_out1 = mul111_out1.add(mul112_out1);
        let mul113_out1 = transpose92_out1.clone().mul(unsqueeze10_out1);
        let slice132_out1 = transpose92_out1.clone().slice(s![.., .., .., 0..32]);
        let slice133_out1 = transpose92_out1.slice(s![.., .., .., 32..]);
        let neg38_out1 = slice133_out1.neg();
        let concat43_out1 = burn::tensor::Tensor::cat(
            [neg38_out1, slice132_out1].into(),
            3,
        );
        let mul114_out1 = concat43_out1.mul(unsqueeze11_out1);
        let add92_out1 = mul113_out1.add(mul114_out1);
        let matmul111_out1 = crate::inference::attention::forward(
            add91_out1,
            add92_out1,
            transpose93_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose95_out1 = matmul111_out1.permute([0, 2, 1, 3]);
        let reshape38_out1 = transpose95_out1.reshape(concat7_out1);
        let linear74_out1 = self.linear74.forward(reshape38_out1);
        let add94_out1 = add90_out1.add(linear74_out1);
        add94_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule11 {
    layernormalization38: LayerNorm,
    linear75: Linear,
    linear76: Linear,
    layernormalization39: LayerNorm,
    linear77: Linear,
    linear78: Linear,
    layernormalization40: LayerNorm,
    linear79: Linear,
    linear80: Linear,
    layernormalization41: LayerNorm,
    linear81: Linear,
    linear82: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule11 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization38 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear75 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear76 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization39 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear77 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear78 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization40 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear79 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear80 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization41 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear81 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear82 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization38,
            linear75,
            linear76,
            layernormalization39,
            linear77,
            linear78,
            layernormalization40,
            linear79,
            linear80,
            layernormalization41,
            linear81,
            linear82,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add94_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
        concat7_out1: [i64; 3],
    ) -> Tensor<3> {
        let layernormalization38_out1 = {
            let dtype = add94_out1.clone().dtype();
            self.layernormalization38
                .forward(add94_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear75_out1 = self.linear75.forward(layernormalization38_out1);
        let split_tensors = linear75_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split19_out1, split19_out2] = split_tensors.try_into().unwrap();
        let gelu19_out1 = burn::tensor::activation::gelu(split19_out1);
        let mul116_out1 = gelu19_out1.mul(split19_out2);
        let linear76_out1 = self.linear76.forward(mul116_out1);
        let add95_out1 = add94_out1.add(linear76_out1);
        let layernormalization39_out1 = {
            let dtype = add95_out1.clone().dtype();
            self.layernormalization39
                .forward(add95_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear77_out1 = self.linear77.forward(layernormalization39_out1);
        let reshape39_out1 = linear77_out1.reshape(concat4_out1);
        let slice134_out1 = reshape39_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze60_out1 = slice134_out1.squeeze_dims::<4>(&[-3]);
        let slice135_out1 = reshape39_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze61_out1 = slice135_out1.squeeze_dims::<4>(&[-3]);
        let slice136_out1 = reshape39_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze62_out1 = slice136_out1.squeeze_dims::<4>(&[-3]);
        let transpose96_out1 = squeeze60_out1.permute([0, 2, 1, 3]);
        let transpose97_out1 = squeeze61_out1.permute([0, 2, 1, 3]);
        let transpose98_out1 = squeeze62_out1.permute([0, 2, 1, 3]);
        let mul117_out1 = transpose96_out1.clone().mul(unsqueeze12_out1.clone());
        let slice137_out1 = transpose96_out1.clone().slice(s![.., .., .., 0..32]);
        let slice138_out1 = transpose96_out1.slice(s![.., .., .., 32..]);
        let neg39_out1 = slice138_out1.neg();
        let concat44_out1 = burn::tensor::Tensor::cat(
            [neg39_out1, slice137_out1].into(),
            3,
        );
        let mul118_out1 = concat44_out1.mul(unsqueeze13_out1.clone());
        let add96_out1 = mul117_out1.add(mul118_out1);
        let mul119_out1 = transpose97_out1.clone().mul(unsqueeze12_out1.clone());
        let slice139_out1 = transpose97_out1.clone().slice(s![.., .., .., 0..32]);
        let slice140_out1 = transpose97_out1.slice(s![.., .., .., 32..]);
        let neg40_out1 = slice140_out1.neg();
        let concat45_out1 = burn::tensor::Tensor::cat(
            [neg40_out1, slice139_out1].into(),
            3,
        );
        let mul120_out1 = concat45_out1.mul(unsqueeze13_out1.clone());
        let add97_out1 = mul119_out1.add(mul120_out1);
        let matmul117_out1 = crate::inference::attention::forward(
            add96_out1,
            add97_out1,
            transpose98_out1,
            Some(where2_out1.clone()),
            Some(0.125f64),
        );
        let transpose100_out1 = matmul117_out1.permute([0, 2, 1, 3]);
        let reshape40_out1 = transpose100_out1.reshape(concat7_out1);
        let linear78_out1 = self.linear78.forward(reshape40_out1);
        let add99_out1 = add95_out1.add(linear78_out1);
        let layernormalization40_out1 = {
            let dtype = add99_out1.clone().dtype();
            self.layernormalization40
                .forward(add99_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear79_out1 = self.linear79.forward(layernormalization40_out1);
        let split_tensors = linear79_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split20_out1, split20_out2] = split_tensors.try_into().unwrap();
        let gelu20_out1 = burn::tensor::activation::gelu(split20_out1);
        let mul122_out1 = gelu20_out1.mul(split20_out2);
        let linear80_out1 = self.linear80.forward(mul122_out1);
        let add100_out1 = add99_out1.add(linear80_out1);
        let layernormalization41_out1 = {
            let dtype = add100_out1.clone().dtype();
            self.layernormalization41
                .forward(add100_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear81_out1 = self.linear81.forward(layernormalization41_out1);
        let reshape41_out1 = linear81_out1.reshape(concat4_out1);
        let slice141_out1 = reshape41_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze63_out1 = slice141_out1.squeeze_dims::<4>(&[-3]);
        let slice142_out1 = reshape41_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze64_out1 = slice142_out1.squeeze_dims::<4>(&[-3]);
        let slice143_out1 = reshape41_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze65_out1 = slice143_out1.squeeze_dims::<4>(&[-3]);
        let transpose101_out1 = squeeze63_out1.permute([0, 2, 1, 3]);
        let transpose102_out1 = squeeze64_out1.permute([0, 2, 1, 3]);
        let transpose103_out1 = squeeze65_out1.permute([0, 2, 1, 3]);
        let mul123_out1 = transpose101_out1.clone().mul(unsqueeze12_out1.clone());
        let slice144_out1 = transpose101_out1.clone().slice(s![.., .., .., 0..32]);
        let slice145_out1 = transpose101_out1.slice(s![.., .., .., 32..]);
        let neg41_out1 = slice145_out1.neg();
        let concat46_out1 = burn::tensor::Tensor::cat(
            [neg41_out1, slice144_out1].into(),
            3,
        );
        let mul124_out1 = concat46_out1.mul(unsqueeze13_out1.clone());
        let add101_out1 = mul123_out1.add(mul124_out1);
        let mul125_out1 = transpose102_out1.clone().mul(unsqueeze12_out1);
        let slice146_out1 = transpose102_out1.clone().slice(s![.., .., .., 0..32]);
        let slice147_out1 = transpose102_out1.slice(s![.., .., .., 32..]);
        let neg42_out1 = slice147_out1.neg();
        let concat47_out1 = burn::tensor::Tensor::cat(
            [neg42_out1, slice146_out1].into(),
            3,
        );
        let mul126_out1 = concat47_out1.mul(unsqueeze13_out1);
        let add102_out1 = mul125_out1.add(mul126_out1);
        let matmul123_out1 = crate::inference::attention::forward(
            add101_out1,
            add102_out1,
            transpose103_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose105_out1 = matmul123_out1.permute([0, 2, 1, 3]);
        let reshape42_out1 = transpose105_out1.reshape(concat7_out1);
        let linear82_out1 = self.linear82.forward(reshape42_out1);
        let add104_out1 = add100_out1.add(linear82_out1);
        add104_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule12 {
    layernormalization42: LayerNorm,
    linear83: Linear,
    linear84: Linear,
    layernormalization43: LayerNorm,
    linear85: Linear,
    linear86: Linear,
    layernormalization44: LayerNorm,
    linear87: Linear,
    linear88: Linear,
    layernormalization45: LayerNorm,
    linear89: Linear,
    linear90: Linear,
    layernormalization46: LayerNorm,
    linear91: Linear,
    linear92: Linear,
    layernormalization47: LayerNorm,
    linear93: Linear,
    linear94: Linear,
    layernormalization48: LayerNorm,
    linear95: Linear,
    linear96: Linear,
    layernormalization49: LayerNorm,
    linear97: Linear,
    linear98: Linear,
    layernormalization50: LayerNorm,
    linear99: Linear,
    linear100: Linear,
    layernormalization51: LayerNorm,
    linear101: Linear,
    linear102: Linear,
    layernormalization52: LayerNorm,
    linear103: Linear,
    linear104: Linear,
    layernormalization53: LayerNorm,
    linear105: Linear,
    linear106: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule12 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization42 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear83 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear84 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization43 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear85 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear86 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization44 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear87 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear88 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization45 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear89 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear90 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization46 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear91 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear92 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization47 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear93 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear94 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization48 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear95 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear96 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization49 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear97 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear98 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization50 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear99 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear100 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization51 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear101 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear102 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization52 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear103 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear104 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization53 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear105 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear106 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        Self {
            layernormalization42,
            linear83,
            linear84,
            layernormalization43,
            linear85,
            linear86,
            layernormalization44,
            linear87,
            linear88,
            layernormalization45,
            linear89,
            linear90,
            layernormalization46,
            linear91,
            linear92,
            layernormalization47,
            linear93,
            linear94,
            layernormalization48,
            linear95,
            linear96,
            layernormalization49,
            linear97,
            linear98,
            layernormalization50,
            linear99,
            linear100,
            layernormalization51,
            linear101,
            linear102,
            layernormalization52,
            linear103,
            linear104,
            layernormalization53,
            linear105,
            linear106,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add104_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        unsqueeze12_out1: Tensor<4>,
        unsqueeze13_out1: Tensor<4>,
        where2_out1: Tensor<4>,
    ) -> Tensor<3> {
        let layernormalization42_out1 = {
            let dtype = add104_out1.clone().dtype();
            self.layernormalization42
                .forward(add104_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear83_out1 = self.linear83.forward(layernormalization42_out1);
        let split_tensors = linear83_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split21_out1, split21_out2] = split_tensors.try_into().unwrap();
        let gelu21_out1 = burn::tensor::activation::gelu(split21_out1);
        let mul128_out1 = gelu21_out1.mul(split21_out2);
        let linear84_out1 = self.linear84.forward(mul128_out1);
        let add105_out1 = add104_out1.add(linear84_out1);
        let layernormalization43_out1 = {
            let dtype = add105_out1.clone().dtype();
            self.layernormalization43
                .forward(add105_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear85_out1 = self.linear85.forward(layernormalization43_out1);
        let reshape43_out1 = linear85_out1.reshape(concat4_out1);
        let slice148_out1 = reshape43_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze66_out1 = slice148_out1.squeeze_dims::<4>(&[-3]);
        let slice149_out1 = reshape43_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze67_out1 = slice149_out1.squeeze_dims::<4>(&[-3]);
        let slice150_out1 = reshape43_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze68_out1 = slice150_out1.squeeze_dims::<4>(&[-3]);
        let transpose106_out1 = squeeze66_out1.permute([0, 2, 1, 3]);
        let transpose107_out1 = squeeze67_out1.permute([0, 2, 1, 3]);
        let transpose108_out1 = squeeze68_out1.permute([0, 2, 1, 3]);
        let mul129_out1 = transpose106_out1.clone().mul(unsqueeze10_out1.clone());
        let slice151_out1 = transpose106_out1.clone().slice(s![.., .., .., 0..32]);
        let slice152_out1 = transpose106_out1.slice(s![.., .., .., 32..]);
        let neg43_out1 = slice152_out1.neg();
        let concat48_out1 = burn::tensor::Tensor::cat(
            [neg43_out1, slice151_out1].into(),
            3,
        );
        let mul130_out1 = concat48_out1.mul(unsqueeze11_out1.clone());
        let add106_out1 = mul129_out1.add(mul130_out1);
        let mul131_out1 = transpose107_out1.clone().mul(unsqueeze10_out1.clone());
        let slice153_out1 = transpose107_out1.clone().slice(s![.., .., .., 0..32]);
        let slice154_out1 = transpose107_out1.slice(s![.., .., .., 32..]);
        let neg44_out1 = slice154_out1.neg();
        let concat49_out1 = burn::tensor::Tensor::cat(
            [neg44_out1, slice153_out1].into(),
            3,
        );
        let mul132_out1 = concat49_out1.mul(unsqueeze11_out1.clone());
        let add107_out1 = mul131_out1.add(mul132_out1);
        let matmul129_out1 = crate::inference::attention::forward(
            add106_out1,
            add107_out1,
            transpose108_out1,
            Some(where1_out1.clone()),
            Some(0.125f64),
        );
        let transpose110_out1 = matmul129_out1.permute([0, 2, 1, 3]);
        let reshape44_out1 = transpose110_out1.reshape(concat7_out1);
        let linear86_out1 = self.linear86.forward(reshape44_out1);
        let add109_out1 = add105_out1.add(linear86_out1);
        let layernormalization44_out1 = {
            let dtype = add109_out1.clone().dtype();
            self.layernormalization44
                .forward(add109_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear87_out1 = self.linear87.forward(layernormalization44_out1);
        let split_tensors = linear87_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split22_out1, split22_out2] = split_tensors.try_into().unwrap();
        let gelu22_out1 = burn::tensor::activation::gelu(split22_out1);
        let mul134_out1 = gelu22_out1.mul(split22_out2);
        let linear88_out1 = self.linear88.forward(mul134_out1);
        let add110_out1 = add109_out1.add(linear88_out1);
        let layernormalization45_out1 = {
            let dtype = add110_out1.clone().dtype();
            self.layernormalization45
                .forward(add110_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear89_out1 = self.linear89.forward(layernormalization45_out1);
        let reshape45_out1 = linear89_out1.reshape(concat4_out1);
        let slice155_out1 = reshape45_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze69_out1 = slice155_out1.squeeze_dims::<4>(&[-3]);
        let slice156_out1 = reshape45_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze70_out1 = slice156_out1.squeeze_dims::<4>(&[-3]);
        let slice157_out1 = reshape45_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze71_out1 = slice157_out1.squeeze_dims::<4>(&[-3]);
        let transpose111_out1 = squeeze69_out1.permute([0, 2, 1, 3]);
        let transpose112_out1 = squeeze70_out1.permute([0, 2, 1, 3]);
        let transpose113_out1 = squeeze71_out1.permute([0, 2, 1, 3]);
        let mul135_out1 = transpose111_out1.clone().mul(unsqueeze12_out1.clone());
        let slice158_out1 = transpose111_out1.clone().slice(s![.., .., .., 0..32]);
        let slice159_out1 = transpose111_out1.slice(s![.., .., .., 32..]);
        let neg45_out1 = slice159_out1.neg();
        let concat50_out1 = burn::tensor::Tensor::cat(
            [neg45_out1, slice158_out1].into(),
            3,
        );
        let mul136_out1 = concat50_out1.mul(unsqueeze13_out1.clone());
        let add111_out1 = mul135_out1.add(mul136_out1);
        let mul137_out1 = transpose112_out1.clone().mul(unsqueeze12_out1.clone());
        let slice160_out1 = transpose112_out1.clone().slice(s![.., .., .., 0..32]);
        let slice161_out1 = transpose112_out1.slice(s![.., .., .., 32..]);
        let neg46_out1 = slice161_out1.neg();
        let concat51_out1 = burn::tensor::Tensor::cat(
            [neg46_out1, slice160_out1].into(),
            3,
        );
        let mul138_out1 = concat51_out1.mul(unsqueeze13_out1.clone());
        let add112_out1 = mul137_out1.add(mul138_out1);
        let matmul135_out1 = crate::inference::attention::forward(
            add111_out1,
            add112_out1,
            transpose113_out1,
            Some(where2_out1.clone()),
            Some(0.125f64),
        );
        let transpose115_out1 = matmul135_out1.permute([0, 2, 1, 3]);
        let reshape46_out1 = transpose115_out1.reshape(concat7_out1);
        let linear90_out1 = self.linear90.forward(reshape46_out1);
        let add114_out1 = add110_out1.add(linear90_out1);
        let layernormalization46_out1 = {
            let dtype = add114_out1.clone().dtype();
            self.layernormalization46
                .forward(add114_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear91_out1 = self.linear91.forward(layernormalization46_out1);
        let split_tensors = linear91_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split23_out1, split23_out2] = split_tensors.try_into().unwrap();
        let gelu23_out1 = burn::tensor::activation::gelu(split23_out1);
        let mul140_out1 = gelu23_out1.mul(split23_out2);
        let linear92_out1 = self.linear92.forward(mul140_out1);
        let add115_out1 = add114_out1.add(linear92_out1);
        let layernormalization47_out1 = {
            let dtype = add115_out1.clone().dtype();
            self.layernormalization47
                .forward(add115_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear93_out1 = self.linear93.forward(layernormalization47_out1);
        let reshape47_out1 = linear93_out1.reshape(concat4_out1);
        let slice162_out1 = reshape47_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze72_out1 = slice162_out1.squeeze_dims::<4>(&[-3]);
        let slice163_out1 = reshape47_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze73_out1 = slice163_out1.squeeze_dims::<4>(&[-3]);
        let slice164_out1 = reshape47_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze74_out1 = slice164_out1.squeeze_dims::<4>(&[-3]);
        let transpose116_out1 = squeeze72_out1.permute([0, 2, 1, 3]);
        let transpose117_out1 = squeeze73_out1.permute([0, 2, 1, 3]);
        let transpose118_out1 = squeeze74_out1.permute([0, 2, 1, 3]);
        let mul141_out1 = transpose116_out1.clone().mul(unsqueeze12_out1.clone());
        let slice165_out1 = transpose116_out1.clone().slice(s![.., .., .., 0..32]);
        let slice166_out1 = transpose116_out1.slice(s![.., .., .., 32..]);
        let neg47_out1 = slice166_out1.neg();
        let concat52_out1 = burn::tensor::Tensor::cat(
            [neg47_out1, slice165_out1].into(),
            3,
        );
        let mul142_out1 = concat52_out1.mul(unsqueeze13_out1.clone());
        let add116_out1 = mul141_out1.add(mul142_out1);
        let mul143_out1 = transpose117_out1.clone().mul(unsqueeze12_out1.clone());
        let slice167_out1 = transpose117_out1.clone().slice(s![.., .., .., 0..32]);
        let slice168_out1 = transpose117_out1.slice(s![.., .., .., 32..]);
        let neg48_out1 = slice168_out1.neg();
        let concat53_out1 = burn::tensor::Tensor::cat(
            [neg48_out1, slice167_out1].into(),
            3,
        );
        let mul144_out1 = concat53_out1.mul(unsqueeze13_out1.clone());
        let add117_out1 = mul143_out1.add(mul144_out1);
        let matmul141_out1 = crate::inference::attention::forward(
            add116_out1,
            add117_out1,
            transpose118_out1,
            Some(where2_out1.clone()),
            Some(0.125f64),
        );
        let transpose120_out1 = matmul141_out1.permute([0, 2, 1, 3]);
        let reshape48_out1 = transpose120_out1.reshape(concat7_out1);
        let linear94_out1 = self.linear94.forward(reshape48_out1);
        let add119_out1 = add115_out1.add(linear94_out1);
        let layernormalization48_out1 = {
            let dtype = add119_out1.clone().dtype();
            self.layernormalization48
                .forward(add119_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear95_out1 = self.linear95.forward(layernormalization48_out1);
        let split_tensors = linear95_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split24_out1, split24_out2] = split_tensors.try_into().unwrap();
        let gelu24_out1 = burn::tensor::activation::gelu(split24_out1);
        let mul146_out1 = gelu24_out1.mul(split24_out2);
        let linear96_out1 = self.linear96.forward(mul146_out1);
        let add120_out1 = add119_out1.add(linear96_out1);
        let layernormalization49_out1 = {
            let dtype = add120_out1.clone().dtype();
            self.layernormalization49
                .forward(add120_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear97_out1 = self.linear97.forward(layernormalization49_out1);
        let reshape49_out1 = linear97_out1.reshape(concat4_out1);
        let slice169_out1 = reshape49_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze75_out1 = slice169_out1.squeeze_dims::<4>(&[-3]);
        let slice170_out1 = reshape49_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze76_out1 = slice170_out1.squeeze_dims::<4>(&[-3]);
        let slice171_out1 = reshape49_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze77_out1 = slice171_out1.squeeze_dims::<4>(&[-3]);
        let transpose121_out1 = squeeze75_out1.permute([0, 2, 1, 3]);
        let transpose122_out1 = squeeze76_out1.permute([0, 2, 1, 3]);
        let transpose123_out1 = squeeze77_out1.permute([0, 2, 1, 3]);
        let mul147_out1 = transpose121_out1.clone().mul(unsqueeze10_out1.clone());
        let slice172_out1 = transpose121_out1.clone().slice(s![.., .., .., 0..32]);
        let slice173_out1 = transpose121_out1.slice(s![.., .., .., 32..]);
        let neg49_out1 = slice173_out1.neg();
        let concat54_out1 = burn::tensor::Tensor::cat(
            [neg49_out1, slice172_out1].into(),
            3,
        );
        let mul148_out1 = concat54_out1.mul(unsqueeze11_out1.clone());
        let add121_out1 = mul147_out1.add(mul148_out1);
        let mul149_out1 = transpose122_out1.clone().mul(unsqueeze10_out1);
        let slice174_out1 = transpose122_out1.clone().slice(s![.., .., .., 0..32]);
        let slice175_out1 = transpose122_out1.slice(s![.., .., .., 32..]);
        let neg50_out1 = slice175_out1.neg();
        let concat55_out1 = burn::tensor::Tensor::cat(
            [neg50_out1, slice174_out1].into(),
            3,
        );
        let mul150_out1 = concat55_out1.mul(unsqueeze11_out1);
        let add122_out1 = mul149_out1.add(mul150_out1);
        let matmul147_out1 = crate::inference::attention::forward(
            add121_out1,
            add122_out1,
            transpose123_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose125_out1 = matmul147_out1.permute([0, 2, 1, 3]);
        let reshape50_out1 = transpose125_out1.reshape(concat7_out1);
        let linear98_out1 = self.linear98.forward(reshape50_out1);
        let add124_out1 = add120_out1.add(linear98_out1);
        let layernormalization50_out1 = {
            let dtype = add124_out1.clone().dtype();
            self.layernormalization50
                .forward(add124_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear99_out1 = self.linear99.forward(layernormalization50_out1);
        let split_tensors = linear99_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split25_out1, split25_out2] = split_tensors.try_into().unwrap();
        let gelu25_out1 = burn::tensor::activation::gelu(split25_out1);
        let mul152_out1 = gelu25_out1.mul(split25_out2);
        let linear100_out1 = self.linear100.forward(mul152_out1);
        let add125_out1 = add124_out1.add(linear100_out1);
        let layernormalization51_out1 = {
            let dtype = add125_out1.clone().dtype();
            self.layernormalization51
                .forward(add125_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear101_out1 = self.linear101.forward(layernormalization51_out1);
        let reshape51_out1 = linear101_out1.reshape(concat4_out1);
        let slice176_out1 = reshape51_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze78_out1 = slice176_out1.squeeze_dims::<4>(&[-3]);
        let slice177_out1 = reshape51_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze79_out1 = slice177_out1.squeeze_dims::<4>(&[-3]);
        let slice178_out1 = reshape51_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze80_out1 = slice178_out1.squeeze_dims::<4>(&[-3]);
        let transpose126_out1 = squeeze78_out1.permute([0, 2, 1, 3]);
        let transpose127_out1 = squeeze79_out1.permute([0, 2, 1, 3]);
        let transpose128_out1 = squeeze80_out1.permute([0, 2, 1, 3]);
        let mul153_out1 = transpose126_out1.clone().mul(unsqueeze12_out1.clone());
        let slice179_out1 = transpose126_out1.clone().slice(s![.., .., .., 0..32]);
        let slice180_out1 = transpose126_out1.slice(s![.., .., .., 32..]);
        let neg51_out1 = slice180_out1.neg();
        let concat56_out1 = burn::tensor::Tensor::cat(
            [neg51_out1, slice179_out1].into(),
            3,
        );
        let mul154_out1 = concat56_out1.mul(unsqueeze13_out1.clone());
        let add126_out1 = mul153_out1.add(mul154_out1);
        let mul155_out1 = transpose127_out1.clone().mul(unsqueeze12_out1.clone());
        let slice181_out1 = transpose127_out1.clone().slice(s![.., .., .., 0..32]);
        let slice182_out1 = transpose127_out1.slice(s![.., .., .., 32..]);
        let neg52_out1 = slice182_out1.neg();
        let concat57_out1 = burn::tensor::Tensor::cat(
            [neg52_out1, slice181_out1].into(),
            3,
        );
        let mul156_out1 = concat57_out1.mul(unsqueeze13_out1.clone());
        let add127_out1 = mul155_out1.add(mul156_out1);
        let matmul153_out1 = crate::inference::attention::forward(
            add126_out1,
            add127_out1,
            transpose128_out1,
            Some(where2_out1.clone()),
            Some(0.125f64),
        );
        let transpose130_out1 = matmul153_out1.permute([0, 2, 1, 3]);
        let reshape52_out1 = transpose130_out1.reshape(concat7_out1);
        let linear102_out1 = self.linear102.forward(reshape52_out1);
        let add129_out1 = add125_out1.add(linear102_out1);
        let layernormalization52_out1 = {
            let dtype = add129_out1.clone().dtype();
            self.layernormalization52
                .forward(add129_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear103_out1 = self.linear103.forward(layernormalization52_out1);
        let split_tensors = linear103_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split26_out1, split26_out2] = split_tensors.try_into().unwrap();
        let gelu26_out1 = burn::tensor::activation::gelu(split26_out1);
        let mul158_out1 = gelu26_out1.mul(split26_out2);
        let linear104_out1 = self.linear104.forward(mul158_out1);
        let add130_out1 = add129_out1.add(linear104_out1);
        let layernormalization53_out1 = {
            let dtype = add130_out1.clone().dtype();
            self.layernormalization53
                .forward(add130_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear105_out1 = self.linear105.forward(layernormalization53_out1);
        let reshape53_out1 = linear105_out1.reshape(concat4_out1);
        let slice183_out1 = reshape53_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze81_out1 = slice183_out1.squeeze_dims::<4>(&[-3]);
        let slice184_out1 = reshape53_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze82_out1 = slice184_out1.squeeze_dims::<4>(&[-3]);
        let slice185_out1 = reshape53_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze83_out1 = slice185_out1.squeeze_dims::<4>(&[-3]);
        let transpose131_out1 = squeeze81_out1.permute([0, 2, 1, 3]);
        let transpose132_out1 = squeeze82_out1.permute([0, 2, 1, 3]);
        let transpose133_out1 = squeeze83_out1.permute([0, 2, 1, 3]);
        let mul159_out1 = transpose131_out1.clone().mul(unsqueeze12_out1.clone());
        let slice186_out1 = transpose131_out1.clone().slice(s![.., .., .., 0..32]);
        let slice187_out1 = transpose131_out1.slice(s![.., .., .., 32..]);
        let neg53_out1 = slice187_out1.neg();
        let concat58_out1 = burn::tensor::Tensor::cat(
            [neg53_out1, slice186_out1].into(),
            3,
        );
        let mul160_out1 = concat58_out1.mul(unsqueeze13_out1.clone());
        let add131_out1 = mul159_out1.add(mul160_out1);
        let mul161_out1 = transpose132_out1.clone().mul(unsqueeze12_out1);
        let slice188_out1 = transpose132_out1.clone().slice(s![.., .., .., 0..32]);
        let slice189_out1 = transpose132_out1.slice(s![.., .., .., 32..]);
        let neg54_out1 = slice189_out1.neg();
        let concat59_out1 = burn::tensor::Tensor::cat(
            [neg54_out1, slice188_out1].into(),
            3,
        );
        let mul162_out1 = concat59_out1.mul(unsqueeze13_out1);
        let add132_out1 = mul161_out1.add(mul162_out1);
        let matmul159_out1 = crate::inference::attention::forward(
            add131_out1,
            add132_out1,
            transpose133_out1,
            Some(where2_out1),
            Some(0.125f64),
        );
        let transpose135_out1 = matmul159_out1.permute([0, 2, 1, 3]);
        let reshape54_out1 = transpose135_out1.reshape(concat7_out1);
        let linear106_out1 = self.linear106.forward(reshape54_out1);
        let add134_out1 = add130_out1.add(linear106_out1);
        add134_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule13 {
    layernormalization54: LayerNorm,
    linear107: Linear,
    linear108: Linear,
    layernormalization55: LayerNorm,
    linear109: Linear,
    linear110: Linear,
    layernormalization56: LayerNorm,
    linear111: Linear,
    linear112: Linear,
    layernormalization57: LayerNorm,
    constant77: burn::module::Param<Tensor<2>>,
    layernormalization58: LayerNorm,
    constant59: burn::module::Param<Tensor<2>>,
    constant60: burn::module::Param<Tensor<1>>,
    linear113: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule13 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization54 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear107 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear108 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization55 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear109 = LinearConfig::new(1024, 3072).with_bias(false).init(device);
        let linear110 = LinearConfig::new(1024, 1024).with_bias(false).init(device);
        let layernormalization56 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let linear111 = LinearConfig::new(1024, 5248).with_bias(false).init(device);
        let linear112 = LinearConfig::new(2624, 1024).with_bias(false).init(device);
        let layernormalization57 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(false)
            .init(device);
        let constant77: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([3, 1024], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [3, 1024].into(),
        );
        let layernormalization58 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant59: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([3072, 1024], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [3072, 1024].into(),
        );
        let constant60: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([3072], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [3072].into(),
        );
        let linear113 = LinearConfig::new(1024, 1024).with_bias(true).init(device);
        Self {
            layernormalization54,
            linear107,
            linear108,
            layernormalization55,
            linear109,
            linear110,
            layernormalization56,
            linear111,
            linear112,
            layernormalization57,
            constant77,
            layernormalization58,
            constant59,
            constant60,
            linear113,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add134_out1: Tensor<3>,
        concat4_out1: [i64; 5],
        unsqueeze10_out1: Tensor<4>,
        unsqueeze11_out1: Tensor<4>,
        where1_out1: Tensor<4>,
        concat7_out1: [i64; 3],
        qtype: Tensor<1, Int>,
        cast1_out1: Tensor<2, Bool>,
        constant91_out1: f32,
        shape1_out1: [i64; 1],
        shape2_out1: [i64; 1],
        constant228_out1: [i64; 1],
    ) -> (Tensor<3>, [i64; 4], [i64; 1], Tensor<4, Bool>, [i64; 3]) {
        let layernormalization54_out1 = {
            let dtype = add134_out1.clone().dtype();
            self.layernormalization54
                .forward(add134_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear107_out1 = self.linear107.forward(layernormalization54_out1);
        let split_tensors = linear107_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split27_out1, split27_out2] = split_tensors.try_into().unwrap();
        let gelu27_out1 = burn::tensor::activation::gelu(split27_out1);
        let mul164_out1 = gelu27_out1.mul(split27_out2);
        let linear108_out1 = self.linear108.forward(mul164_out1);
        let add135_out1 = add134_out1.add(linear108_out1);
        let layernormalization55_out1 = {
            let dtype = add135_out1.clone().dtype();
            self.layernormalization55
                .forward(add135_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear109_out1 = self.linear109.forward(layernormalization55_out1);
        let reshape55_out1 = linear109_out1.reshape(concat4_out1);
        let slice190_out1 = reshape55_out1.clone().slice(s![.., .., 0..1, .., ..]);
        let squeeze84_out1 = slice190_out1.squeeze_dims::<4>(&[-3]);
        let slice191_out1 = reshape55_out1.clone().slice(s![.., .., 1..2, .., ..]);
        let squeeze85_out1 = slice191_out1.squeeze_dims::<4>(&[-3]);
        let slice192_out1 = reshape55_out1.slice(s![.., .., 2..3, .., ..]);
        let squeeze86_out1 = slice192_out1.squeeze_dims::<4>(&[-3]);
        let transpose136_out1 = squeeze84_out1.permute([0, 2, 1, 3]);
        let transpose137_out1 = squeeze85_out1.permute([0, 2, 1, 3]);
        let transpose138_out1 = squeeze86_out1.permute([0, 2, 1, 3]);
        let mul165_out1 = transpose136_out1.clone().mul(unsqueeze10_out1.clone());
        let slice193_out1 = transpose136_out1.clone().slice(s![.., .., .., 0..32]);
        let slice194_out1 = transpose136_out1.slice(s![.., .., .., 32..]);
        let neg55_out1 = slice194_out1.neg();
        let concat60_out1 = burn::tensor::Tensor::cat(
            [neg55_out1, slice193_out1].into(),
            3,
        );
        let mul166_out1 = concat60_out1.mul(unsqueeze11_out1.clone());
        let add136_out1 = mul165_out1.add(mul166_out1);
        let mul167_out1 = transpose137_out1.clone().mul(unsqueeze10_out1);
        let slice195_out1 = transpose137_out1.clone().slice(s![.., .., .., 0..32]);
        let slice196_out1 = transpose137_out1.slice(s![.., .., .., 32..]);
        let neg56_out1 = slice196_out1.neg();
        let concat61_out1 = burn::tensor::Tensor::cat(
            [neg56_out1, slice195_out1].into(),
            3,
        );
        let mul168_out1 = concat61_out1.mul(unsqueeze11_out1);
        let add137_out1 = mul167_out1.add(mul168_out1);
        let matmul165_out1 = crate::inference::attention::forward(
            add136_out1,
            add137_out1,
            transpose138_out1,
            Some(where1_out1),
            Some(0.125f64),
        );
        let transpose140_out1 = matmul165_out1.permute([0, 2, 1, 3]);
        let reshape56_out1 = transpose140_out1.reshape(concat7_out1);
        let linear110_out1 = self.linear110.forward(reshape56_out1);
        let add139_out1 = add135_out1.add(linear110_out1);
        let layernormalization56_out1 = {
            let dtype = add139_out1.clone().dtype();
            self.layernormalization56
                .forward(add139_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear111_out1 = self.linear111.forward(layernormalization56_out1);
        let split_tensors = linear111_out1.split_with_sizes([2624, 2624].into(), 2);
        let [split28_out1, split28_out2] = split_tensors.try_into().unwrap();
        let gelu28_out1 = burn::tensor::activation::gelu(split28_out1);
        let mul170_out1 = gelu28_out1.mul(split28_out2);
        let linear112_out1 = self.linear112.forward(mul170_out1);
        let add140_out1 = add139_out1.add(linear112_out1);
        let layernormalization57_out1 = {
            let dtype = add140_out1.dtype();
            self.layernormalization57
                .forward(add140_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let constant77_out1 = self.constant77.val();
        let gather2_out1 = {
            let axis_size = constant77_out1.dims()[0] as i64;
            let negative = qtype.clone().lower_elem(0i64);
            let corrected = qtype.clone() + axis_size;
            let indices = qtype.mask_where(negative, corrected);
            constant77_out1.take::<1, 2>(0, indices)
        };
        let unsqueeze14_out1: Tensor<3> = gather2_out1.unsqueeze_dims::<3>(&[1]);
        let add141_out1 = layernormalization57_out1.add(unsqueeze14_out1);
        let not1_out1 = cast1_out1.bool_not();
        let shape4_out1: [i64; 2] = {
            let axes = &not1_out1.clone().dims()[0..2];
            let mut output = [0i64; 2];
            for i in 0..2 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let expand3_out1 = {
            let input = Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([constant91_out1]),
                (&self.device, burn::tensor::DType::F32),
            );
            input.expand(shape4_out1)
        };
        let constant208_out1 = f32::NEG_INFINITY;
        let where3_out1 = expand3_out1.mask_fill(not1_out1, constant208_out1);
        let layernormalization58_out1 = {
            let dtype = add141_out1.clone().dtype();
            self.layernormalization58
                .forward(add141_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose141_out1 = layernormalization58_out1.permute([1, 0, 2]);
        let constant59_out1 = self.constant59.val();
        let split_tensors = constant59_out1
            .split_with_sizes([1024, 1024, 1024].into(), 0);
        let [split29_out1, split29_out2, split29_out3] = split_tensors
            .try_into()
            .unwrap();
        let constant60_out1 = self.constant60.val();
        let split_tensors = constant60_out1
            .split_with_sizes([1024, 1024, 1024].into(), 0);
        let [split30_out1, split30_out2, split30_out3] = split_tensors
            .try_into()
            .unwrap();
        let transpose142_out1 = split29_out1.permute([1, 0]);
        let matmul169_out1 = transpose141_out1
            .clone()
            .matmul(transpose142_out1.unsqueeze::<3usize>());
        let add142_out1 = matmul169_out1
            .add((split30_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant231_out1: [i64; 1] = [16i64];
        let concat62_out1: [i64; 4usize] = [
            &shape1_out1[..],
            &shape2_out1[..],
            &constant231_out1[..],
            &constant228_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape57_out1 = add142_out1.reshape(concat62_out1);
        let transpose143_out1 = reshape57_out1.permute([1, 2, 0, 3]);
        let transpose144_out1 = split29_out2.permute([1, 0]);
        let matmul170_out1 = transpose141_out1
            .clone()
            .matmul(transpose144_out1.unsqueeze::<3usize>());
        let add143_out1 = matmul170_out1
            .add((split30_out2).unsqueeze_dims(&[0isize, 1isize]));
        let reshape58_out1 = add143_out1.reshape(concat62_out1);
        let transpose145_out1 = reshape58_out1.permute([1, 2, 0, 3]);
        let transpose146_out1 = split29_out3.permute([1, 0]);
        let matmul171_out1 = transpose141_out1
            .matmul(transpose146_out1.unsqueeze::<3usize>());
        let add144_out1 = matmul171_out1
            .add((split30_out3).unsqueeze_dims(&[0isize, 1isize]));
        let reshape59_out1 = add144_out1.reshape(concat62_out1);
        let transpose147_out1 = reshape59_out1.permute([1, 2, 0, 3]);
        let unsqueeze15_out1: Tensor<4> = where3_out1.unsqueeze_dims::<4>(&[1, 2]);
        let equal1_out1 = unsqueeze15_out1.equal_elem(constant91_out1);
        let shape5_out1: [i64; 1] = {
            let axes = &transpose143_out1.clone().dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let shape6_out1: [i64; 1] = {
            let axes = &transpose143_out1.clone().dims()[2..3];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let shape7_out1: [i64; 1] = {
            let axes = &transpose145_out1.clone().dims()[2..3];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant221_out1: [i64; 1] = [1i64];
        let concat63_out1: [i64; 4usize] = [
            &shape5_out1[..],
            &constant221_out1[..],
            &shape6_out1[..],
            &shape7_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let expand4_out1 = {
            let onnx_shape: [i64; 4usize] = concat63_out1;
            let input_dims = equal1_out1.clone().dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            equal1_out1.clone().expand(shape)
        };
        let attention1_out1 = crate::inference::attention::forward(
            transpose143_out1,
            transpose145_out1,
            transpose147_out1,
            {
                let allowed = expand4_out1;
                Some(
                    allowed
                        .clone()
                        .float()
                        .mul_scalar(0.0)
                        .mask_fill(allowed.bool_not(), f32::NEG_INFINITY),
                )
            },
            None,
        );
        let transpose148_out1 = attention1_out1.permute([2, 0, 1, 3]);
        let constant232_out1: [i64; 1] = [1024i64];
        let concat64_out1: [i64; 3usize] = [
            &shape1_out1[..],
            &shape2_out1[..],
            &constant232_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape60_out1 = transpose148_out1.reshape(concat64_out1);
        let transpose149_out1 = reshape60_out1.permute([1, 0, 2]);
        let linear113_out1 = self.linear113.forward(transpose149_out1);
        let add145_out1 = add141_out1.add(linear113_out1);
        (add145_out1, concat62_out1, constant221_out1, equal1_out1, concat64_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule14 {
    layernormalization59: LayerNorm,
    linear114: Linear,
    linear115: Linear,
    layernormalization60: LayerNorm,
    constant68: burn::module::Param<Tensor<2>>,
    constant69: burn::module::Param<Tensor<1>>,
    linear116: Linear,
    layernormalization61: LayerNorm,
    linear117: Linear,
    linear118: Linear,
    layernormalization62: LayerNorm,
    linear119: Linear,
    linear120: Linear,
    constant233: burn::module::Param<Tensor<1>>,
    linear121: Linear,
    linear122: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule14 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization59 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear114 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear115 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization60 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant68: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([3072, 1024], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [3072, 1024].into(),
        );
        let constant69: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([3072], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [3072].into(),
        );
        let linear116 = LinearConfig::new(1024, 1024).with_bias(true).init(device);
        let layernormalization61 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear117 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear118 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization62 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear119 = LinearConfig::new(1024, 1024).with_bias(true).init(device);
        let linear120 = LinearConfig::new(1024, 1).with_bias(true).init(device);
        let constant233: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([255f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear121 = LinearConfig::new(1028, 256).with_bias(true).init(device);
        let linear122 = LinearConfig::new(256, 2).with_bias(true).init(device);
        Self {
            layernormalization59,
            linear114,
            linear115,
            layernormalization60,
            constant68,
            constant69,
            linear116,
            layernormalization61,
            linear117,
            linear118,
            layernormalization62,
            linear119,
            linear120,
            constant233,
            linear121,
            linear122,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add145_out1: Tensor<3>,
        concat62_out1: [i64; 4],
        constant221_out1: [i64; 1],
        equal1_out1: Tensor<4, Bool>,
        concat64_out1: [i64; 3],
        marker_pos: Tensor<2, Int>,
        marker_mask: Tensor<2, Bool>,
    ) -> (Tensor<2>, Tensor<2>) {
        let layernormalization59_out1 = {
            let dtype = add145_out1.clone().dtype();
            self.layernormalization59
                .forward(add145_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear114_out1 = self.linear114.forward(layernormalization59_out1);
        let relu1_out1 = burn::tensor::activation::relu(linear114_out1);
        let linear115_out1 = self.linear115.forward(relu1_out1);
        let add146_out1 = add145_out1.add(linear115_out1);
        let layernormalization60_out1 = {
            let dtype = add146_out1.clone().dtype();
            self.layernormalization60
                .forward(add146_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose150_out1 = layernormalization60_out1.permute([1, 0, 2]);
        let constant68_out1 = self.constant68.val();
        let split_tensors = constant68_out1
            .split_with_sizes([1024, 1024, 1024].into(), 0);
        let [split31_out1, split31_out2, split31_out3] = split_tensors
            .try_into()
            .unwrap();
        let constant69_out1 = self.constant69.val();
        let split_tensors = constant69_out1
            .split_with_sizes([1024, 1024, 1024].into(), 0);
        let [split32_out1, split32_out2, split32_out3] = split_tensors
            .try_into()
            .unwrap();
        let transpose151_out1 = split31_out1.permute([1, 0]);
        let matmul175_out1 = transpose150_out1
            .clone()
            .matmul(transpose151_out1.unsqueeze::<3usize>());
        let add147_out1 = matmul175_out1
            .add((split32_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reshape61_out1 = add147_out1.reshape(concat62_out1);
        let transpose152_out1 = reshape61_out1.permute([1, 2, 0, 3]);
        let transpose153_out1 = split31_out2.permute([1, 0]);
        let matmul176_out1 = transpose150_out1
            .clone()
            .matmul(transpose153_out1.unsqueeze::<3usize>());
        let add148_out1 = matmul176_out1
            .add((split32_out2).unsqueeze_dims(&[0isize, 1isize]));
        let reshape62_out1 = add148_out1.reshape(concat62_out1);
        let transpose154_out1 = reshape62_out1.permute([1, 2, 0, 3]);
        let transpose155_out1 = split31_out3.permute([1, 0]);
        let matmul177_out1 = transpose150_out1
            .matmul(transpose155_out1.unsqueeze::<3usize>());
        let add149_out1 = matmul177_out1
            .add((split32_out3).unsqueeze_dims(&[0isize, 1isize]));
        let reshape63_out1 = add149_out1.reshape(concat62_out1);
        let transpose156_out1 = reshape63_out1.permute([1, 2, 0, 3]);
        let shape8_out1: [i64; 1] = {
            let axes = &transpose152_out1.clone().dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let shape9_out1: [i64; 1] = {
            let axes = &transpose152_out1.clone().dims()[2..3];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let shape10_out1: [i64; 1] = {
            let axes = &transpose154_out1.clone().dims()[2..3];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let concat65_out1: [i64; 4usize] = [
            &shape8_out1[..],
            &constant221_out1[..],
            &shape9_out1[..],
            &shape10_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let expand5_out1 = {
            let onnx_shape: [i64; 4usize] = concat65_out1;
            let input_dims = equal1_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            equal1_out1.expand(shape)
        };
        let attention2_out1 = crate::inference::attention::forward(
            transpose152_out1,
            transpose154_out1,
            transpose156_out1,
            {
                let allowed = expand5_out1;
                Some(
                    allowed
                        .clone()
                        .float()
                        .mul_scalar(0.0)
                        .mask_fill(allowed.bool_not(), f32::NEG_INFINITY),
                )
            },
            None,
        );
        let transpose157_out1 = attention2_out1.permute([2, 0, 1, 3]);
        let reshape64_out1 = transpose157_out1.reshape(concat64_out1);
        let transpose158_out1 = reshape64_out1.permute([1, 0, 2]);
        let linear116_out1 = self.linear116.forward(transpose158_out1);
        let add150_out1 = add146_out1.add(linear116_out1);
        let layernormalization61_out1 = {
            let dtype = add150_out1.clone().dtype();
            self.layernormalization61
                .forward(add150_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear117_out1 = self.linear117.forward(layernormalization61_out1);
        let relu2_out1 = burn::tensor::activation::relu(linear117_out1);
        let linear118_out1 = self.linear118.forward(relu2_out1);
        let add151_out1 = add150_out1.add(linear118_out1);
        let clip1_out1 = marker_pos.clamp_min(0f64);
        let unsqueeze16_out1: Tensor<3, Int> = clip1_out1.unsqueeze_dims::<3>(&[2]);
        let expand6_out1 = {
            let onnx_shape: [i64; 3usize] = [1, 1, 1024];
            let input_dims = unsqueeze16_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..3usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze16_out1.expand(shape)
        };
        let gatherelements1_out1 = {
            let axis_size = add151_out1.clone().dims()[1] as i64;
            let negative = expand6_out1.clone().lower_elem(0i64);
            let corrected = expand6_out1.clone() + axis_size;
            let indices = expand6_out1.mask_where(negative, corrected);
            add151_out1.clone().gather(1, indices)
        };
        let layernormalization62_out1 = {
            let dtype = gatherelements1_out1.dtype();
            self.layernormalization62
                .forward(gatherelements1_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear119_out1 = self.linear119.forward(layernormalization62_out1);
        let gelu29_out1 = burn::tensor::activation::gelu(linear119_out1);
        let linear120_out1 = self.linear120.forward(gelu29_out1);
        let squeeze87_out1 = linear120_out1.squeeze_dims::<2>(&[-1]);
        let not2_out1 = marker_mask.clone().bool_not();
        let constant218_out1 = -10000f32;
        let where4_out1 = squeeze87_out1.mask_fill(not2_out1, constant218_out1);
        let softmax29_out1 = burn::tensor::activation::softmax(where4_out1.clone(), 1);
        let cast3_out1 = marker_mask.int().cast(burn::tensor::DType::I64);
        let reducesum1_out1 = {
            cast3_out1.sum_dim(1usize).squeeze_dims::<1usize>(&[1])
        };
        let clip2_out1 = reducesum1_out1.clamp_min(2f64);
        let cast4_out1 = clip2_out1.float().cast(burn::tensor::DType::F32);
        let clip3_out1 = softmax29_out1
            .clone()
            .clamp_min(0.0000000009999999717180685f64);
        let log1_out1 = clip3_out1.log();
        let mul171_out1 = softmax29_out1.clone().mul(log1_out1);
        let reducesum2_out1 = {
            mul171_out1.sum_dim(1usize).squeeze_dims::<1usize>(&[1])
        };
        let neg57_out1 = reducesum2_out1.neg();
        let log2_out1 = cast4_out1.clone().log();
        let div1_out1 = neg57_out1.div(log2_out1);
        let (topk1_out1, topk1_out2) = {
            let (values, indices) = softmax29_out1.topk_with_indices(2, 1);
            (values, indices.cast(burn::tensor::DType::I64))
        };
        let gather3_out1 = {
            let sliced = topk1_out1.clone().slice(s![.., 0]);
            sliced.squeeze_dim::<1usize>(1)
        };
        let gather4_out1 = {
            let sliced = topk1_out1.slice(s![.., 1]);
            sliced.squeeze_dim::<1usize>(1)
        };
        let sub2_out1 = gather3_out1.clone().sub(gather4_out1);
        let constant233_out1 = self.constant233.val();
        let div2_out1 = cast4_out1.div(constant233_out1);
        let unsqueeze17_out1: Tensor<2> = gather3_out1.unsqueeze_dims::<2>(&[-1]);
        let unsqueeze18_out1: Tensor<2> = sub2_out1.unsqueeze_dims::<2>(&[-1]);
        let unsqueeze19_out1: Tensor<2> = div1_out1.unsqueeze_dims::<2>(&[-1]);
        let unsqueeze20_out1: Tensor<2> = div2_out1.unsqueeze_dims::<2>(&[-1]);
        let concat66_out1 = burn::tensor::Tensor::cat(
            [unsqueeze17_out1, unsqueeze18_out1, unsqueeze19_out1, unsqueeze20_out1]
                .into(),
            1,
        );
        let gather5_out1 = {
            let sliced = add151_out1.slice(s![.., 0, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let concat67_out1 = burn::tensor::Tensor::cat(
            [gather5_out1, concat66_out1].into(),
            1,
        );
        let linear121_out1 = self.linear121.forward(concat67_out1);
        let gelu30_out1 = burn::tensor::activation::gelu(linear121_out1);
        let linear122_out1 = self.linear122.forward(gelu30_out1);
        (where4_out1, linear122_out1)
    }
}

#[derive(Module, Debug)]
pub struct Model {
    submodule1: Submodule1,
    submodule2: Submodule2,
    submodule3: Submodule3,
    submodule4: Submodule4,
    submodule5: Submodule5,
    submodule6: Submodule6,
    submodule7: Submodule7,
    submodule8: Submodule8,
    submodule9: Submodule9,
    submodule10: Submodule10,
    submodule11: Submodule11,
    submodule12: Submodule12,
    submodule13: Submodule13,
    submodule14: Submodule14,
    #[module(skip)]
    device: Device,
}


extern crate std;

impl Default for Model {
    fn default() -> Self {
        Self::from_file(
            "laya.bpk",
            &Default::default(),
        )
    }
}

impl Model {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(&file);
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| {
                panic!("Failed to load burnpack file {}: {e}", file.as_ref().display())
            });
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| panic!("Failed to load burnpack bytes: {e}"));
        model
    }
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        let submodule5 = Submodule5::new(device);
        let submodule6 = Submodule6::new(device);
        let submodule7 = Submodule7::new(device);
        let submodule8 = Submodule8::new(device);
        let submodule9 = Submodule9::new(device);
        let submodule10 = Submodule10::new(device);
        let submodule11 = Submodule11::new(device);
        let submodule12 = Submodule12::new(device);
        let submodule13 = Submodule13::new(device);
        let submodule14 = Submodule14::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            submodule6,
            submodule7,
            submodule8,
            submodule9,
            submodule10,
            submodule11,
            submodule12,
            submodule13,
            submodule14,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Int>,
        marker_pos: Tensor<2, Int>,
        marker_mask: Tensor<2, Bool>,
        qtype: Tensor<1, Int>,
    ) -> (Tensor<2>, Tensor<2>) {
        let (
            add4_out1,
            concat4_out1,
            cos1_out1,
            sin1_out1,
            where2_out1,
            concat7_out1,
            unsqueeze10_out1,
            unsqueeze11_out1,
            where1_out1,
            cast1_out1,
            constant91_out1,
            shape1_out1,
            shape2_out1,
            constant228_out1,
        ) = self.submodule1.forward(attention_mask, qtype.clone(), input_ids);
        let (add14_out1, unsqueeze12_out1, unsqueeze13_out1) = self
            .submodule2
            .forward(
                add4_out1,
                concat4_out1.clone(),
                cos1_out1,
                sin1_out1,
                where2_out1.clone(),
                concat7_out1.clone(),
            );
        let add24_out1 = self
            .submodule3
            .forward(
                add14_out1,
                concat4_out1.clone(),
                unsqueeze10_out1.clone(),
                unsqueeze11_out1.clone(),
                where1_out1.clone(),
                concat7_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
            );
        let add34_out1 = self
            .submodule4
            .forward(
                add24_out1,
                concat4_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
                concat7_out1.clone(),
                unsqueeze10_out1.clone(),
                unsqueeze11_out1.clone(),
                where1_out1.clone(),
            );
        let add44_out1 = self
            .submodule5
            .forward(
                add34_out1,
                concat4_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
                concat7_out1.clone(),
            );
        let add54_out1 = self
            .submodule6
            .forward(
                add44_out1,
                concat4_out1.clone(),
                unsqueeze10_out1.clone(),
                unsqueeze11_out1.clone(),
                where1_out1.clone(),
                concat7_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
            );
        let add64_out1 = self
            .submodule7
            .forward(
                add54_out1,
                concat4_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
                concat7_out1.clone(),
                unsqueeze10_out1.clone(),
                unsqueeze11_out1.clone(),
                where1_out1.clone(),
            );
        let add74_out1 = self
            .submodule8
            .forward(
                add64_out1,
                concat4_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
                concat7_out1.clone(),
            );
        let add84_out1 = self
            .submodule9
            .forward(
                add74_out1,
                concat4_out1.clone(),
                unsqueeze10_out1.clone(),
                unsqueeze11_out1.clone(),
                where1_out1.clone(),
                concat7_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
            );
        let add94_out1 = self
            .submodule10
            .forward(
                add84_out1,
                concat4_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
                concat7_out1.clone(),
                unsqueeze10_out1.clone(),
                unsqueeze11_out1.clone(),
                where1_out1.clone(),
            );
        let add104_out1 = self
            .submodule11
            .forward(
                add94_out1,
                concat4_out1.clone(),
                unsqueeze12_out1.clone(),
                unsqueeze13_out1.clone(),
                where2_out1.clone(),
                concat7_out1.clone(),
            );
        let add134_out1 = self
            .submodule12
            .forward(
                add104_out1,
                concat4_out1.clone(),
                unsqueeze10_out1.clone(),
                unsqueeze11_out1.clone(),
                where1_out1.clone(),
                concat7_out1.clone(),
                unsqueeze12_out1,
                unsqueeze13_out1,
                where2_out1,
            );
        let (add145_out1, concat62_out1, constant221_out1, equal1_out1, concat64_out1) = self
            .submodule13
            .forward(
                add134_out1,
                concat4_out1,
                unsqueeze10_out1,
                unsqueeze11_out1,
                where1_out1,
                concat7_out1,
                qtype,
                cast1_out1,
                constant91_out1,
                shape1_out1,
                shape2_out1,
                constant228_out1,
            );
        let (where4_out1, linear122_out1) = self
            .submodule14
            .forward(
                add145_out1,
                concat62_out1,
                constant221_out1,
                equal1_out1,
                concat64_out1,
                marker_pos,
                marker_mask,
            );
        (where4_out1, linear122_out1)
    }
}
