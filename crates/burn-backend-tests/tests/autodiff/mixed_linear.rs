use super::*;
use burn_tensor::{DType, TensorData, module::mixed_linear};

// Regression guard for the LORA_BF16_BASE melt (confirmed in-app 2026-07-18):
// `mixed_linear` runs the base matmul in bf16 for speed, but its gradient MUST
// stay f32. If a bf16 tensor leaks into the autodiff graph, the residual-stream
// gradient is bf16-rounded and re-rounds once per layer, compounding into a
// training divergence. This test asserts (1) the input gradient dtype is f32 —
// the direct guard against that failure mode — and (2) a two-layer mixed_linear
// chain's input gradient matches a plain-f32 matmul reference within bf16
// tolerance (correctness of the custom backward). Weights are frozen (no
// require_grad), matching the LoRA base linear.
#[test]
fn mixed_linear_grad_stays_f32_and_matches_reference() {
    let device = AutodiffDevice::new();

    let input_data = TensorData::from([[0.5f32, -0.25, 1.0], [0.1, 0.3, -0.7]]); // [2, 3]
    let w1_data = TensorData::from([[0.2f32, -0.1], [0.4, 0.3], [-0.2, 0.5]]); // [3, 2]
    let w2_data = TensorData::from([[0.7f32, 0.1, -0.3], [0.2, -0.5, 0.4]]); // [2, 3]

    // bf16-compute path through the custom autodiff op.
    let input = TestTensor::<2>::from_data(input_data.clone(), &device).require_grad();
    let w1 = TestTensor::<2>::from_data(w1_data.clone(), &device);
    let w2 = TestTensor::<2>::from_data(w2_data.clone(), &device);
    let out = mixed_linear(mixed_linear(input.clone(), w1), w2);
    let grads = out.sum().backward();
    let grad = input.grad(&grads).unwrap();

    // (1) THE regression guard: gradient must be f32, never a bf16 graph tensor.
    assert_eq!(
        grad.dtype(),
        DType::F32,
        "mixed_linear leaked a non-f32 gradient ({:?}) — the melt-class bug",
        grad.dtype()
    );

    // (2) f32 matmul reference (bias-free linear == matmul).
    let input_ref = TestTensor::<2>::from_data(input_data, &device).require_grad();
    let w1_ref = TestTensor::<2>::from_data(w1_data, &device);
    let w2_ref = TestTensor::<2>::from_data(w2_data, &device);
    let out_ref = input_ref.clone().matmul(w1_ref).matmul(w2_ref);
    let grads_ref = out_ref.sum().backward();
    let grad_ref = input_ref.grad(&grads_ref).unwrap();

    let g = grad.to_data().to_vec::<f32>().unwrap();
    let gr = grad_ref.to_data().to_vec::<f32>().unwrap();
    let max_abs_err = g
        .iter()
        .zip(gr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_err < 5e-2,
        "mixed_linear input-grad vs f32 reference max_abs_err {max_abs_err} exceeds bf16 tolerance"
    );
}
