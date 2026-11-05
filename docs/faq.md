# FAQs

## What's this?

HGQ is a method for quantization aware training of neural works to be deployed on FPGAs, which allows for per-weight and per-activation bitwidth optimization. HGQ2 is the second generation of a library implementing this method.

## Why is it useful?

Depending on the specific [application](https://arxiv.org/abs/2006.10159), HGQ could achieve up to 10x resource reduction compared to the traditional `AutoQkeras` approach, while maintaining the same accuracy. For some more challenging [tasks](https://arxiv.org/abs/2202.04976), where the model is already under-fitted, HGQ could still improve the performance under the same on-board resource consumption. For more details, please refer to our paper at [arXiv](https://arxiv.org/abs/2405.00645).

## Can I use it?

The primary usage for HGQ is to quantize your model for FPGA deployment. Currently the recommended way to use HGQ is to train your model with HGQ, and then convert the trained model to HLS C++ code with `hls4ml` or `da4ml` However, you can also use the quantized model for other purposes, such as deploying on other platforms, or even just for model compression.

For the training part, you can use HGQ as a drop-in replacement for `Keras` layers. For the deployment part, the following conditions should be met:

1. Your model is compatible with `da4ml` or `hls4ml` (i.e., no dynamic shapes, static dataflow, etc.)
   1. Almost all layers are supported by `da4ml`; most layers are supported by `hls4ml` but some may not be supported under certain configurations.
   2. If using `hls4ml`: You are using `Vitis` or `Vivado` or `OneAPI` (preliminary support) as your FPGA backend.
   - other backend **MAY** work if you don't use heterogeneous activation quantization.
2. Your model is representable in HGQ2 layers
   - Some layers in HGQ2 are not supported by `hls4ml` under certain configurations.
   - `da4ml`'s layer support is more limited than `hls4ml`, but supports more fine-grained operations (e.g., take arbitrary elements from a tensor, free-rearranging, etc.)

If you meet all the above conditions, you can probably use HGQ to quantize your model.

## What's the status of the project?

The project is still under development. The codebase and documentation are not stable yet, and we are working on it. However, the core functionality is already implemented and tested, and the APIs are considered semi-stable. If you encounter any issues, please feel free to contact us.

## I'm getting terrible results during/after training, what should I do?

Do you observe a collapse of loss value, as well as EBOPs and accuracy? If so, you may want to try the following:

1. Increase the initial bitwidth of the quantizer (`b`, `f`, and `i` (non-`WRAP` overflow mode) in the quantizer config).
2. Reduce `beta0` in quantizer config.
3. Reduce or remove bitwidth regularization in quantizer config.
4. If it still doesn't work, please try using the quantizer type `dummy` to see if the problem is caused by the quantizer. If not, it may be caused by the model itself. If the keras model works but not the HGQ model with dummy quantizer, please report an issue.

## `ERROR: [XFORM 203-504]` for Vivado HLS

```{warning}
Please consider switching to Vitis HLS, as Vivado HLS is considered deprecated. Vitis HLS does not have this limitation.
```

This is due to some hard-coded limitation in Vivado HLS, and there is no known way to disable this check. You can try to reduce the size of your layer, or try to use `parallel_factor` for convolutional layers, or split dense layers manually.

## QKeras?

Most `QKeras` models targeting `hls4ml` may be represented in HGQ2, but usually not vice versa. However, there are certain layers in `QKeras` that are not supported by HGQ2, such as the RNN family and hard activation layers (though can be emulated with a properly set fixed point quantizer with `SAT` or `SAT_SYM`).

HGQ2 comes with a preliminary `QKeras` compatibility layer in **alpha** quality and is not intended for production use. We have not decided whether to maintain those in the future, but they are available for testing and experimentation.

## Premission and License

This library, HGQ2, is a free software licensed under the LGPLv3 license. You can find the full license text in the `LICENSE` file in the root directory of the repository.

If you use this work in your research, we would appreciate a citation to our paper at [arXiv](https://arxiv.org/abs/2405.00645).
