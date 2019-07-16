#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class CustomConv2dOp : public OpKernel {
    public:
    explicit CustomConv2dOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {

        const Tensor& input_tensor1 = context->input(0);
        const Tensor& input_tensor2 = context->input(1);

        const Tensor& input_tensor3 = context->input(2);
        auto strides = input_tensor3.flat<float>();
        
        int FIL_H = input_tensor2.shape().dim_size(0);
        int FIL_W =input_tensor2.shape().dim_size(1);
        int IMG_H = input_tensor1.shape().dim_size(3);
        int IMG_W = input_tensor1.shape().dim_size(2);

        int width = ceil((input_tensor1.shape().dim_size(2)-input_tensor2.shape().dim_size(1)+1)/strides(1));
        int height = ceil((input_tensor1.shape().dim_size(1)-input_tensor2.shape().dim_size(0)+1)/strides(0));

        tensorflow::TensorShape ts(input_tensor1.shape());
        ts.set_dim(0, 1);
        ts.set_dim(1, width);
        ts.set_dim(2, height);
        ts.set_dim(3, 1);

        auto img = input_tensor1.flat<float>();
        auto filter = input_tensor2.flat<float>();
        float partial_conv;
        float conv_res[height][width];

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, ts,&output_tensor));
        auto output_flat = output_tensor->flat<float>();
        
        // convolution body
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++){
                partial_conv = 0;
                for (int k = 0; k < FIL_H; k+=1)
                    for (int l = 0; l < FIL_W; l+=1)
                        partial_conv += filter(k*FIL_W+l) * img((i*strides(0)+k)*IMG_W+l+j*strides(1));
                
                conv_res[i][j] = partial_conv;
            }
        for (int i = 0; i <(height) * (width) ; i++)
            output_flat(i) = conv_res[i/(width)][i%(width)];

    }
};

REGISTER_KERNEL_BUILDER(Name("CustomConv2d").Device(DEVICE_CPU), CustomConv2dOp);