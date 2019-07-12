#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

#define IMG_W 2
#define IMG_H 2
#define FIL_W 2
#define FIL_H 2

class CustomConv2dOp : public OpKernel {
    public:
    explicit CustomConv2dOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
    // Grab the input tensor
        const Tensor& input_tensor1 = context->input(0);
        const Tensor& input_tensor2 = context->input(1);
        
        tensorflow::TensorShape ts(input_tensor1.shape());
        int width = (input_tensor1.shape().dim_size(1)-input_tensor2.shape().dim_size(1)+1);
        int height = (input_tensor1.shape().dim_size(0)-input_tensor2.shape().dim_size(0)+1);
        ts.set_dim(0, height);
        ts.set_dim(1, width);

        auto input = input_tensor1.flat<float>();
        float partial_conv;
        float conv_res[height][width];
        int filter[FIL_H][FIL_W] = {{1, 0}, {1, -1}};
        float img[IMG_H][IMG_W] = {{1, 0}, {1, -1}};

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, ts,&output_tensor));
        auto output_flat = output_tensor->flat<float>();
        
        // convolution body
        
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++){
                partial_conv = 0;
                for (int k = 0; k < FIL_H; k++)
                    for (int l = 0; l < FIL_W; l++)
                        partial_conv += filter[k][l] * img[i+k][l+j];
                
                conv_res[i][j] = partial_conv;
            }
        for (int i = 0; i <(height) * (width) ; i++) {
            output_flat(i) = conv_res[i/(height)][i%(width)];
        }

    }
};

REGISTER_KERNEL_BUILDER(Name("CustomConv2d").Device(DEVICE_CPU), CustomConv2dOp);