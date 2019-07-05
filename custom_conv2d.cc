#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class CustomConv2dOp : public OpKernel {
    public:
    explicit CustomConv2dOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
    // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<float32>();

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                        &output_tensor));
        auto output_flat = output_tensor->flat<float32>();

        // convolution body
        float partial_conv;
        float conv_res[IMG_H-FIL_H+1][IMG_W-FIL_W+1];
        int filter[FIL_H][FIL_W] = {{1, 0}, {1, -1}};
        float img[IMG_H][IMG_W] = {{1, 2.1, 3, 4.1, 0.5}, {1, 2, 3, 4, 5}, {1.2, 2, 3, 4, 5}, {1.3, 2, 3.1, 4, 5}, {1, 2.5, 3.8, 4.9, 5.7}};

        for (int i = 0; i < IMG_H-FIL_H+1; i++)
            for (int j = 0; j < IMG_W-FIL_W+1; j++){
                partial_conv = 0;
                for (int k = 0; k < FIL_H; k++)
                    for (int l = 0; l < FIL_W; l++)
                        partial_conv += filter[k][l] * img[i+k][l+j];
                
                conv_res[i][j] = partial_conv;
                cout<< "i = " << i << " j = "<< j << " res = " << partial_conv << endl;
            }
        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        for (int i = 1; i <(IMG_H-FIL_H+1) * (IMG_W-FIL_W+1) ; i++) {
            output_flat(i) = conv_res[i/IMG_H-FIL_H+1][i%IMG_W-FIL_W+1];
        }

        // // Preserve the first input value if possible.
        // if (N > 0) output_flat(0) = 1;
    }
};

REGISTER_KERNEL_BUILDER(Name("CustomConv2d").Device(DEVICE_CPU), CustomConv2dOp);