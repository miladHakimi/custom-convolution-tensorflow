#include "tensorflow/core/framework/op_kernel.h"
#include <string.h>

using namespace tensorflow;

class CustomConv2dOp : public OpKernel {
    public:
    explicit CustomConv2dOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override {

        const Tensor& input_tensor1 = context->input(0);
        const Tensor& input_tensor2 = context->input(1);
        const Tensor& input_tensor3 = context->input(2);
        const Tensor& input_tensor4 = context->input(3);
        
        auto padding = input_tensor4.flat<string>();
        auto strides = input_tensor3.flat<float>();
        
        int FIL_H = input_tensor2.shape().dim_size(0);
        int FIL_W =input_tensor2.shape().dim_size(1);
        int IMG_H = input_tensor1.shape().dim_size(1);
        int IMG_W = input_tensor1.shape().dim_size(2);

        int width , height, count, depth;

        if(padding(0) == "SAME"){
            width = ceil(IMG_W/strides(1));
            height = ceil(IMG_H/strides(0));
        } 
        else{
            width = ceil((input_tensor1.shape().dim_size(2)-input_tensor2.shape().dim_size(1)+1)/strides(1));
            height = ceil((input_tensor1.shape().dim_size(1)-input_tensor2.shape().dim_size(0)+1)/strides(0));
        }
        count = input_tensor1.shape().dim_size(0);
        depth = input_tensor2.shape().dim_size(3);
        tensorflow::TensorShape ts(input_tensor1.shape());
        ts.set_dim(0, count);
        ts.set_dim(1, height);
        ts.set_dim(2, width);
        ts.set_dim(3, depth);

        auto img = input_tensor1.flat<float>();
        auto filter = input_tensor2.flat<float>();
        float partial_conv;
        float conv_res[height][width];
        float conv_ans[count][height][width];
        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, ts,&output_tensor));
        auto output_flat = output_tensor->flat<float>();
        // convolution body
        for (int m = 0; m < count; m++){
            for (int n = 0; n < depth; n++){
                for (int i = 0; i < height; i+=1){
                    for (int j = 0; j < width; j+=1){
                        partial_conv = 0;
                        for (int k = 0; k < FIL_H; k+=1)
                            if(i*strides(0)+k < IMG_H)
                                for (int l = 0; l < FIL_W; l+=1)
                                    if(j*strides(0)+l < IMG_W)
                                        partial_conv += filter(n*FIL_H*FIL_W+k*FIL_W+l) * img(m*(IMG_H*IMG_W)+ (i*strides(0)+k)*IMG_W+l+j*strides(1));
                                    else
                                        break;           
                            else
                                break;                                 
                        output_flat(m*height*width*depth+n*height*width+i*width+j) = partial_conv;
                    }
                }
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("CustomConv2d").Device(DEVICE_CPU), CustomConv2dOp);