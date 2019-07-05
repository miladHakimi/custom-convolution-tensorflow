#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <string>

using namespace tensorflow;

REGISTER_OP("CustomConv2d")
    .Input("
        input: float32[][],
        filter: float32[][], 
        strides: String, 
        Padding: String,
        name=String,
        ");
    .Output("conved: float32[][]")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });