#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/padding.h"
#include <vector>
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;

REGISTER_OP("CustomConv2d")
    // .Attr("strides: list(int) = []")
    // .Attr("padding: string")
    .Input("input: float32")
    .Input("filter: float32")
    .Output("conved: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });