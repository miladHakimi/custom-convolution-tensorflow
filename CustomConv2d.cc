#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/padding.h"
#include <vector>

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
    // .SetShapeFn(::tensorflow::shape_inference::DepthwiseConv2DNativeShape);
    // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //     ::tensorflow::shape_inference::DimensionOrConstant doc(1);
    //     return ConcatV2Shape(c);
    //     // c->set_output(0, c->MakeShape({doc}));
    //     // return Status::OK();
    // });