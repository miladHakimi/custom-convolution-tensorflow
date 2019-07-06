#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <string>

using namespace tensorflow;

REGISTER_OP("CustomConv2d")
    .Input("input: float32")
    .Input("filter: float32")
    .Output("conved: float32")
    // .SetShapeFn(::tensorflow::shape_inference::DepthwiseConv2DNativeShape);
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeManager a;
        ::tensorflow::shape_inference::ShapeHandle b;
        // b = a.MakeShape({1, 1});

        // TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &n));
        ::tensorflow::shape_inference::ShapeHandle out;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
        // TF_RETURN_IF_ERROR(c->Concatenate(n, c->input(0), &out));
        c->set_output(0, out);
        return Status::OK();
    });