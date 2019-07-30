#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/padding.h"
#include <vector>
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;

REGISTER_OP("CustomConv2d")
    .Input("input: float32")
    .Input("filter: float32")
    .Input("strides: int32")
    .Input("padding: string")
    .Output("conved: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    	::tensorflow::shape_inference::ShapeHandle input_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
		::tensorflow::shape_inference::ShapeHandle filter_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
		
		::tensorflow::shape_inference::DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
		::tensorflow::shape_inference::DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
		::tensorflow::shape_inference::DimensionHandle in_cols_dim = c->Dim(input_shape, 2);
		::tensorflow::shape_inference::DimensionHandle output_depth_dim = c->Dim(filter_shape, 3);

		::tensorflow::shape_inference::ShapeHandle output_shape;
  		TensorFormat data_format;
      	output_shape = c->MakeShape({batch_size_dim, in_rows_dim,
                                 in_cols_dim, output_depth_dim});
		c->set_output(0, output_shape);
		return Status::OK();
    });
   