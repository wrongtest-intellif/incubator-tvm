/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file combine_parallel_dense.cc
 * \brief Combine parallel dense ops into a single dense.
 *
 * This pass replaces dense ops that share the same input node, same shape,
 * and don't have "units" defined with a single batch matrix multiplication.
 * The inputs of the new batch_matmul is the stack of the original inputs.
 * Elemwise and broadcast ops following dense are also combined if possible.
 *
 * This prevents launching multiple kernels in networks with multiple
 * dense branches, such as BERT.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "./combine_parallel_op_batch.h"
#include "./expr_subst.h"
#include "./pattern_util.h"

namespace tvm {
namespace relay {

class ParallelDenseBatchCombiner : public ParallelOpBatchCombiner {
 public:
  explicit ParallelDenseBatchCombiner(uint64_t min_num_branches)
      : ParallelOpBatchCombiner("nn.dense", "nn.batch_matmul", min_num_branches) {}

 protected:
  virtual bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    AttrsEqual eq;
    const auto* attrs_a = a->attrs.as<DenseAttrs>();
    const auto* attrs_b = b->attrs.as<DenseAttrs>();
    CHECK(attrs_a);
    CHECK(attrs_b);
    const auto* weight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* weight_b = b->args[1]->type_as<TensorTypeNode>();

    return eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
           eq(weight_a->shape[0], weight_b->shape[0]) && eq(weight_a->shape[1], weight_b->shape[1]);
  }
};

class ParallelDenseFlatCombiner : public ParallelOpCombiner {
 public:
  explicit ParallelDenseFlatCombiner(uint64_t min_num_branches)
      : ParallelOpCombiner("nn.dense", min_num_branches) {}

 protected:
  bool IsSupportedOp(const CallNode* n) { return true; }

  bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    AttrsEqual eq;
    const auto* attrs_a = a->attrs.as<DenseAttrs>();
    const auto* attrs_b = b->attrs.as<DenseAttrs>();
    const auto* weight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* weight_b = b->args[1]->type_as<TensorTypeNode>();
    CHECK(attrs_a != nullptr && attrs_b != nullptr && weight_a != nullptr && weight_b != nullptr);
    // output sizes can be different
    return eq(attrs_a->out_dtype, attrs_b->out_dtype) && eq(weight_a->shape[1], weight_b->shape[1]);
  }

  Call MakeCombinedOp(const Group& branches) {
    const Op& dense_op = Op::Get("nn.dense");
    Expr input = branches[0][0]->args[0];
    Expr new_weight;
    IndexExpr new_output_dims;
    std::tie(new_weight, new_output_dims) = TransformWeight(branches);
    const auto* origin_attrs = branches[0][0]->attrs.as<DenseAttrs>();
    CHECK(origin_attrs);
    const auto dense_attrs = make_object<DenseAttrs>();
    dense_attrs->units = new_output_dims;
    dense_attrs->out_dtype = origin_attrs->out_dtype;
    return CallNode::make(dense_op, {input, new_weight}, Attrs{dense_attrs}, {});
  }

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) {
    AttrsEqual eq;
    auto ta = a->args[index]->type_as<TensorTypeNode>();
    auto tb = b->args[index]->type_as<TensorTypeNode>();
    auto toutput_a = a->type_as<TensorTypeNode>();
    auto toutput_b = b->type_as<TensorTypeNode>();
    CHECK(ta != nullptr && tb != nullptr && toutput_a != nullptr && toutput_b != nullptr);

    if (!eq(ta->dtype, tb->dtype) || ta->shape.size() != tb->shape.size()) {
      return false;
    }
    if (toutput_a->shape.size() < ta->shape.size() || toutput_b->shape.size() < tb->shape.size()) {
      return false;  // not broadcast/elemwise
    }
    for (size_t i = 0; i < ta->shape.size() - 1; i++) {
      // shape dims must match except last dim
      if (!eq(ta->shape[i], tb->shape[i])) return false;
    }
    return true;
  }

  Call MakeCombinedCallFromFollowingOps(const Expr& data, const Group& branches, size_t depth,
                                        size_t parent_index) {
    Array<Expr> new_args;
    const CallNode* call = branches[0][depth];
    for (size_t i = 0; i < call->args.size(); i++) {
      if (i == parent_index) {
        new_args.push_back(data);
        continue;
      }
      size_t arg_ndim = call->args[i]->type_as<TensorTypeNode>()->shape.size();
      size_t concat_axis = arg_ndim == 0 ? 0 : arg_ndim - 1;

      Array<Expr> tuple;
      for (const auto& branch : branches) {
        auto parent = branch[depth]->args[parent_index];
        auto& parent_shape = parent->type_as<TensorTypeNode>()->shape;
        auto out_dim = as_const_int(parent_shape[parent_shape.size() - 1]);
        CHECK(out_dim != nullptr);

        auto arg = branch[depth]->args[i];
        auto& arg_shape = arg->type_as<TensorTypeNode>()->shape;
        bool repeat_last_dim = false;
        if (arg_ndim == 0) {
          repeat_last_dim = true;
        } else {
          auto arg_last_dim = as_const_int(arg_shape[arg_shape.size() - 1]);
          CHECK(arg_last_dim != nullptr);
          if (*out_dim > 1 && *arg_last_dim == 1) {
            repeat_last_dim = true;
          }
        }
        if (repeat_last_dim) {
          // ensure broadcast is valid after concat args
          arg = MakeRepeat(arg, *out_dim, 0);
        }
        tuple.push_back(arg);
      }
      auto concat = MakeConcatenate(TupleNode::make(tuple), concat_axis);
      new_args.push_back(std::move(concat));
    }
    return CallNode::make(call->op, new_args, call->attrs, {});
  }

  void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth,
                         ExprSubstMap* subst_map) {
    int index = 0;
    for (const auto& branch : branches) {
      const CallNode* dense = branch[0];
      auto weight = dense->args[1]->type_as<TensorTypeNode>();
      int out_dims = *as_const_int(weight->shape[0]);
      Array<Integer> begin = {0, index};
      index += out_dims;
      Array<Integer> end = {NullValue<Integer>(), index};
      auto slice = MakeStridedSlice(data, std::move(begin), std::move(end), Array<Integer>{});
      subst_map->insert({GetRef<Expr>(branch[depth]), slice});
    }
  }

 private:
  std::tuple<Expr, IndexExpr> TransformWeight(const Group& branches) {
    int64_t output_dims = 0;
    Array<Expr> weights;
    for (const auto& branch : branches) {
      auto weight = branch[0]->args[1];
      weights.push_back(weight);
      output_dims += *as_const_int(weight->type_as<TensorTypeNode>()->shape[0]);
    }
    return std::make_tuple(MakeConcatenate(TupleNode::make(weights), 0),
                           MakeConstScalar(Int(32), output_dims));
  }
};

/*! \brief Combine parallel dense if number of branches >= min_num_branches */
Expr CombineParallelDense(const Expr& expr, uint64_t min_num_branches, bool to_batch) {
  if (to_batch) {
    return ParallelDenseBatchCombiner(min_num_branches).Combine(expr);
  } else {
    return ParallelDenseFlatCombiner(min_num_branches).Combine(expr);
  }
}

namespace transform {

Pass CombineParallelDense(uint64_t min_num_branches, bool to_batch) {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(CombineParallelDense(f, min_num_branches, to_batch));
      };
  return CreateFunctionPass(pass_func, 4, "CombineParallelDense",
                            {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.CombineParallelDense").set_body_typed(CombineParallelDense);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
