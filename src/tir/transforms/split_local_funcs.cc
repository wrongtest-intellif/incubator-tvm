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
 * \file split_local_funcs.cc
 * \brief Split localy invoked functions.
 */
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>


namespace tvm {
namespace tir {


class LocalCallRewritter : public StmtExprMutator {
 public:
  explicit LocalCallRewritter(IRModule ir_module_)
      : ir_module_(ir_module_) {}

  IRModule Rewrite() {
    IRModule updated;

    // replace local func invocations
    for (const auto& kv : ir_module_->functions) {
      if (!kv.second->IsInstance<PrimFuncNode>()) {
        continue;
      }
      PrimFunc func = Downcast<PrimFunc>(kv.second);
      func.CopyOnWrite()->body = VisitStmt(func->body);
      updated->Add(kv.first, func);
    }

    // split local func from origin
    for (const auto& kv : local_gv_dict_) {
      const GlobalVar& local_gv = kv.first;
      PrimFunc func = Downcast<PrimFunc>(ir_module_->Lookup(kv.first));
      Array<PrimExpr> local_args;
      for (const Var& param : func->params) {
        auto it = func->buffer_map.find(param);
        if (it == func->buffer_map.end()) {
          local_args.push_back(param);
        } else {
          local_args.push_back((*it).second->data);
        }
      }
      PrimType ret_type = Downcast<PrimType>(func->ret_type);
      Call subcall(ret_type->dtype, local_gv, local_args);
      if (ret_type->dtype.is_void()) {
        func.CopyOnWrite()->body = Evaluate(subcall);
      } else {
        func.CopyOnWrite()->body = Evaluate(ret(subcall));
      }
      updated->Add(local_gv, MakeLocalFunction(func, local_gv->name_hint));
      updated->AddUnchecked(kv.first, func);
    }
    ir_module_->Update(updated);
    return std::move(ir_module_);
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (!op->op->IsInstance<GlobalVarNode>()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    GlobalVar origin_gv = Downcast<GlobalVar>(op->op);
    PrimFunc callee = Downcast<PrimFunc>(ir_module_->Lookup(origin_gv));
    Integer is_entry = callee->GetAttr<Integer>(tir::attr::kIsGlobalFunc).value_or(Integer(0));
    Integer is_global = callee->GetAttr<Integer>(tir::attr::kIsEntryFunc).value_or(Integer(0));
    Integer call_conv = callee->GetAttr<Integer>(tvm::attr::kCallingConv).value_or(Integer(CallingConv::kDefault));
    ICHECK_EQ(is_entry->value, 0) << "Callee function " << origin_gv << " should not be entry function";
    ICHECK_EQ(is_global->value, 0) << "Callee function " << origin_gv << " should not be global function";
    ICHECK(call_conv == CallingConv::kDefault) << "Callee function " << origin_gv << " should have default call convention";

    // replace with new gv representing the local func
    GlobalVar local_gv;
    auto it = local_gv_dict_.find(origin_gv);
    if (it == local_gv_dict_.end()) {
      std::string local_symbol = origin_gv->name_hint + "_local" + std::to_string(local_gv_dict_.size());
      local_gv = GlobalVar(local_symbol);
      local_gv_dict_.insert(it, {origin_gv, local_gv});
    } else {
      local_gv = (*it).second;
    }
    Array<PrimExpr> args;
    for (const auto& e : op->args) {
      args.push_back(VisitExpr(e));
    }
    return std::move(Call(op->dtype, local_gv, args, op->span));
  }

  /*! \brief convert function to a function which can be invoked locally */
  PrimFunc MakeLocalFunction(PrimFunc func, String funcname) {
    auto n = func.CopyOnWrite();
    Array<Var> new_params;
    Map<tir::Var, PrimExpr> remap_vars;
    for (const Var& param : func->params) {
      auto it = func->buffer_map.find(param);
      if (it != func->buffer_map.end()) {
        tir::Var new_param(param->name_hint, PointerType(PrimType((*it).second->dtype)));
        new_params.push_back(new_param);
        remap_vars.Set((*it).second->data, new_param);
      } else {
        new_params.push_back(param);
      }
    }
    n->body = Substitute(n->body, remap_vars);
    n->params = new_params;
    n->buffer_map.clear();
    n->preflattened_buffer_map.clear();
    PrimFunc local_func = GetRef<PrimFunc>(n);
    local_func = WithAttr(local_func, tvm::attr::kGlobalSymbol, funcname);
    local_func = WithAttr(local_func, tvm::attr::kCallingConv, Integer(CallingConv::kDefault));
    local_func = WithAttr(local_func, tir::attr::kIsGlobalFunc, Integer(0));
    return local_func;
  }

  /*! \brief origin ir module */
  IRModule ir_module_;  
  /*! \brief mapping from origin gv to local func gv */
  std::unordered_map<GlobalVar, GlobalVar, ObjectPtrHash, ObjectPtrEqual> local_gv_dict_;
};



namespace transform {

Pass SplitLocalFuncs() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    LocalCallRewritter rewritter(mod);
    return rewritter.Rewrite();
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.SplitLocalFuncs", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SplitLocalFuncs").set_body_typed(SplitLocalFuncs);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
