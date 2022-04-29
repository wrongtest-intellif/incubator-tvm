import tvm
from tvm.script import tir as T
import numpy as np


@tvm.script.ir_module
class MyModule:

    @T.prim_func
    def plus_one(a: T.handle):
        T.func_attr({"global_symbol": "plus_one", "tir.noalias": True,})
        A = T.match_buffer(a, (8,), dtype="float32")
        for i in range(8):
            with T.block("plus"):
                vi = T.axis.remap("S", [i])
                A[vi] = A[vi] + 1.0

    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            with T.block("B"):
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0
        T.evaluate(plus_one(B.data, dtype=""))


f = tvm.build(MyModule, target="llvm")
print(f.get_source())
x = tvm.nd.array(np.ones([8]).astype("float32"))
y = tvm.nd.array(np.ones([8]).astype("float32"))
f["main"](x, y)
print(y.numpy())
