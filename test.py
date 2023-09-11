import numpy
import tvm
import time
import tvm.testing

load_mmap = tvm.get_global_func("runtime.LoadParamsMMap")
save_mmap = tvm.get_global_func("runtime.SaveParamsMMap")
#path = "/12t-data/baoxinqi/tianshu/quan_tianshu_4096_min_max_prof1_floatop_add_mult_int4_out_int8"

path = "tianshu"
begin = time.time()
params1 = {
  "x": tvm.nd.array(numpy.random.randint(-100, 100, [320, 10])),
  "y": tvm.nd.array(numpy.random.randint(-100, 100, [320, 10]))
}
# params1 = tvm.runtime.load_param_dict_from_file(path + ".params")
print("load_param_dict_from_file耗时: ", time.time() - begin)

begin = time.time()
tvm.runtime.save_param_dict_to_file(params1, path + ".copy.params")
print("save dict耗时: ", time.time() - begin)

begin = time.time()
save_mmap(params1, path + ".mmap.params")
print("save mmap dict耗时: ", time.time() - begin)

begin = time.time()
params2 = load_mmap(path + ".mmap.params")
print("load mmap耗时: ", time.time() - begin)

for k in params1:
    tvm.testing.assert_allclose(params1[k].numpy(), params2[k].numpy(), 0, 0)

