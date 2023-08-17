from paddle.utils.cpp_extension import CUDAExtension, setup, load

setup(
    name='flash_atten2',
    ext_modules=CUDAExtension(
        sources = [
            'flash_attn_fwd.cu',
            "flash_attention/flash_fwd_hdim32_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim32_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim64_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim64_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim96_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim96_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim128_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim128_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim160_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim160_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim192_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim192_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim224_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim224_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim256_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim256_bf16_sm80.cu",
        ]
    ),
    include_dirs=['cutlass/include'],
    extra_compile_args={
        "nvcc": [
            "-gencode",
            "arch=compute_86,code=sm_86",
            # "-O3",
            # "-std=c++17",
            # "-U__CUDA_NO_HALF_OPERATORS__",
            # "-U__CUDA_NO_HALF_CONVERSIONS__",
            # "-U__CUDA_NO_HALF2_OPERATORS__",
            # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            # "--expt-relaxed-constexpr",
            # "--expt-extended-lambda",
            # "--use_fast_math",
            # "--ptxas-options=-v",
            # "-lineinfo"
        ],
        },  
)

# custom_op_module = load(
#     name="custom_setup_ops",    # 生成动态链接库的名称
#     sources = ['flash_attn_fwd.cu', 'flash_attention/flash_fwd_sm80.cu'],
#     extra_cuda_cflags = ['-std=c++17', '-arch=compute_80', '-code=sm_80'],  #编译选项
#     extra_include_paths = ['/root/paddlejob/workspace/output/lizhenyun/custom_op/cutlass/include'],
#     # build_directory = './custom_ops',
#     verbose=True,    # 打印编译过程中的日志信息
# )