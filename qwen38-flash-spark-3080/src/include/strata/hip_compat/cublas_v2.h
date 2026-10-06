#pragma once
#include <hipblas/hipblas.h>
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode hipblasSetMathMode
#define cublasSetStream hipblasSetStream
#define cublasSetWorkspace hipblasSetWorkspace
#define cublasStatus_t hipblasStatus_t
