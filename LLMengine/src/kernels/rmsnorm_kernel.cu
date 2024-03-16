#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/rmsnorm_kernel.h"


template <typename T> __device__  T warpReduceSum(T val){
	for(int i = 32/2;i>0;i>>=1){
		val += __shfl_xor_sync(0xffffffff,val,i);
	}
	return val;
}

template<typename T> blockReduceSum(T val){
	int tid = threadIdx.x;
	int wid = tid/32;
	int laned = tid%32;
	int warpNum = (blockDim.x+31)/32;
	static __shared__ int warpSum[64];
	val = warpReduceSum<T>(val);
	if(laned == 0){
		warpSum[wid]=val;
	}
	__syncthreads();
	T sum = tid<warpNum ? warpSum[tid]:(T)0;
	sum = warpReduceSum<T>(sum);
	return sum;
}

template<typename T>
__global__ void RMSNorm(T* decoder_out,
						T* scale,
						float eps,
						int num_tokens,
						int hidden_units
						){
	int vec_size = Vec<T>::size;
	using Vec_t = typename Vec<T>::Type;
	float thread_sum =0.0f;
	Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out+blockIdx.x*hidden_units);
	for(int idx = threadIdx.x;idx < hidden_units/vec_size;idx+=blockDim.x){
		thread_sum += dout[idx].x*dout[idx].x;
		thread_sum += dout[idx].y*dout[idx].y;
		thread_sum += dout[idx].z*dout[idx].z;
		thread_sum += dout[idx].w*dout[idx].w;
	}
	thread_sum = blcokReduceSum<float>(thread_sum);
	if(threadIdx.x==0){
		__shared__ float inv_mean;
		inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
	}
	Vec_t* s = reinterpret_cast<Vec_t*>(scale+blockIdx.x*hidden_units);
	for(int idx = threadIdx.x;idx<hidden_units/vec_size;idx+=blockDim.x){
		dout[idx] = dout[idx]*inv_mean * s[idx].x;
		dout[idx] = dout[idx]*inv_mean * s[idx].y;
		dout[idx] = dout[idx]*inv_mean * s[idx].z;
		dout[idx] = dout[idx]*inv_mean * s[idx].w;
	}

}



template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                    LayerNormWeight<T>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last // for print last rmsnorm output to debug
                    )
{
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / 4; //vec size // assume head size can be divided by 4 and 2
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSNorm<T><<<grid, block>>>(decoder_out->data,
                            attn_norm_weight.gamma,
                            eps,
                            num_tokens,
                            hidden_units);
}

template void launchRMSNorm( TensorWrapper<float>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<float>* decoder_residual,
                    LayerNormWeight<float>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last
                    );