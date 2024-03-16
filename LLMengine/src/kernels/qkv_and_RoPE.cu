#include <math.h>
#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/qkv_bias_and_RoPE.h"


inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    // (RussWong) note: 每个token所属的id, 它的freq值都是固定的, id的上限为max position embedding
    // t_step表示token id（这里考虑了多轮对话历史上下文长度)
    // 每个freq值对应于zid = head size维度上0 2 4 6 ... 64带入下式计算
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}

template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T *q_buf,
                                                   T *k_buf,
                                                   T *v_buf,
                                                   T *QKV,
                                                   const int *padding_offset, // created before qkv linear
                                                   const int *history_length,
                                                   const int *input_length, // actual length of each seq
                                                   const int batch_size,
                                                   const int seq_len, // max_seq_len to pad to
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotary_embedding_base, // default 10000 in llama
                                                   int max_position_embeddings, /*default 2048 in llama*/
                                                   bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
{

    int token_id = blockIdx.x;
    int head_id = blockIdx.y; //blockIdx.y对应与head_size
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];
    // 1. prapare rebuilding , do rebuild padding and transpose when store
    int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding

    int batch_id = dst_token_id / seq_len;       // seqlen is max_seq_len for padding used to unify all seq's length
    int local_token_id = dst_token_id % seq_len; // 每个seq中的局部token id

    int qkv_head_num = head_num + 2 * kv_head_num;
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid;
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size;
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size + kv_head_num * head_size;

    float v = QKV[v_id];
    int dst_q_id = batch_id * seq_len * head_num * head_size +
                   head_id * seq_len * head_size +
                   local_token_id * head_size + tid;

    int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                    head_id * seq_len * head_size +
                    local_token_id * head_size + tid;
    if (head_id < kv_head_num)
    {
        v_buf[dst_kv_id] = v;
    }
    // 3. RoPE
    const int cur_seq_history_len = history_length[batch_id];
    const int context_length = cur_seq_history_len + input_length[batch_id];
    const int timestep = cur_seq_history_len + local_token_id; 
    if (tid >= rotary_embedding_dim / 2)
    {
        return;
    } // tid = [0,1,2,...,63]

    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, timestep);
    float2 q_rotate = GetRoPEres(QKV[q_id], QKV[q_id + head_size / 2], cos_sin);
    float2 k_rotate = GetRoPEres(QKV[k_id], QKV[k_id + head_size / 2], cos_sin);
    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;
    if (head_id < kv_head_num)
    { // for MQA and GQA
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}

template <typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T> *q_buf,
                                           TensorWrapper<T> *k_buf,
                                           TensorWrapper<T> *v_buf,
                                           TensorWrapper<T> *QKV,
                                           BaseWeight<T> &qkv,
                                           // Tensor* qkv_bias,
                                           TensorWrapper<int> *padding_offset,
                                           TensorWrapper<int> *history_length,
                                           TensorWrapper<int> *input_length,
                                           LLaMAAttentionStaticParams &params)
{
    //input shape [token_num,head_num,head_size]
    //线程对到最后一维度
    int token_num = QKV->shape[0];
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];
    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];
    int kv_head_num = (qkv_head_num - head_num) / 2;

    dim3 grid(token_num, head_num);
    dim3 block(head_size);
    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(q_buf->data,
                                                           k_buf->data,
                                                           v_buf->data,
                                                           QKV->data,
                                                           /*optional*/qkv.bias,
                                                           padding_offset->data,
                                                           history_length->data,
                                                           input_length->data,
                                                           batch_size,
                                                           seq_len,
                                                           token_num,
                                                           head_num,
                                                           kv_head_num,
                                                           head_size,
                                                           params.rotary_embedding_dim,
                                                           params.rotary_embedding_base,
                                                           params.max_position_embeddings,
                                                           params.use_dynamic_ntk);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(q_buf->data);
#else
#endif
}


template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<float> *q_buf,
                                                    TensorWrapper<float> *k_buf,
                                                    TensorWrapper<float> *v_buf,
                                                    TensorWrapper<float> *QKV,
                                                    BaseWeight<float> &qkv,
                                                    TensorWrapper<int> *padding_offset,
                                                    TensorWrapper<int> *history_length,
                                                    TensorWrapper<int> *input_length,
                                                    LLaMAAttentionStaticParams &params);




