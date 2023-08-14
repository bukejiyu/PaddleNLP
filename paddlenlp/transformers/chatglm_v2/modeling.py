# Copyright (c) 2023 ChatGLM2-6B Model Team and PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from custom_setup_ops import  write_cache_kv
USE_FLASH2 = True
if USE_FLASH2:
    from flash_atten2 import flash_attn_varlen_fwd
else:
    flash_attn_varlen_fwd = None
from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
    ModelOutput,
)
from .configuration import CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP, ChatGLMv2Config

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
]


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, original_impl=False):
        super().__init__()
        self.dtype = paddle.get_default_dtype()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(self, seq_len: int, n_elem: int, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (paddle.arange(0, n_elem, 2, dtype="float32") / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = paddle.arange(0, seq_len, dtype=theta.dtype)

        # Calculate the product of position index and $\theta_i$
        idx_theta = paddle.outer(seq_idx, theta).astype(self.dtype)

        cache = paddle.stack([paddle.cos(idx_theta), paddle.sin(idx_theta)], axis=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if self.dtype in (paddle.float16, paddle.bfloat16, paddle.int8):
            cache = cache.astype(self.dtype)
            # cache = cache.bfloat16() if dtype == paddle.bfloat16 else cache.astype("float16")
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(seq_len=max_seq_len, n_elem=self.dim)


# @paddle.jit.script
def apply_rotary_pos_emb(x: paddle.Tensor, rope_cache: paddle.Tensor) -> paddle.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape([sq, -1, np, rot_dim // 2, 2])
    rope_cache = rope_cache.reshape([sq, -1, 1, xshaped.shape[3], 2])
    x_out2 = paddle.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return paddle.concat((x_out2, x_pass), axis=-1)


class RMSNorm(nn.Layer):
    def __init__(self, hidden_size, epsilon=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.epsilon = 1e-5 if epsilon is None else epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
        hidden_states = paddle.rsqrt(variance + self.epsilon) * hidden_states
        output = (hidden_states * self.weight).astype(input_dtype)

        # if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
        #     hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return output
def save_tensor_to_txt(tensor, file_path):
    np_array = tensor.numpy()  # Convert Paddle tensor to NumPy array
    np.savetxt(file_path, np_array)  # Save the NumPy array to the text file

class CoreAttention(nn.Layer):
    def __init__(self, config: ChatGLMv2Config, layer_number):
        super(CoreAttention, self).__init__()

        self.dtype = paddle.get_default_dtype()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # Raw attention scores
        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape([output_size[2], output_size[0] * output_size[1], -1])
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape([output_size[3], output_size[0] * output_size[1], -1])

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = paddle.bmm(query_layer.transpose([1, 0, 2]), key_layer.transpose([1, 2, 0])) * (
            1.0 / self.norm_factor
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.reshape(output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.astype("float32")
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = paddle.tril(
                paddle.ones((output_size[0], 1, output_size[2], output_size[3]), dtype="bool")
            )
            attention_mask = ~attention_mask

        if attention_mask is not None:
            attention_scores = paddle.where(
                attention_mask > 0,
                paddle.full_like(attention_scores, paddle.finfo(attention_scores.dtype).min),
                attention_scores,
            )

        attention_probs = F.softmax(attention_scores.astype("float32"), axis=-1)
        attention_probs = attention_probs.astype(self.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]de
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape([value_layer.shape[0], output_size[0] * output_size[1], -1])
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape([output_size[0] * output_size[1], output_size[2], -1])
        # matmul: [b * np, sq, hn]
        context_layer = paddle.bmm(attention_probs, value_layer.transpose([1, 0, 2]))
        # change view [b, np, sq, hn]
        context_layer = context_layer.reshape(output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.transpose([2, 0, 1, 3])
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context_layer.shape[:-2] + [self.hidden_size_per_partition]
        context_layer = context_layer.reshape(new_context_shape)

        return context_layer
def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None,
    kvpacked=False, qkvpacked=False):
    """
    Arguments:
        qkv: (batch_size, seqlen_q, 3, num_heads, head_size)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool/int
        key_padding_mask: (batch_size, seqlen), bool/int
    """
    # print("q:", q.shape)
    # print("k:", k.shape)
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, num_heads, head_size = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == [batch_size, seqlen_k, nheads_k, head_size]
    assert v.shape == [batch_size, seqlen_k, nheads_k, head_size]
    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen_q, num_heads, head_size)
    else:
        q_unpad=q.reshape([q.shape[0]*q.shape[1],q.shape[2],q.shape[3]])
        cu_seqlens_q=paddle.arange(0,(batch_size+1)*seqlen_q,step=seqlen_q,dtype=paddle.int32)
        max_seqlen_q=paddle.to_tensor([seqlen_q])
        output_pad_fn = lambda output_unpad:  output_unpad.reshape([batch_size,-1,output_unpad.shape[1],output_unpad.shape[2]])
        
    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = k.reshape([k.shape[0]*k.shape[1],k.shape[2],k.shape[3]])
        v_unpad = v.reshape([v.shape[0]*v.shape[1],v.shape[2],v.shape[3]])
        cu_seqlens_k = paddle.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=paddle.int32)
        max_seqlen_k = paddle.to_tensor([seqlen_k])

    if (not qkvpacked) and (not kvpacked):
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k, num_heads, head_size)
        return (q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, k, v, output_pad_fn, None, None)

def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = paddle.sum(attention_mask, axis=-1, dtype="int32")
    indices = paddle.flatten(paddle.nonzero(paddle.flatten(attention_mask.cast(paddle.int32)), as_tuple=False))
    max_seqlen_in_batch = seqlens_in_batch.max()
    cu_seqlens = F.pad(
        paddle.cumsum(seqlens_in_batch, axis=0, dtype="int32"),
        (1, 0))
    
    hidden_states = paddle.flatten(hidden_states, start_axis = 0, stop_axis = 1)
    hidden_states_new = []
    for i in indices:
        hidden_states_new.append(hidden_states[i:i+1])
    return (
            paddle.concat(hidden_states_new), 
            indices, cu_seqlens, max_seqlen_in_batch
        )

def pad_input(hidden_states, indices, batch, seqlen, num_heads, head_size):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    output = paddle.zeros([batch * seqlen, num_heads, head_size], dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    for i in range(indices.size):
        output[indices[i]] = hidden_states[i]
    return output.reshape([batch, seqlen, num_heads, head_size])


def use_kernel_encoder(query_layer,key_layer,value_layer,num_head,num_head_kv,dim_head,cache_kvs=None,attention_mask=None):
    from flash_atten2 import flash_attn_varlen_fwd
    q_ = query_layer.transpose([1, 0, 2, 3])
    k_ = key_layer.transpose([1, 0, 2, 3])
    v_ = value_layer.transpose([1, 0, 2, 3])
    (q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, k, v,
                output_pad_fn, _, _) = generate_qkv(
                    q_,
                    k_,
                    v_,
                    attention_mask,
                    attention_mask,
                    )
    scale = float(dim_head ** -0.5)
    zero_tensors = False
    is_causal = True
    fmha_out = flash_attn_varlen_fwd(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, zero_tensors, is_causal)
    fmha_out = output_pad_fn(fmha_out)
    fmha_out = fmha_out.transpose([1,0,2,3])
    fmha_out =fmha_out.reshape([fmha_out.shape[0],fmha_out.shape[1],fmha_out.shape[2]*fmha_out.shape[3]])
    return fmha_out
def use_kernel_decoder(q,k,v,kv_cache,num_head,num_head_kv,dim_head,mode="mqa",cache_kvs=None):
    #q 做完位置编码  [s_q,bz,num_head,head_dim]
    #k 做完位置编码，seq=1 [s_kv,bz,num_head_kv,head_dim]
    #v seq=1 [s_kv,bz,num_head_kv,head_dim]
    #cache_k [seq,bz,num_head_kv,head_dim]
    #cache_kv_mmha [2,bz,num_head_kv,seq,head_dim]
    #output_size [seq,bz,num_head*dim_head]
    output_size = (1, q.shape[1], q.shape[2]*q.shape[3])
    q=q.squeeze(0)
    k=k.squeeze(0)
    v=v.squeeze(0)
    bz=q.shape[0]
    cache_k,cache_v=kv_cache
    seq_before_concat=cache_k.shape[0]+1
    src_mask=paddle.zeros([bz,1,1,seq_before_concat]).cast('float16')
    seq_len=seq_before_concat-1
    cache_kv_out=paddle.concat((cache_k.unsqueeze(0).transpose([0,2,3,1,4]), cache_v.unsqueeze(0).transpose([0,2,3,1,4])), axis=0).cast('float16')
    #x=paddle.concat((q, k, v), axis=1).cast('float16')
    if mode=="mqa":
        from paddle.incubate.nn.functional.masked_multiquery_attention import masked_multiquery_attention
        # save_tensor_to_txt(q.reshape([q.shape[0]*q.shape[1],q.shape[2]]),"/workspace/test_logs/data_input/q_input.txt")
        # save_tensor_to_txt(k.reshape([k.shape[0]*k.shape[1],k.shape[2]]),"/workspace/test_logs/data_input/k_input.txt")
        # save_tensor_to_txt(v.reshape([v.shape[0]*v.shape[1],v.shape[2]]),"/workspace/test_logs/data_input/v_input.txt")
        # save_tensor_to_txt(cache_k.reshape([cache_k.shape[0]*cache_k.shape[1]*cache_k.shape[2],cache_k.shape[3]]),"/workspace/test_logs/data_input/cache_k.txt")
        # save_tensor_to_txt(cache_v.reshape([cache_v.shape[0]*cache_v.shape[1]*cache_v.shape[2],cache_v.shape[3]]),"/workspace/test_logs/data_input/cache_v.txt")
        # save_tensor_to_txt(cache_kvs.reshape([cache_kvs.shape[0]*cache_kvs.shape[1]*cache_kvs.shape[2]*cache_kvs.shape[3],cache_kvs.shape[4]]),"/workspace/test_logs/data_input/cache_kv.txt")
        paddle_mqa_out = masked_multiquery_attention(
                x=q,
                cache_kv=cache_kvs,
                kv_input= paddle.concat((k,v),axis=1),
                src_mask=src_mask,
                sequence_lengths=None,
                rotary_tensor=None,
                beam_cache_offset=None,
                qkv_out_scale=None,
                out_linear_shift=None,
                out_linear_smooth=None,
                seq_len=seq_len,
                rotary_emb_dims=0,
                kv_split=True,
                head_kv=num_head_kv,
                use_neox_rotary_style=False,        
                )
        paddle_mqa_out=paddle_mqa_out[0]
        paddle_mqa_out=paddle_mqa_out.transpose([1,0,2,3])
        paddle_mqa_out = paddle_mqa_out.reshape(output_size)
        return paddle_mqa_out
    elif mode=="mmha":
        k = k.unsqueeze(-2)
        k = k.tile(
                    [ 1, 1, num_head // num_head_kv, 1]
            )
        k = k.reshape(
                    k.shape[:1] + [num_head, dim_head]
            )
        v = v.unsqueeze(-2)
        v = v.tile(
                   [ 1, 1, num_head // num_head_kv, 1]
                )
        v = v.reshape(
                    v.shape[:1] + [num_head, dim_head]
            )
        x = paddle.concat([q.unsqueeze(1).cast('float16'),k.unsqueeze(1).cast('float16'),v.unsqueeze(1).cast('float16')],axis=1).cast('float16')
        cum_offsets=None
        sequence_lengths=None
        rotary_tensor=None
        beam_cache_offset=None
        qkv_out_scale=None
        out_linear_shift=None
        seq_len= seq_len
        out_linear_in_scale= -1
        rotary_emb_dims = 0
        use_neox_rotary_style = False
        quant_round_type=1
        quant_max_bound=126
        quant_min_bound=-126
        paddle_mmha_out = paddle._C_ops.masked_multihead_attention_(
            x,
            cache_kvs,
            paddle.zeros([bz,num_head,dim_head],dtype=x.dtype),
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_linear_shift,
            None,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_linear_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
        paddle_mmha_out=paddle_mmha_out[0]
        paddle_mmha_out=paddle_mmha_out.unsqueeze(0)
        paddle_mmha_out=paddle_mmha_out.reshape(output_size)
        # from paddle.incubate.nn.functional.masked_multihead_attention import masked_multihead_attention
        # paddle_mmha_out=masked_multihead_attention(
        #     x,
        #     cache_kv=cache_kvs,
        #     src_mask=src_mask,
        #     cum_offsets=cum_offsets,
        #     sequence_lengths=sequence_lengths,
        #     rotary_tensor=rotary_tensor,
        #     beam_cache_offset=beam_cache_offset,
        #     qkv_out_scale=qkv_out_scale,
        #     out_linear_shift=out_linear_shift,
        #     out_linear_smooth=None,
        #     seq_len=seq_len,
        #     rotary_emb_dims=rotary_emb_dims,
        #     use_neox_rotary_style=use_neox_rotary_style,
        #     out_linear_in_scale=out_linear_in_scale,
        #     quant_round_type=quant_round_type,
        #     quant_max_bound=quant_max_bound,
        #     quant_min_bound=quant_min_bound,
        # )
        # paddle_mmha_out=paddle_mmha_out[0]
        # paddle_mmha_out=paddle_mmha_out.unsqueeze(0)
        # paddle_mmha_out=paddle_mmha_out.reshape(output_size)
        return paddle_mmha_out
    elif "navie":
        cache_k, cache_v = paddle.split(cache_kv_out, 2, axis=0)
        k = paddle.to_tensor(np.expand_dims(k, axis=2))
        v = paddle.to_tensor(np.expand_dims(v, axis=2))
        q = paddle.to_tensor(np.expand_dims(q, axis=2))
        k = paddle.concat([cache_k.squeeze(0), k], axis=2)
        v = paddle.concat([cache_v.squeeze(0), v], axis=2)
        for i in range(num_head):
            if(i<16):
                tmp = paddle.matmul(
                    x=q[:, i, :, :] * (dim_head**-0.5),
                    y=k[:, 0, :, :],
                    transpose_y=True,
                )  
            else:
                tmp = paddle.matmul(
                    x=q[:, i, :, :] * (dim_head**-0.5),
                    y=k[:, 1, :, :],
                    transpose_y=True,
                ) 
            if i == 0:
                product = tmp
            else:
                product = np.concatenate((product, tmp), axis=1)
        product = np.expand_dims(product, axis=2)
        product = product + src_mask

        product = paddle.nn.functional.softmax(product)
        for i in range(num_head):
            if(i<16):
                tmp = paddle.matmul(
                    product[:, i, :, :], v[:, 0, :, :]
                )
            else:
                tmp = paddle.matmul(
                    product[:, i, :, :], v[:, 1, :, :]
                )
            if i == 0:
                out = tmp
            else:
                out = np.concatenate((out, tmp), axis=1)
        out = np.expand_dims(out, axis=1)
        out=out.reshape(output_size)
        return out



class SelfAttention(nn.Layer):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMv2Config, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias_attr=config.add_bias_linear or config.add_qkv_bias,
        )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias_attr=config.add_bias_linear)

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return paddle.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True, is_first_forward=True,usekernel=True,cache_kvs=None):
        # hidden_states: [seq_length, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [seq_length, b, h] --> [seq_length, b, (np * 3 * hn)]

        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                axis=-1,
            )
            query_layer = query_layer.reshape(
                query_layer.shape[:-1] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
            )
            key_layer = key_layer.reshape(
                key_layer.shape[:-1] + [self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head]
            )
            value_layer = value_layer.reshape(
                value_layer.shape[:-1]
                + [self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head]
            )
        else:
            new_tensor_shape = mixed_x_layer.shape[:-1] + [
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            ]
            mixed_x_layer = mixed_x_layer.reshape(new_tensor_shape)

            # [seq_length, b, np, 3 * hn] --> 3 [seq_length, b, np, hn]
            (query_layer, key_layer, value_layer) = paddle.split(mixed_x_layer, 3, axis=-1)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
        # adjust key and value for inference

        k_before_concat=key_layer
        v_before_concat=value_layer
        kv_cache_before_concat=kv_cache
        if use_cache:
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                key_layer = paddle.concat((cache_k, key_layer), axis=0)
                value_layer = paddle.concat((cache_v, value_layer), axis=0)
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None
        
        #encoder使用
        k_after_tile=key_layer
        v_after_tile=value_layer
        #[mqa,mmhq,navie]
        mode="mqa"
        usekernel=True    
        if is_first_forward and usekernel:
            #query_layer[seq_l,batch_size,num_head,head_dim]
            #encoder
            #cache_kv
            dim_head=query_layer.shape[3]
            num_head=query_layer.shape[2]
            num_head_kv=key_layer.shape[2]
            seq_lens=key_layer.shape[0]
            k_out=key_layer.transpose([1,2,0,3]).cast('float16')
            v_out=value_layer.transpose([1,2,0,3]).cast('float16')
            write_cache_kv(k_out, v_out, cache_kvs, paddle.to_tensor([seq_lens]).cast("int32"))
            context_layer=use_kernel_encoder(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                num_head=num_head,
                num_head_kv=num_head_kv,
                dim_head=dim_head,
                cache_kvs=cache_kvs,
                attention_mask=attention_mask,
            )

        elif not is_first_forward and usekernel:
            #decoder 
            context_layer=use_kernel_decoder(
                q=query_layer,
                k=k_before_concat,
                v=v_before_concat,
                kv_cache=kv_cache_before_concat,
                num_head=self.num_attention_heads_per_partition,
                num_head_kv=self.num_multi_query_groups_per_partition,
                dim_head=self.hidden_size_per_attention_head,
                cache_kvs=cache_kvs,
                mode=mode
                )
        elif not usekernel :    
            if self.multi_query_attention:
                key_layer = key_layer.unsqueeze(-2)
                key_layer = key_layer.tile(
                    [1, 1, 1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, 1]
                )
                key_layer = key_layer.reshape(
                    key_layer.shape[:2] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
                )
                value_layer = value_layer.unsqueeze(-2)
                value_layer = value_layer.tile(
                    [1, 1, 1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, 1]
                )
                value_layer = value_layer.reshape(
                    value_layer.shape[:2] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
                )

            #做kernel的decoder   
            if use_cache and usekernel and is_first_forward and mode == "mmha":
                # decoder 阶段 需要tile后的k、v
                #key_layer [s,bz,num_head,dim_head]
                seq_lens=key_layer.shape[0]
                k_out=key_layer.transpose([1,2,0,3]).cast('float16')
                v_out=value_layer.transpose([1,2,0,3]).cast('float16')
                write_cache_kv(k_out, v_out, cache_kvs, paddle.to_tensor([seq_lens]).cast("int32"))


            # ==================================
            # core attention computation
            # ==================================
            #import pdb;pdb.set_trace()
            context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        # =================
        # Output. [seq_length, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


class MLP(nn.Layer):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMv2Config, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias_attr=self.add_bias)

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias_attr=self.add_bias,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # Special Slicing to accomodate Tensor Parallel
        # Even channels is ffc_fc, odd channels is gate
        ffn_fc = intermediate_parallel[..., 0::2]
        gate = intermediate_parallel[..., 1::2]
        intermediate_parallel = F.silu(ffn_fc) * gate
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Layer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMv2Config, layer_number):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

        # MLP
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
        is_first_forward=True,
        cache_kvs=None,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache,is_first_forward=is_first_forward,cache_kvs=cache_kvs
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = F.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = F.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMTransformer(nn.Layer):
    """Transformer class."""

    def __init__(self, config: ChatGLMv2Config):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_hidden_layers = config.num_hidden_layers
        
        self.max_sequence_length = config.max_sequence_length
        self.caches = []
        self.head_dim = config.kv_channels * config.num_attention_heads//config.num_attention_heads
        self.num_head = config.num_attention_heads
        self.num_head_kv=config.multi_query_group_num

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number)

        self.layers = nn.LayerList([build_layer(i + 1) for i in range(self.num_hidden_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
        is_first_forward: bool = True,
    ):
        #print(self.max_sequence_length)
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_hidden_layers)]
        if len(self.caches) == 0:
            paddle.set_default_dtype('float16')
            self.caches = [
                    paddle.fluid.layers.fill_constant_batch_size_like(
                        paddle.zeros([2, 1, self.num_head_kv, self.max_sequence_length, self.head_dim]),
                        shape=[2, -1, self.num_head_kv, self.max_sequence_length, self.head_dim],
                        input_dim_idx=0,
                        output_dim_idx=1,
                        value=0.,
                        dtype=paddle.get_default_dtype())for _ in range(self.num_hidden_layers)]
        presents = () if use_cache else None
        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)

            hidden_states, kv_cache = layer(
                hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_caches[index], use_cache=use_cache,is_first_forward=is_first_forward,cache_kvs=self.caches[index]
            )
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMv2PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = ChatGLMv2Config
    pretrained_resource_files_map = CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "chatglm_v2"

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = paddle.tril(paddle.ones([batch_size, seq_length, seq_length]))
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = paddle.concat(
                [paddle.ones([batch_size, seq_length, past_length]), full_attention_mask], axis=-1
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).astype("bool")
        full_attention_mask.unsqueeze(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).tile([batch_size, 1])
        return position_ids


class Embedding(nn.Layer):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMv2Config):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(config.padded_vocab_size, self.hidden_size)
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        # Data format change to avoid explicit tranposes
        # [batch_size, seq_length, hidden_size] --> [seq_length, batch_size, hidden_size].
        embeddings = embeddings.transpose([1, 0, 2])
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.astype("float32")
        return embeddings


@register_base_model
class ChatGLMv2Model(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config, empty_init=True):
        super().__init__(config)
        self.embedding = Embedding(config)

        # Rotary positional embeddings
        self.max_sequence_length = config.max_sequence_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2)
        self.encoder = GLMTransformer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def forward(
        self,
        input_ids,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        full_attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor, paddle.Tensor], ...]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_first_forward: bool = True,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.astype("bool").all()) or (
                past_key_values and seq_length != 1
            ):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        #import pdb;pdb.set_trace()
        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.max_sequence_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            is_first_forward=is_first_forward,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            # attentions=all_self_attentions,
        )


class ChatGLMv2ForConditionalGeneration(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config):
        super().__init__(config)

        self.max_sequence_length = config.max_sequence_length
        self.chatglm_v2 = ChatGLMv2Model(config)
    def update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs[1] if isinstance(outputs, tuple) else outputs["past_key_values"]
       # model_kwargs["caches"]=outputs[4] if isinstance(outputs, tuple) else outputs["caches"]

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            new_attention_mask = paddle.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
            model_kwargs["attention_mask"] = paddle.concat([attention_mask, new_attention_mask], axis=-1)

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = paddle.concat([position_ids, new_position_id], axis=-1)

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: paddle.Tensor,
        past_key_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        is_first_forward: bool = True,
        **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if not is_first_forward:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "is_first_forward":is_first_forward,
        }

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        is_first_forward: bool = True,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.chatglm_v2(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_first_forward=is_first_forward,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.chatglm_v2.output_layer(hidden_states)
        lm_logits = lm_logits.transpose([1, 0, 2])

        loss = None
        if labels is not None:
            lm_logits = lm_logits.astype("float32")

            # Shift so that tokens < n predict n and flatten the logits and labels
            shift_logits = lm_logits[..., :-1, :]
            shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
            shift_labels = labels[..., 1:].reshape([-1])

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            lm_logits = lm_logits.astype(hidden_states.dtype)
            loss = loss.astype(hidden_states.dtype)
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
