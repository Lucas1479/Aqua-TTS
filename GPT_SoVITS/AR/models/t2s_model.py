# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import math
import threading
from typing import List, Optional
import torch
from tqdm import tqdm

from AR.models.utils import make_pad_mask, make_pad_mask_left
from AR.models.utils import (
    topk_sampling,
    sample,
    logits_to_probs,
    multinomial_sample_one_no_sync,
    dpo_loss,
    make_reject_y,
    get_batch_logps
)
from AR.modules.embedding import SinePositionalEmbedding
from AR.modules.embedding import TokenEmbedding
from AR.modules.transformer import LayerNorm
from AR.modules.transformer import TransformerEncoder
from AR.modules.transformer import TransformerEncoderLayer
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}

# @torch.jit.script ## 使用的话首次推理会非常慢，而且推理速度不稳定
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask:Optional[torch.Tensor]=None, scale:Optional[torch.Tensor]=None) -> torch.Tensor:
    B, H, L, S =query.size(0), query.size(1), query.size(-2), key.size(-2)
    if scale is None:
        scale_factor = torch.tensor(1 / math.sqrt(query.size(-1)))
    else:
        scale_factor = scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask, 0)
        else:
            attn_mask[attn_mask!=float("-inf")] =0
            attn_mask[attn_mask==float("-inf")] =1
            attn_weight.masked_fill_(attn_mask, 0)

    return attn_weight @ value

@torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        x = F.relu(F.linear(x, self.w1, self.b1))
        x = F.linear(x, self.w2, self.b2)
        return x


@torch.jit.script
class T2SBlock:
    def __init__(
            self,
            num_heads,
            hidden_dim: int,
            mlp: T2SMLP,
            qkv_w,
            qkv_b,
            out_w,
            out_b,
            norm_w1,
            norm_b1,
            norm_eps1,
            norm_w2,
            norm_b2,
            norm_eps2,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

        self.false = torch.tensor(False, dtype=torch.bool)

    @torch.jit.ignore
    def to_mask(self, x:torch.Tensor, padding_mask:Optional[torch.Tensor]):
        if padding_mask is None:
            return x
        
        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask
        
    def process_prompt(self, x:torch.Tensor, attn_mask : torch.Tensor, padding_mask:Optional[torch.Tensor]=None, torch_sdpa:bool=True):

            
        q, k, v = F.linear(self.to_mask(x, padding_mask), self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]
        
        q = self.to_mask(q, padding_mask)
        k_cache = self.to_mask(k, padding_mask)
        v_cache = self.to_mask(v, padding_mask)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(self.to_mask(attn, padding_mask), self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache
    
    def decode_next_token(self, x:torch.Tensor, k_cache:torch.Tensor, v_cache:torch.Tensor, attn_mask:torch.Tensor=None, torch_sdpa:bool=True):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)
        
        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k_cache.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)


        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, (~attn_mask) if attn_mask is not None else None)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


# 🚀 新增：支持固定缓冲区的 T2SBlock（用于 CUDA Graph 优化）
@torch.jit.script
class T2SBlockWithStaticCache:
    """
    支持固定大小 KV Cache 的 T2SBlock
    使用索引写入策略，避免 torch.cat 导致的形状变化
    """
    def __init__(
            self,
            num_heads,
            hidden_dim: int,
            mlp: T2SMLP,
            qkv_w,
            qkv_b,
            out_w,
            out_b,
            norm_w1,
            norm_b1,
            norm_eps1,
            norm_w2,
            norm_b2,
            norm_eps2,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

    def process_prompt(self, x:torch.Tensor, attn_mask : torch.Tensor, padding_mask:Optional[torch.Tensor]=None, torch_sdpa:bool=True):
        # 🔧 修复：直接在函数内处理 padding_mask，避免 JIT 类型推断问题
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(-1)
            x_masked = x * padding_mask
        else:
            x_masked = x
        
        q, k, v = F.linear(x_masked, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]
        
        if padding_mask is not None:
            q = q * padding_mask
            k_cache = k * padding_mask
            v_cache = v * padding_mask
        else:
            k_cache = k
            v_cache = v

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        
        if padding_mask is not None:
            attn = F.linear(attn * padding_mask, self.out_w, self.out_b)
        else:
            attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache
    
    def decode_next_token_with_static_cache(
        self, 
        x: torch.Tensor, 
        k_cache: torch.Tensor,  # 固定大小 [batch, max_len, hidden]
        v_cache: torch.Tensor,  # 固定大小 [batch, max_len, hidden]
        current_len: int,  # 当前有效长度
        attn_mask: Optional[torch.Tensor] = None, 
        torch_sdpa: bool = True
    ):
        """
        使用固定缓冲区的解码函数
        关键：使用索引写入，避免 torch.cat 改变形状
        """
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        # 🚀 关键优化：使用索引写入，而不是 torch.cat
        # k_cache 和 v_cache 的形状保持不变
        k_cache[:, current_len:current_len + 1, :] = k
        v_cache[:, current_len:current_len + 1, :] = v
        new_len = current_len + 1
        
        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = new_len  # 只使用有效部分
        
        # 只使用有效长度的 cache
        k_valid = k_cache[:, :kv_len, :]
        v_valid = v_cache[:, :kv_len, :]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_valid.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_valid.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, (~attn_mask) if attn_mask is not None else None)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache, new_len


@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks : int, blocks: List[T2SBlock]):
        self.num_blocks : int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self, x:torch.Tensor, attn_mask : torch.Tensor,
        padding_mask : Optional[torch.Tensor]=None, 
        torch_sdpa:bool=True
        ):
        k_cache : List[torch.Tensor] = []
        v_cache : List[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask, padding_mask, torch_sdpa)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token(
        self, x:torch.Tensor, 
        k_cache: List[torch.Tensor], 
        v_cache: List[torch.Tensor], 
        attn_mask : torch.Tensor=None,
        torch_sdpa:bool=True
    ):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(x, k_cache[i], v_cache[i], attn_mask, torch_sdpa)
        return x, k_cache, v_cache


# 🚀 新增：支持固定缓冲区的 T2STransformer
@torch.jit.script
class T2STransformerWithStaticCache:
    """
    支持固定大小 KV Cache 的 Transformer
    配合 T2SBlockWithStaticCache 使用
    """
    def __init__(self, num_blocks: int, blocks: List[T2SBlockWithStaticCache]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self, 
        x: torch.Tensor, 
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None, 
        torch_sdpa: bool = True
    ):
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask, padding_mask, torch_sdpa)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token_with_static_cache(
        self, 
        x: torch.Tensor, 
        k_cache: List[torch.Tensor],  # 固定大小的缓冲区
        v_cache: List[torch.Tensor],  # 固定大小的缓冲区
        current_lens: List[int],  # 每层的当前有效长度
        attn_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True
    ):
        """
        使用固定缓冲区的解码函数
        """
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i], current_lens[i] = self.blocks[i].decode_next_token_with_static_cache(
                x, k_cache[i], v_cache[i], current_lens[i], attn_mask, torch_sdpa
            )
        return x, k_cache, v_cache, current_lens


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )
        
        # 🚀 KV Cache 静态化配置
        # True: 使用固定缓冲区策略（支持CUDA Graph）
        # False: 使用动态torch.cat策略（原始行为）
        self.use_static_kv_cache = torch.cuda.is_available()  # 默认在CUDA环境下启用
        
        # 🚀 CUDA Graph 分桶优化（Bucketing Strategy）
        # 依赖于 use_static_kv_cache = True
        # 环境变量 ENABLE_CUDA_GRAPH=1 可以启用（默认禁用）
        import os
        cuda_graph_env = os.environ.get('ENABLE_CUDA_GRAPH', '0')  # 默认禁用，可通过环境变量开启
        self.cuda_graph_enabled = self.use_static_kv_cache and (cuda_graph_env == '1')
        
        if not self.cuda_graph_enabled and self.use_static_kv_cache:
            print(f"⚠️ CUDA Graph 已禁用（ENABLE_CUDA_GRAPH={cuda_graph_env}），使用静态KV Cache但允许并发")
        
        # 🔒 CUDA Graph 并发锁
        # CUDA Graph 不支持多线程并发 replay，需要加锁保护
        import threading
        self.cuda_graph_lock = threading.Lock()
        self._bucket_locks = {}
        self._bucket_locks_guard = threading.Lock()
        
        # 定义桶大小：覆盖常见的文本+prompt长度
        # 根据实际场景调整，这里选择 [128, 256, 512, 768, 1024]
        self.kv_cache_buckets = [128, 256, 512, 768, 1024] if self.use_static_kv_cache else []
        
        # 每个桶维护独立的 CUDA Graph 和静态缓冲区
        self.bucket_graphs = {}  # {bucket_size: cuda_graph}
        self.bucket_static_inputs = {}  # {bucket_size: {input_tensors}}
        self.bucket_static_outputs = {}  # {bucket_size: {output_tensors}}
        self.bucket_k_cache_buffers = {}  # {bucket_size: [k_cache per layer]}
        self.bucket_v_cache_buffers = {}  # {bucket_size: [v_cache per layer]}
        
        # 预热参数
        self.cuda_graph_warmup_steps = 3
        
        # 性能统计
        self.cuda_graph_stats = {
            "total_steps": 0,
            "graph_replay_steps": 0,
            "bucket_hits": {},  # {bucket_size: hit_count}
            "bucket_misses": 0,
            "capture_time": {},  # {bucket_size: time}
            "warmup_time": {},  # {bucket_size: time}
        }

        blocks = []
        blocks_static = []  # 🚀 新增：静态缓存版本的 blocks

        for i in range(self.num_layers):
            layer = self.h.layers[i]
            t2smlp = T2SMLP(
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias
            )

            # 原始版本的 block
            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps
            )
            blocks.append(block)
            
            # 🚀 新增：静态缓存版本的 block（共享相同的权重）
            block_static = T2SBlockWithStaticCache(
                self.num_head,
                self.model_dim,
                t2smlp,  # 共享 MLP
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps
            )
            blocks_static.append(block_static)
        
        # 创建两套 transformer
        self.t2s_transformer = T2STransformer(self.num_layers, blocks)
        self.t2s_transformer_static = T2STransformerWithStaticCache(self.num_layers, blocks_static)

    def _select_bucket(self, kv_cache_len):
        """
        根据 KV cache 的实际长度，选择最小的合适桶
        如果没有合适的桶，返回 None（使用正常模式）
        """
        if not self.kv_cache_buckets:
            return None
        
        for bucket_size in self.kv_cache_buckets:
            if kv_cache_len <= bucket_size:
                return bucket_size
        
        # 超出最大桶，无法使用 CUDA Graph
        return None
    
    def _pad_kv_cache_to_bucket(self, k_cache, v_cache, bucket_size):
        """
        将 KV cache 填充到指定桶大小
        返回填充后的 k_cache, v_cache 和有效长度
        """
        current_len = k_cache[0].shape[1]  # 获取第一层的 cache 长度
        
        if current_len >= bucket_size:
            # 不需要填充
            return k_cache, v_cache, current_len
        
        # 计算需要填充的长度
        pad_len = bucket_size - current_len
        
        # 为每一层填充
        padded_k_cache = []
        padded_v_cache = []
        
        for k, v in zip(k_cache, v_cache):
            # 在序列维度（dim=1）上填充 0
            k_padded = F.pad(k, (0, 0, 0, pad_len), value=0.0)
            v_padded = F.pad(v, (0, 0, 0, pad_len), value=0.0)
            padded_k_cache.append(k_padded)
            padded_v_cache.append(v_padded)
        
        return padded_k_cache, padded_v_cache, current_len
    
    def _get_bucket_lock(self, bucket_size: Optional[int]):
        """
        获取指定桶的锁，未启用桶时退回到全局锁
        """
        if bucket_size is None:
            return self.cuda_graph_lock
        with self._bucket_locks_guard:
            lock = self._bucket_locks.get(bucket_size)
            if lock is None:
                lock = threading.Lock()
                self._bucket_locks[bucket_size] = lock
        return lock

    def _warmup_and_capture_bucket(self, bucket_size, device):
        """
        为指定的桶大小预热并捕获 CUDA Graph
        🔧 关键：使用静态版本的 transformer（不使用 torch.cat）
        """
        import time
        
        print(f"🔥 开始为桶大小 {bucket_size} 预热和捕获 CUDA Graph...")
        warmup_start = time.perf_counter()
        
        try:
            # 创建模拟数据进行预热
            batch_size = 1
            hidden_dim = self.model_dim
            
            # 检测模型的实际 dtype
            model_dtype = next(self.ar_predict_layer.parameters()).dtype
            print(f"   📌 检测到模型 dtype: {model_dtype}")
            
            # 🚀 关键修复：使用固定缓冲区的 KV cache（不会 torch.cat）
            k_cache = [torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device) 
                      for _ in range(self.num_layers)]
            v_cache = [torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device) 
                      for _ in range(self.num_layers)]
            
            # 模拟一些初始内容（例如 prompt 后的状态）
            initial_len = bucket_size // 2  # 假设 prompt 占一半
            for i in range(self.num_layers):
                k_cache[i][:, :initial_len, :] = torch.randn(batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device)
                v_cache[i][:, :initial_len, :] = torch.randn(batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device)
            
            current_lens = [initial_len] * self.num_layers
            
            # 模拟 xy_pos
            xy_pos = torch.randn(batch_size, 1, hidden_dim, dtype=model_dtype, device=device)
            
            # 🚀 预热：使用静态版本的 transformer
            for warmup_idx in range(self.cuda_graph_warmup_steps):
                xy_dec, k_cache, v_cache, current_lens = self.t2s_transformer_static.decode_next_token_with_static_cache(
                    xy_pos, k_cache, v_cache, current_lens
                )
                logits = self.ar_predict_layer(xy_dec[:, -1])
            
            warmup_time = time.perf_counter() - warmup_start
            self.cuda_graph_stats["warmup_time"][bucket_size] = warmup_time
            print(f"✅ 桶 {bucket_size} 预热完成: {warmup_time:.4f}s")
            
            # 捕获阶段
            capture_start = time.perf_counter()
            
            # 重置为固定状态
            k_cache = [torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device) 
                      for _ in range(self.num_layers)]
            v_cache = [torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device) 
                      for _ in range(self.num_layers)]
            for i in range(self.num_layers):
                k_cache[i][:, :initial_len, :] = torch.randn(batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device)
                v_cache[i][:, :initial_len, :] = torch.randn(batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device)
            current_lens = [initial_len] * self.num_layers
            xy_pos = torch.randn(batch_size, 1, hidden_dim, dtype=model_dtype, device=device)
            
            # 捕获 CUDA Graph
            cuda_graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(cuda_graph, capture_error_mode='relaxed'):
                xy_dec, k_cache_out, v_cache_out, current_lens_out = self.t2s_transformer_static.decode_next_token_with_static_cache(
                    xy_pos, k_cache, v_cache, current_lens
                )
                logits = self.ar_predict_layer(xy_dec[:, -1])
            
            capture_time = time.perf_counter() - capture_start
            self.cuda_graph_stats["capture_time"][bucket_size] = capture_time
            
            # 存储到桶字典
            self.bucket_graphs[bucket_size] = cuda_graph
            self.bucket_static_inputs[bucket_size] = {
                'xy_pos': xy_pos,
                'k_cache': k_cache,
                'v_cache': v_cache,
                'current_lens': current_lens,
            }
            self.bucket_static_outputs[bucket_size] = {
                'xy_dec': xy_dec,
                'logits': logits,
            }
            
            # 初始化统计
            self.cuda_graph_stats["bucket_hits"][bucket_size] = 0
            
            print(f"📸 桶 {bucket_size} CUDA Graph 捕获成功: {capture_time:.4f}s")
            return True
            
        except Exception as e:
            print(f"❌ 桶 {bucket_size} CUDA Graph 捕获失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def precapture_cuda_graph(self, buckets: Optional[List[int]] = None):
        """
        手动预捕获指定桶的 CUDA Graph，用于首句前的预热
        """
        if not self.cuda_graph_enabled or not self.kv_cache_buckets:
            print("⚠️ 当前未启用 CUDA Graph 或没有可用桶，跳过预捕获")
            return {}

        device = next(self.parameters()).device
        target_buckets = buckets or [self.kv_cache_buckets[0]]
        results = {}

        for bucket in target_buckets:
            if bucket not in self.kv_cache_buckets:
                print(f"⚠️ 目标桶 {bucket} 不在配置列表中，跳过")
                results[bucket] = False
                continue

            lock = self._get_bucket_lock(bucket)
            with lock:
                if bucket in self.bucket_graphs:
                    print(f"ℹ️ 桶 {bucket} 已存在 CUDA Graph，直接复用")
                    results[bucket] = True
                    continue
                success = self._warmup_and_capture_bucket(bucket, device)
                results[bucket] = success
        return results

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)

        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        # x_attn_mask[:, x_len]=False
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )

        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)

        return xy_pos, xy_attn_mask, targets

    def forward(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """

        reject_y, reject_y_lens = make_reject_y(y, y_lens)

        xy_pos, xy_attn_mask, targets = self.make_input_data(x, x_lens, y, y_lens, bert_feature)

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        x_len = x_lens.max()
        logits = self.ar_predict_layer(xy_dec[:, x_len:])

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(x, x_lens, reject_y, reject_y_lens, bert_feature)

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )
        x_len = x_lens.max()
        reject_logits = self.ar_predict_layer(reject_xy_dec[:, x_len:])

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        loss_1 = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.permute(0, 2, 1).detach(), targets).item()

        A_logits, R_logits = get_batch_logps(logits, reject_logits, targets, reject_targets)
        loss_2, _, _ = dpo_loss(A_logits, R_logits, 0, 0, 0.2, reference_free=True)
        
        loss = loss_1 + loss_2

        return loss, acc

    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        loss = F.cross_entropy(logits, targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(
            self,
            x,
            x_lens,
            prompts,
            bert_feature,
            top_k: int = -100,
            early_stop_num: int = -1,
            temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        for _ in tqdm(range(1500)):
            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)
            # x 和逐渐增长的 y 一起输入给模型
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(
                y.device
            )

            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            # import os
            # os._exit(2333)
            y = torch.concat([y, samples], dim=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # 错位
        return targets[:, :-1], targets[:, 1:]

    def infer_panel_batch_infer(
        self,
        x:List[torch.LongTensor],  #####全部文本token
        x_lens:torch.LongTensor,
        prompts:torch.LongTensor,  ####参考音频token
        bert_feature:List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        if prompts is None:
            print("Warning: Prompt free is not supported batch_infer! switch to naive_infer")
            return self.infer_panel_naive_batched(x, x_lens, prompts, bert_feature, top_k=top_k, top_p=top_p, early_stop_num=early_stop_num, temperature=temperature, **kwargs)


        max_len = kwargs.get("max_len",x_lens.max())
        x_list = []
        for x_item, bert_item in zip(x, bert_feature):
            # max_len = max(max_len, x_item.shape[0], bert_item.shape[1])
            x_item = self.ar_text_embedding(x_item.unsqueeze(0))
            x_item = x_item + self.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position(x_item).squeeze(0)
            # x_item = F.pad(x_item,(0,0,0,max_len-x_item.shape[0]),value=0) if x_item.shape[0]<max_len else x_item  ### padding right
            x_item = F.pad(x_item,(0,0,max_len-x_item.shape[0],0),value=0) if x_item.shape[0]<max_len else x_item   ### padding left
            x_list.append(x_item)
        x:torch.Tensor = torch.stack(x_list, dim=0)


        # AR Decoder
        y = prompts
        
        x_len = x.shape[1]
        stop = False

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        assert y is not None, "Error: Prompt free is not supported batch_infer!"
        ref_free = False

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_lens = torch.LongTensor([y_emb.shape[1]]*y_emb.shape[0]).to(x.device)
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)



        ##### create mask #####
        bsz = x.shape[0]
        src_len = x_len + y_len
        y_paddind_mask = make_pad_mask_left(y_lens, y_len)
        x_paddind_mask = make_pad_mask_left(x_lens, max_len)
        
        # (bsz, x_len + y_len)
        padding_mask = torch.concat([x_paddind_mask, y_paddind_mask], dim=1)

        x_mask = F.pad(  
                    torch.zeros(x_len, x_len, dtype=torch.bool, device=x.device), 
                    (0, y_len),
                    value=True,
                )

        y_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1), 
            (x_len, 0),
            value=False,
        )
        
        causal_mask = torch.concat([x_mask, y_mask], dim=0).view(1 , src_len, src_len).repeat(bsz, 1, 1).to(x.device)
        # padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2) ### [b, x+y, x+y]
        ### 上面是错误的，会导致padding的token被"看见"

        # 正确的padding_mask应该是：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6]]

        padding_mask = padding_mask.view(bsz, 1, src_len).repeat(1, src_len, 1)

        attn_mask:torch.Tensor = causal_mask.logical_or(padding_mask)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1).bool()


        # 正确的attn_mask应该是这样的：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5,   6]]


        ###### decode #####
        y_list = [None]*y.shape[0]
        batch_idx_map = list(range(y.shape[0]))
        idx_list = [None]*y.shape[0]
        for idx in tqdm(range(1500)):
            if idx == 0:
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, attn_mask, None)
            else:
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache, attn_mask)
            logits = self.ar_predict_layer(
                xy_dec[:, -1]
            )

            if idx == 0:
                attn_mask = F.pad(attn_mask[:,:,-1].unsqueeze(-2),(0,1),value=False)
                logits = logits[:, :-1]
            else:
                attn_mask = F.pad(attn_mask,(0,1),value=False)

            samples = sample(
                    logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
                )[0]

            y = torch.concat([y, samples], dim=1)
            
            ####### 移除batch中已经生成完毕的序列,进一步优化计算量
            tokens = torch.argmax(logits, dim=-1)
            reserved_idx_of_batch_for_y = None
            if (self.EOS in samples[:, 0]) or \
                (self.EOS in tokens):  ###如果生成到EOS，则停止
                    l1 = samples[:, 0]==self.EOS
                    l2 = tokens==self.EOS
                    l = l1.logical_or(l2)
                    removed_idx_of_batch_for_y = torch.where(l==True)[0].tolist()
                    reserved_idx_of_batch_for_y = torch.where(l==False)[0]
                    # batch_indexs = torch.tensor(batch_idx_map, device=y.device)[removed_idx_of_batch_for_y]
                    for i in removed_idx_of_batch_for_y:
                        batch_index = batch_idx_map[i]
                        idx_list[batch_index] = idx
                        y_list[batch_index] = y[i, :-1]
                
                    batch_idx_map = [batch_idx_map[i] for i in reserved_idx_of_batch_for_y.tolist()]
                
            # 只保留batch中未生成完毕的序列 
            if reserved_idx_of_batch_for_y is not None:
                # index = torch.LongTensor(batch_idx_map).to(y.device)
                y = torch.index_select(y, dim=0, index=reserved_idx_of_batch_for_y)
                attn_mask = torch.index_select(attn_mask, dim=0, index=reserved_idx_of_batch_for_y)
                if k_cache is not None :
                    for i in range(len(k_cache)):
                        k_cache[i] = torch.index_select(k_cache[i], dim=0, index=reserved_idx_of_batch_for_y)
                        v_cache[i] = torch.index_select(v_cache[i], dim=0, index=reserved_idx_of_batch_for_y)
                
                
            if (early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num) or idx==1499:
                print("use early stop num:", early_stop_num)
                stop = True
                for i, batch_index in enumerate(batch_idx_map):
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]
                
            if not (None in idx_list):
                stop = True
                
            if stop:
                if y.shape[1]==0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to( dtype= y_emb.dtype,device=y_emb.device)            

        if (None in idx_list):
            for i in range(x.shape[0]):
                if idx_list[i] is None:
                    idx_list[i] = 1500-1  ###如果没有生成到EOS，就用最大长度代替
                    
        if ref_free:
            return y_list, [0]*x.shape[0]
        # print(idx_list)
        return y_list, idx_list
    
    def infer_panel_naive_batched(self,
        x:List[torch.LongTensor],  #####全部文本token
        x_lens:torch.LongTensor,
        prompts:torch.LongTensor,  ####参考音频token
        bert_feature:List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs
        ):
        y_list = []
        idx_list = []
        for i in range(len(x)):
            y, idx = self.infer_panel_naive(x[i].unsqueeze(0), 
                                                  x_lens[i], 
                                                  prompts[i].unsqueeze(0) if prompts is not None else None, 
                                                  bert_feature[i].unsqueeze(0), 
                                                  top_k, 
                                                  top_p, 
                                                  early_stop_num, 
                                                  temperature,
                                                  repetition_penalty,
                                                  **kwargs)
            y_list.append(y[0])
            idx_list.append(idx)
        
        return y_list, idx_list
    
    def infer_panel_naive(
        self,
        x:torch.LongTensor,  #####全部文本token
        x_lens:torch.LongTensor,
        prompts:torch.LongTensor,  ####参考音频token
        bert_feature:torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        # print(1111111,self.num_layers)

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_attn_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)\
                                                .unsqueeze(0)\
                                                .expand(bsz*self.num_head, -1, -1)\
                                                .view(bsz, self.num_head, src_len, src_len)\
                                                .to(device=x.device, dtype=torch.bool)

        enable_cuda_graph = kwargs.get("enable_cuda_graph", self.cuda_graph_enabled)
        enable_static_kv = kwargs.get("enable_static_kv", self.use_static_kv_cache)
        if enable_cuda_graph and not self.cuda_graph_enabled:
            enable_cuda_graph = False
        if not enable_static_kv:
            enable_cuda_graph = False
        graph_run_enabled = enable_cuda_graph and bool(self.kv_cache_buckets)

        # 🚀 KV Cache 策略选择和初始化
        current_bucket = None
        bucket_captured = False
        bucket_valid_len = None
        current_lens = None  # 用于静态缓存模式的长度追踪
        
        # 选择使用哪套 transformer
        static_transformer = self.t2s_transformer_static
        dynamic_transformer = self.t2s_transformer
        static_mode_active = enable_static_kv

        if static_mode_active:
            transformer = static_transformer
            print(f"🚀 使用静态KV Cache模式（支持CUDA Graph优化）")
        else:
            transformer = dynamic_transformer
            print(f"📌 使用动态KV Cache模式（原始torch.cat行为）")
        
        for idx in tqdm(range(1500)):
            if xy_attn_mask is not None:
                # 第一步：process_prompt
                xy_dec, k_cache, v_cache = transformer.process_prompt(xy_pos, xy_attn_mask, None)
                
                logits = self.ar_predict_layer(xy_dec[:, -1])
                
                # 如果使用静态缓存，初始化固定缓冲区
                if static_mode_active and k_cache is not None:
                    kv_cache_len = k_cache[0].shape[1]
                    selected_bucket = self._select_bucket(kv_cache_len)
                    
                    if selected_bucket is not None:
                        current_bucket = selected_bucket
                        
                        # 创建固定大小的缓冲区
                        batch_size = k_cache[0].shape[0]
                        hidden_dim = k_cache[0].shape[2]
                        device = k_cache[0].device
                        dtype = k_cache[0].dtype
                        
                        # 预分配固定缓冲区
                        k_cache_static = [
                            torch.zeros(batch_size, current_bucket, hidden_dim, dtype=dtype, device=device)
                            for _ in range(len(k_cache))
                        ]
                        v_cache_static = [
                            torch.zeros(batch_size, current_bucket, hidden_dim, dtype=dtype, device=device)
                            for _ in range(len(v_cache))
                        ]
                        
                        # 复制当前内容到缓冲区
                        for i in range(len(k_cache)):
                            k_cache_static[i][:, :kv_cache_len, :] = k_cache[i]
                            v_cache_static[i][:, :kv_cache_len, :] = v_cache[i]
                        
                        # 替换为静态缓冲区
                        k_cache = k_cache_static
                        v_cache = v_cache_static
                        current_lens = [kv_cache_len] * len(k_cache)
                        
                        print(f"✅ 初始化固定缓冲区: 桶大小={current_bucket}, 当前长度={kv_cache_len}")
                        
                        # 如果启用CUDA Graph，尝试捕获
                        if graph_run_enabled and current_bucket not in self.bucket_graphs:
                            bucket_lock = self._get_bucket_lock(current_bucket)
                            # 🔒 捕获时也需要加锁，避免多个线程同时尝试捕获同一个桶
                            with bucket_lock:
                                # 双重检查：可能另一个线程已经完成了捕获
                                if current_bucket not in self.bucket_graphs:
                                    success = self._warmup_and_capture_bucket(current_bucket, x.device)
                                    if success:
                                        bucket_captured = True
                                        print(f"📸 CUDA Graph 已捕获")
                                else:
                                    bucket_captured = True
                                    print(f"📸 CUDA Graph 已被另一线程捕获，直接使用")
                    else:
                        print(f"⚠️ KV cache 长度 {kv_cache_len} 超出所有桶，使用正常模式")
                        static_mode_active = False
                        transformer = dynamic_transformer
                        graph_run_enabled = False
                
            elif static_mode_active and current_bucket is not None and current_lens is not None:
                # 🚀 使用静态缓存模式
                
                # 🔧 关键修复：在写入前检查是否需要滑动窗口
                if current_lens[0] >= current_bucket - 1:
                    # 缓冲区即将满，使用滑动窗口：保留最新的 N-1 个 token
                    keep_len = current_bucket - 1
                    # 只在第一次触发时打印，避免日志过多
                    if not hasattr(self, '_sliding_window_triggered'):
                        self._sliding_window_triggered = True
                        print(f"⚠️ 缓冲区即将满（{current_lens[0]}/{current_bucket}），启动滑动窗口模式（保留最新 {keep_len} tokens）")
                    
                    # 移动 KV cache，丢弃最旧的 token
                    for i in range(len(k_cache)):
                        k_cache[i][:, :keep_len, :] = k_cache[i][:, -keep_len:, :].clone()
                        v_cache[i][:, :keep_len, :] = v_cache[i][:, -keep_len:, :].clone()
                        # 清零后面的部分
                        k_cache[i][:, keep_len:, :].zero_()
                        v_cache[i][:, keep_len:, :].zero_()
                    
                    # 重置 current_lens
                    current_lens = [keep_len] * len(current_lens)
                
                # 🚀 尝试使用 CUDA Graph（如果已捕获）
                if graph_run_enabled and bucket_captured and current_bucket in self.bucket_graphs:
                    replay_failed = False
                    bucket_lock = self._get_bucket_lock(current_bucket)
                    with bucket_lock:
                        try:
                            cuda_graph = self.bucket_graphs[current_bucket]
                            static_inputs = self.bucket_static_inputs[current_bucket]
                            static_outputs = self.bucket_static_outputs[current_bucket]
                            
                            static_inputs['xy_pos'].copy_(xy_pos)
                            for i in range(len(k_cache)):
                                actual_len = current_lens[i]
                                static_inputs['k_cache'][i][:, :actual_len, :].copy_(k_cache[i][:, :actual_len, :])
                                static_inputs['v_cache'][i][:, :actual_len, :].copy_(v_cache[i][:, :actual_len, :])
                            static_inputs['current_lens'] = current_lens.copy()
                            
                            cuda_graph.replay()
                            torch.cuda.synchronize()
                            
                            xy_dec = static_outputs['xy_dec']
                            logits = static_outputs['logits']
                            
                            for i in range(len(k_cache)):
                                k_cache[i].copy_(static_inputs['k_cache'][i])
                                v_cache[i].copy_(static_inputs['v_cache'][i])
                            current_lens = static_inputs['current_lens']
                            
                            self.cuda_graph_stats["graph_replay_steps"] += 1
                            
                            if not hasattr(self, '_cuda_graph_replay_started'):
                                self._cuda_graph_replay_started = True
                                print(f"♻️ 开始使用 CUDA Graph 加速（桶 {current_bucket}）")
                        except RuntimeError as e:
                            replay_failed = True
                            graph_run_enabled = False
                            bucket_captured = False
                            print(f"⚠️ CUDA Graph 重放失败，降级为静态模式：{repr(e)}")
                    if replay_failed:
                        xy_dec, k_cache, v_cache, current_lens = transformer.decode_next_token_with_static_cache(
                            xy_pos, k_cache, v_cache, current_lens
                        )
                        logits = self.ar_predict_layer(xy_dec[:, -1])
                    
                else:
                    # 正常静态缓存模式（未使用 CUDA Graph）
                    xy_dec, k_cache, v_cache, current_lens = transformer.decode_next_token_with_static_cache(
                        xy_pos, k_cache, v_cache, current_lens
                    )
                    logits = self.ar_predict_layer(xy_dec[:, -1])
                
            else:
                # 正常模式（动态KV Cache）
                if transformer is not dynamic_transformer:
                    transformer = dynamic_transformer
                xy_dec, k_cache, v_cache = transformer.decode_next_token(xy_pos, k_cache, v_cache)
                logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                xy_attn_mask = None
            if(idx<11):###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]

            y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                
                # 打印 CUDA Graph 分桶性能统计
                if graph_run_enabled and bucket_captured and current_bucket is not None:
                    self.cuda_graph_stats["total_steps"] = idx + 1
                    graph_ratio = self.cuda_graph_stats["graph_replay_steps"] / self.cuda_graph_stats["total_steps"] * 100
                    print(f"📊 CUDA Graph 分桶统计:")
                    print(f"   - 总步数: {self.cuda_graph_stats['total_steps']}")
                    print(f"   - 图复用: {self.cuda_graph_stats['graph_replay_steps']} ({graph_ratio:.1f}%)")
                    print(f"   - 使用桶: {current_bucket}")
                    print(f"   - 桶命中: {self.cuda_graph_stats['bucket_hits'].get(current_bucket, 0)}")
                    print(f"   - 桶未命中: {self.cuda_graph_stats['bucket_misses']}")
                    if current_bucket in self.cuda_graph_stats['warmup_time']:
                        print(f"   - 预热时间: {self.cuda_graph_stats['warmup_time'][current_bucket]:.3f}s")
                    if current_bucket in self.cuda_graph_stats['capture_time']:
                        print(f"   - 捕获时间: {self.cuda_graph_stats['capture_time'][current_bucket]:.3f}s")
                break

            ####################### update next step ###################################
            self.cuda_graph_stats["total_steps"] = idx + 1
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(dtype=y_emb.dtype,device=y_emb.device)

        if ref_free:
            return y[:, :-1], 0
        return y[:, :-1], idx
    
    
    def infer_panel(
        self,
        x:torch.LongTensor,  #####全部文本token
        x_lens:torch.LongTensor,
        prompts:torch.LongTensor,  ####参考音频token
        bert_feature:torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs
    ):
        return self.infer_panel_naive(x, x_lens, prompts, bert_feature, top_k, top_p, early_stop_num, temperature, repetition_penalty, **kwargs)
