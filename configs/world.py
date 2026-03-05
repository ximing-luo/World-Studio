from dataclasses import dataclass

@dataclass
class WorldConfig():
    """
    世界模型核心配置 (World Model Core Configuration)
    基于 MLA + MoE 架构，专门为时空预测任务优化。
    适配项目设计哲学：写死专家参数，统一世界模型规格，降低外部认知负担。
    """
    # --- 基础架构配置 ---
    hidden_dim: int = 576          # 隐藏层维度
    n_layer: int = 8               # 层数
    n_head: int = 8                # 注意力头数
    n_kv_head: int = 2             # KV 头数
    max_seq_len: int = 1024        # 最大序列长度
    dropout: float = 0.1           # Dropout 概率
    bias: bool = False             # 是否使用偏置
    
    # --- MLA (Multi-Head Latent Attention) 配置 ---
    kv_lora_rank: int = 32         # KV 压缩秩
    q_lora_rank: int = 32          # Query 压缩秩
    qk_rope_head_dim: int = 64     # RoPE 部分维度 (rope_head_dim)
    qk_nope_head_dim: int = 64     # 非 RoPE 部分维度 (q_head_dim/kv_head_dim)
    v_head_dim: int = 64           # Value 投影维度

    # --- RoPE / YaRN 配置 ---
    rope_base: float = 10000.0     # RoPE 基数
    rope_scale: float = 1.0        # YaRN 插值/NTK 扩展倍数 (1.0 代表不扩展)
    
    # --- MoE (Mixture of Experts) 配置 ---
    num_experts: int = 8           # 总专家数
    num_experts_per_tok: int = 2   # 每个 Token 激活的专家数
    num_shared_experts: int = 1    # 共享专家数
    intermediate_size: int = None  # FFN 中间层维度 (None 则ffn自动计算)
    router_aux_loss_coef: float = 0.01 # 辅助损失系数
    bias_update_rate: float = 0.001 # DeepSeek-V3 动态偏置更新率
    is_causal: bool = True          # 是否为因果注意力 (默认为 True)

@dataclass
class VisionThinkingConfig(WorldConfig):
    """
    编码器配置：侧重空间语义提纯
    - 更多注意力头，捕捉并行空间关系
    - 非因果律，允许全局空间查看
    """
    n_head: int = 12
    kv_lora_rank: int = 64
    num_experts: int = 4
    is_causal: bool = False  # 空间理解不需要因果掩码

@dataclass
class PredictorConfig(WorldConfig):
    """
    预测器配置：侧重物理规律模拟
    - 极低 KV 压缩秩，优化长序列自回归
    - 海量专家，存储复杂的物理规律
    - 严格因果律
    """
    n_head: int = 8
    kv_lora_rank: int = 16    # 极致压缩 KV Cache
    num_experts: int = 16     # 更多专家用于模拟物理世界
    max_seq_len: int = 2048   # 更长的时序上下文
    is_causal: bool = True    # 时间预测必须因果