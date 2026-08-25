import os

# 端到端 benchmark 的开关：NANOVLLM_TRITON=0 走原来的 torch.compile 实现作为基线
USE_TRITON = os.getenv("NANOVLLM_TRITON", "1") == "1"
