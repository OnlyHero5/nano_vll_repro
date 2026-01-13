"""Day 3 测试脚本 - PagedAttention 和 Block Manager"""

import sys
sys.path.insert(0, '.')

import torch
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import Block, BlockManager
from layers.attention import Attention, store_kvcache
from utils.context import set_context, reset_context
from sampling_params import SamplingParams


@torch.inference_mode()
def test_block():
    """测试 Block 类"""
    print("=" * 50)
    print("测试 Block")
    print("=" * 50)
    
    block = Block(block_id=0)
    print(f"初始状态: {block}")
    assert block.ref_count == 0
    assert block.hash == -1
    
    # 模拟分配
    block.reset()
    print(f"分配后: {block}")
    assert block.ref_count == 1
    
    # 模拟更新哈希
    token_ids = [100, 200, 300, 400]
    block.update(hash_value=12345, token_ids=token_ids)
    print(f"更新哈希后: {block}")
    assert block.hash == 12345
    assert block.token_ids == token_ids
    
    print("✅ Block 测试通过!\n")


@torch.inference_mode()
def test_block_manager_basic():
    """测试 BlockManager 基础功能"""
    print("=" * 50)
    print("测试 BlockManager 基础功能")
    print("=" * 50)
    
    num_blocks = 10
    block_size = 4  # 小尺寸便于测试
    
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    print(f"初始状态: {manager}")
    assert manager.get_num_free_blocks() == num_blocks
    
    # 创建序列
    token_ids = [1, 2, 3, 4, 5, 6, 7]  # 7 tokens, 需要 2 blocks
    seq = Sequence(token_ids, SamplingParams())
    seq.block_size = block_size  # 覆盖默认的 256
    
    print(f"序列需要 {seq.num_blocks} 个 blocks")
    
    # 检查是否可以分配
    assert manager.can_allocate(seq)
    
    # 分配
    manager.allocate(seq)
    print(f"分配后: {manager}")
    print(f"序列 block_table: {seq.block_table}")
    
    assert len(seq.block_table) == 2
    assert manager.get_num_free_blocks() == num_blocks - 2
    
    # 释放
    manager.deallocate(seq)
    print(f"释放后: {manager}")
    assert manager.get_num_free_blocks() == num_blocks
    assert len(seq.block_table) == 0
    
    print("✅ BlockManager 基础功能测试通过!\n")


@torch.inference_mode()
def test_block_manager_append():
    """测试 BlockManager append_slot 功能"""
    print("=" * 50)
    print("测试 BlockManager append_slot")
    print("=" * 50)
    
    num_blocks = 10
    block_size = 4
    
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    
    # 创建初始序列
    token_ids = [1, 2, 3]  # 3 tokens, 1 block
    seq = Sequence(token_ids, SamplingParams())
    seq.block_size = block_size
    
    manager.allocate(seq)
    print(f"初始: {len(seq)} tokens, {len(seq.block_table)} blocks")
    
    # 模拟 decode：追加 tokens
    for new_token in [4, 5, 6, 7, 8]:
        seq.append_token(new_token)
        manager.append_slot(seq)
        print(f"追加 token {new_token}: {len(seq)} tokens, {len(seq.block_table)} blocks")
    
    assert len(seq.block_table) == 2  # 8 tokens = 2 blocks
    
    print("✅ BlockManager append_slot 测试通过!\n")


@torch.inference_mode()
def test_slot_mapping():
    """测试 slot mapping 计算"""
    print("=" * 50)
    print("测试 Slot Mapping")
    print("=" * 50)
    
    num_blocks = 10
    block_size = 4
    
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    
    # 创建序列
    token_ids = [1, 2, 3, 4, 5, 6]
    seq = Sequence(token_ids, SamplingParams())
    seq.block_size = block_size
    
    manager.allocate(seq)
    
    # 计算 slot mapping
    slots = manager.get_slot_mapping(seq)
    print(f"Token IDs: {token_ids}")
    print(f"Block Table: {seq.block_table}")
    print(f"Slot Mapping: {slots}")
    
    # 验证 slot 计算
    for i, slot in enumerate(slots):
        block_idx = i // block_size
        offset = i % block_size
        expected_slot = seq.block_table[block_idx] * block_size + offset
        assert slot == expected_slot, f"Token {i}: expected {expected_slot}, got {slot}"
    
    print("✅ Slot Mapping 测试通过!\n")


@torch.inference_mode()
def test_prefix_cache():
    """测试 Prefix Caching"""
    print("=" * 50)
    print("测试 Prefix Caching")
    print("=" * 50)
    
    num_blocks = 20
    block_size = 4
    
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    
    # 第一个序列
    prefix_tokens = [100, 200, 300, 400]  # 完整的一个 block
    seq1_tokens = prefix_tokens + [1, 2]
    seq1 = Sequence(seq1_tokens, SamplingParams())
    seq1.block_size = block_size
    
    manager.allocate(seq1)
    print(f"Seq1 block_table: {seq1.block_table}")
    print(f"Seq1 num_cached_tokens: {seq1.num_cached_tokens}")
    
    # 第二个序列（共享前缀）
    seq2_tokens = prefix_tokens + [3, 4, 5]
    seq2 = Sequence(seq2_tokens, SamplingParams())
    seq2.block_size = block_size
    
    manager.allocate(seq2)
    print(f"Seq2 block_table: {seq2.block_table}")
    print(f"Seq2 num_cached_tokens: {seq2.num_cached_tokens}")
    
    # 验证：第一个 block 应该被共享
    assert seq1.block_table[0] == seq2.block_table[0], "First block should be shared!"
    assert seq2.num_cached_tokens == block_size, "Should have cached the prefix"
    
    # 验证引用计数
    shared_block_id = seq1.block_table[0]
    assert manager.blocks[shared_block_id].ref_count == 2
    
    print("✅ Prefix Caching 测试通过!\n")


@torch.inference_mode()
def test_attention_with_context():
    """测试 Attention 层与 Context"""
    print("=" * 50)
    print("测试 Attention 与 Context")
    print("=" * 50)
    
    device = "cuda"
    dtype = torch.bfloat16

    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    
    attn = Attention(
        num_heads=num_heads,
        head_dim=head_dim,
        scale=head_dim ** -0.5,
        num_kv_heads=num_kv_heads,
    )
    
    # 测试 Prefill
    num_tokens = 5
    q = torch.randn(num_tokens, num_heads, head_dim, device=device,dtype=dtype)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device,dtype=dtype)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device,dtype=dtype)
    
    # 设置 Context（无 KV Cache 的简单情况）
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=num_tokens,
        slot_mapping=torch.arange(num_tokens),
    )
    
    # Forward（不使用 FlashAttention，使用 PyTorch fallback）
    output = attn(q, k, v)
    print(f"Prefill 输入 Q: {q.shape}")
    print(f"Prefill 输出: {output.shape}")
    assert output.shape == (num_tokens, num_heads, head_dim)
    
    reset_context()
    print("✅ Attention 与 Context 测试通过!\n")


@torch.inference_mode()
def test_store_kvcache():
    """测试 KV Cache 存储"""
    print("=" * 50)
    print("测试 store_kvcache")
    print("=" * 50)
    
    num_tokens = 6
    num_blocks = 4
    block_size = 4
    num_kv_heads = 2
    head_dim = 8
    
    # 创建测试数据
    key = torch.randn(num_tokens, num_kv_heads, head_dim).cuda()
    value = torch.randn(num_tokens, num_kv_heads, head_dim).cuda()
    
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim).cuda()
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim).cuda()
    
    # slot mapping: 假设 tokens 分布在 block 1 (slots 4-7) 和 block 2 (slots 8-9)
    slot_mapping = torch.tensor([4, 5, 6, 7, 8, 9], device='cuda')
    
    # 存储
    store_kvcache(key, value, k_cache, v_cache, slot_mapping)
    
    # 验证
    k_cache_flat = k_cache.view(-1, num_kv_heads, head_dim)
    v_cache_flat = v_cache.view(-1, num_kv_heads, head_dim)
    
    for i, slot in enumerate(slot_mapping.tolist()):
        assert torch.allclose(k_cache_flat[slot], key[i]), f"Key mismatch at slot {slot}"
        assert torch.allclose(v_cache_flat[slot], value[i]), f"Value mismatch at slot {slot}"
    
    print(f"存储了 {num_tokens} 个 token 的 KV")
    print(f"K Cache 非零 slots: {(k_cache_flat.abs().sum(dim=(1,2)) > 0).sum().item()}")
    
    print("✅ store_kvcache 测试通过!\n")


if __name__ == "__main__":
    test_block()
    test_block_manager_basic()
    test_block_manager_append()
    test_slot_mapping()
    test_prefix_cache()
    test_attention_with_context()
    
    # 需要 GPU 的测试
    if torch.cuda.is_available():
        test_store_kvcache()
    else:
        print("⚠️ 跳过 GPU 测试 (store_kvcache)")
    
    print("=" * 50)
    print("🎉 Day 3 所有测试通过!")
    print("=" * 50)