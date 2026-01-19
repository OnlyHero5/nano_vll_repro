"""Day 4 完整测试"""
import sys
sys.path.insert(0, '/home/psx/nano_vllm_repro/nano_vll_repro')

import torch


@torch.inference_mode()
def test_linear_layers():
    """测试融合 Linear 层"""
    from layers.linear import QKVLinear, MergedLinear, RowLinear
    
    # QKVLinear
    qkv = QKVLinear(512, num_heads=8, num_kv_heads=2, head_dim=64)
    
    q_weight = torch.randn(8 * 64, 512)
    k_weight = torch.randn(2 * 64, 512)
    v_weight = torch.randn(2 * 64, 512)
    
    qkv.weight.weight_loader(qkv.weight, q_weight, "q")
    qkv.weight.weight_loader(qkv.weight, k_weight, "k")
    qkv.weight.weight_loader(qkv.weight, v_weight, "v")
    
    assert torch.allclose(qkv.weight.data[:512], q_weight)
    assert torch.allclose(qkv.weight.data[512:640], k_weight)
    assert torch.allclose(qkv.weight.data[640:], v_weight)
    
    print("✅ QKVLinear 测试通过")
    
    # MergedLinear
    merged = MergedLinear(512, 1024, num_shards=2)
    gate = torch.randn(1024, 512)
    up = torch.randn(1024, 512)
    
    merged.weight.weight_loader(merged.weight, gate, 0)
    merged.weight.weight_loader(merged.weight, up, 1)
    
    assert torch.allclose(merged.weight.data[:1024], gate)
    assert torch.allclose(merged.weight.data[1024:], up)
    
    print("✅ MergedLinear 测试通过")


@torch.inference_mode()
def test_sampler():
    """测试 Sampler"""
    from layers.sampler import Sampler
    
    sampler = Sampler()
    logits = torch.randn(4, 1000)
    temps = torch.tensor([0.0, 0.5, 1.0, 2.0])
    
    tokens = sampler(logits, temps)
    
    assert tokens.shape == (4,)
    assert tokens[0] == logits[0].argmax()  # greedy
    
    print("✅ Sampler 测试通过")


@torch.inference_mode()
def test_sequence_attributes():
    """测试 Sequence 属性访问"""
    from engine.sequence import Sequence
    from sampling_params import SamplingParams
    
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=0.7))
    
    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.last_token == 5
    assert seq.num_tokens == 5
    assert seq.num_prompt_tokens == 5
    assert seq.prompt_token_ids == [1, 2, 3, 4, 5]
    assert seq.completion_token_ids == []
    
    seq.append_token(6)
    assert seq.token_ids == [1, 2, 3, 4, 5, 6]
    assert seq.last_token == 6
    assert seq.num_tokens == 6
    assert seq.num_completion_tokens == 1
    
    print("✅ Sequence 属性测试通过")


if __name__ == "__main__":
    test_linear_layers()
    test_sampler()
    test_sequence_attributes()
    print("\n🎉 Day 4 所有测试通过！")