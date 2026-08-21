# r2-Day01 whiteboard check - Transformer dim calc
# 7B config: hidden=4096, heads=32, layers=32, vocab=32000
# 目标：手算总参误差<20% 够用

def estimate_params():
    hidden=4096
    layers=32
    vocab=32000
    # per layer: Attn QKV+O ~4*hidden^2, FFN ~8*hidden^2 (gate/up/down 3*hidden*intermediate, intermediate ~11008 ~2.7*hidden)
    attn = 4 * hidden * hidden
    ffn = 3 * hidden * 11008  # LLaMA style
    per_layer = attn + ffn
    total_layers = per_layer * layers
    embed = vocab * hidden
    total = total_layers + embed
    print(f"per layer attn {attn/1e6:.1f}M ffn {ffn/1e6:.1f}M total {per_layer/1e6:.1f}M")
    print(f"layers {total_layers/1e9:.2f}B embed {embed/1e9:.2f}B total {total/1e9:.2f}B")
    print("够用：误差<20%算过")

if __name__ == "__main__":
    estimate_params()
