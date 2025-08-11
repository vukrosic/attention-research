import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    batch_size: int = 32
    max_steps: int = 2000

    # Training parameters
    gradient_accumulation_steps: int = 2
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 4  # a b = result
    num_samples: int = 50000
    vocab_size: int = 202  # -100 to 100 (201 numbers) + EQUALS token

    # Evaluation
    eval_every: int = 400
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def generate_arithmetic_data(config: ModelConfig):
    """Generate arithmetic with clear structure"""
    print(f"🔢 Generating {config.num_samples} arithmetic samples")
    
    # Special tokens
    EQUALS = 201  # "=" token
    
    def num_to_token(num):
        return num + 100
    
    def token_to_num(token):
        if token == EQUALS:
            return None
        return token - 100
    
    sequences = []
    for _ in range(config.num_samples):
        a = random.randint(-100, 100)  # Start small
        b = random.randint(-100, 100)
        result = a + b
        
        # Clamp result to valid range
        result = max(-100, min(100, result))
        
        # Sequence: a b EQUALS result
        seq = [num_to_token(a), num_to_token(b), EQUALS, num_to_token(result)]
        sequences.append(seq)
    
    print(f"✅ Generated {len(sequences)} arithmetic sequences")
    return sequences, num_to_token, token_to_num, EQUALS

class ArithmeticDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Model sees: a b =, predicts: b = result
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        seq_len = x_BTHD.size(2)  # B, H, T, D format
        assert self.cos.size(0) >= seq_len
        cos, sin = self.cos[None, None, :seq_len, :], self.sin[None, None, :seq_len, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x, return_attention=False):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        if return_attention:
            # Manual attention computation to get weights
            scale = 1.0 / math.sqrt(self.d_k)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            
            # Apply causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            return self.w_o(attn_output), attn_weights
        else:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attention(self.norm1(x), return_attention=True)
            x = x + self.dropout(attn_out)
            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)
            return x, attn_weights
        else:
            attn_out = self.attention(self.norm1(x))
            x = x + self.dropout(attn_out)
            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)
            return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_attention=False):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                x, attn_weights = block(x, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        if return_attention:
            return logits, attention_weights
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with Muon optimizer"""
    print(f"\n🚀 Training Small model with Muon optimizer")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

def get_attention_weights(model, input_seq):
    """Extract attention weights from all layers"""
    model.eval()
    device = next(model.parameters()).device
    input_seq = input_seq.to(device)
    
    with torch.no_grad():
        _, attention_weights = model(input_seq, return_attention=True)
    
    return attention_weights

def visualize_attention_patterns(model, test_cases, token_to_num, num_to_token, EQUALS):
    """Visualize attention patterns for different arithmetic categories"""
    
    # Define test categories
    patterns = {
        'small': [(5, 3), (2, 7), (-4, 6)],
        'large': [(85, 90), (-95, -88), (99, 98)],
        'opposite': [(50, -50), (25, -25), (-30, 30)],
        'with_zero': [(0, 5), (0, 0), (100, 0)],
        'boundary': [(100, 100), (-100, -100), (99, 1)]
    }
    
    device = next(model.parameters()).device
    model.eval()
    
    fig, axes = plt.subplots(len(patterns), model.config.n_layers, figsize=(3*model.config.n_layers, 3*len(patterns)))
    if model.config.n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for cat_idx, (category, cases) in enumerate(patterns.items()):
        print(f"\n📊 Analyzing {category} patterns...")
        
        # Average attention across cases in this category
        avg_attention_per_layer = []
        
        for a, b in cases:
            # Create input sequence: [a, b, =]
            input_seq = torch.tensor([num_to_token(a), num_to_token(b), EQUALS], 
                                   dtype=torch.long).unsqueeze(0).to(device)
            
            attention_weights = get_attention_weights(model, input_seq)
            
            if not avg_attention_per_layer:
                avg_attention_per_layer = [torch.zeros_like(attn) for attn in attention_weights]
            
            for layer_idx, attn in enumerate(attention_weights):
                avg_attention_per_layer[layer_idx] += attn
        
        # Average across cases
        for layer_idx in range(len(avg_attention_per_layer)):
            avg_attention_per_layer[layer_idx] /= len(cases)
        
        # Plot heatmaps for each layer
        for layer_idx, avg_attn in enumerate(avg_attention_per_layer):
            # Average across heads and batch
            attn_matrix = avg_attn[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
            
            ax = axes[cat_idx, layer_idx]
            sns.heatmap(attn_matrix, annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=['a', 'b', '='], yticklabels=['a', 'b', '='], ax=ax)
            ax.set_title(f'{category}\nLayer {layer_idx+1}')
            
            if layer_idx == 0:
                ax.set_ylabel('Query Position')
            if cat_idx == len(patterns) - 1:
                ax.set_xlabel('Key Position')
    
    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_attention_stats(model, data_loader, config):
    """Compute attention statistics across the dataset"""
    model.eval()
    device = next(model.parameters()).device
    
    stats = {
        'entropy_per_layer': [[] for _ in range(config.n_layers)],
        'position_bias': [[] for _ in range(config.n_layers)],
        'attention_to_equals': [[] for _ in range(config.n_layers)]
    }
    
    print("📈 Computing attention statistics...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= 50:  # Limit for speed
                break
                
            x = x.to(device)
            attention_weights = get_attention_weights(model, x)
            
            for layer_idx, attn in enumerate(attention_weights):
                # attn shape: [batch, heads, seq_len, seq_len]
                batch_size, n_heads, seq_len, _ = attn.shape
                
                for b in range(batch_size):
                    for h in range(n_heads):
                        # Focus on the last position (where we predict the result)
                        last_pos_attn = attn[b, h, -1, :]  # Attention from last position
                        
                        # Compute entropy
                        entropy = -torch.sum(last_pos_attn * torch.log(last_pos_attn + 1e-8))
                        stats['entropy_per_layer'][layer_idx].append(entropy.item())
                        
                        # Position bias (how much attention goes to each position)
                        stats['position_bias'][layer_idx].append(last_pos_attn.cpu().numpy())
                        
                        # Attention to equals token (position 2)
                        stats['attention_to_equals'][layer_idx].append(last_pos_attn[2].item())
    
    # Compute averages
    for layer_idx in range(config.n_layers):
        avg_entropy = np.mean(stats['entropy_per_layer'][layer_idx])
        avg_pos_bias = np.mean(stats['position_bias'][layer_idx], axis=0)
        avg_equals_attn = np.mean(stats['attention_to_equals'][layer_idx])
        
        print(f"Layer {layer_idx+1}:")
        print(f"  Average entropy: {avg_entropy:.3f}")
        print(f"  Position bias: a={avg_pos_bias[0]:.3f}, b={avg_pos_bias[1]:.3f}, ={avg_pos_bias[2]:.3f}")
        print(f"  Attention to '=': {avg_equals_attn:.3f}")
    
    return stats

def analyze_head_specialization(model, config, num_to_token, token_to_num, EQUALS):
    """Check if specific attention heads specialize in different patterns"""
    model.eval()
    device = next(model.parameters()).device
    
    # Test patterns
    test_patterns = {
        'same_numbers': [(5, 5), (10, 10), (-7, -7)],
        'opposite_signs': [(5, -5), (10, -10), (25, -25)],
        'with_zero': [(0, 5), (0, -3), (7, 0)],
        'large_sums': [(50, 50), (75, 25), (60, 40)],
        'small_sums': [(1, 2), (3, 4), (2, 1)]
    }
    
    print("🔍 Analyzing head specialization...")
    
    head_responses = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad():
        for pattern_name, cases in test_patterns.items():
            for a, b in cases:
                input_seq = torch.tensor([num_to_token(a), num_to_token(b), EQUALS], 
                                       dtype=torch.long).unsqueeze(0).to(device)
                
                attention_weights = get_attention_weights(model, input_seq)
                
                for layer_idx, attn in enumerate(attention_weights):
                    # attn shape: [1, heads, seq_len, seq_len]
                    for head_idx in range(config.n_heads):
                        # Get attention from result position to operands
                        result_attn = attn[0, head_idx, -1, :]  # Last position attention
                        
                        # Store key metrics for this head
                        head_responses[f'L{layer_idx+1}H{head_idx+1}'][pattern_name].append({
                            'attn_to_a': result_attn[0].item(),
                            'attn_to_b': result_attn[1].item(),
                            'attn_to_equals': result_attn[2].item(),
                            'max_attn_pos': result_attn.argmax().item()
                        })
    
    # Analyze specialization
    print("\n🎯 Head Specialization Analysis:")
    for head_name, patterns in head_responses.items():
        print(f"\n{head_name}:")
        
        for pattern_name, responses in patterns.items():
            avg_attn_a = np.mean([r['attn_to_a'] for r in responses])
            avg_attn_b = np.mean([r['attn_to_b'] for r in responses])
            avg_attn_eq = np.mean([r['attn_to_equals'] for r in responses])
            
            print(f"  {pattern_name:15s}: a={avg_attn_a:.3f}, b={avg_attn_b:.3f}, ={avg_attn_eq:.3f}")
    
    return head_responses

def track_attention_flow(model, input_seq, config):
    """Track how attention patterns change across layers"""
    model.eval()
    device = next(model.parameters()).device
    input_seq = input_seq.to(device)
    
    attention_weights = get_attention_weights(model, input_seq)
    
    print("🌊 Attention Flow Analysis:")
    print("Tracking attention from result position across layers...")
    
    for layer_idx, attn in enumerate(attention_weights):
        # Average across heads for simplicity
        avg_attn = attn[0].mean(dim=0)  # [seq_len, seq_len]
        result_attn = avg_attn[-1, :]  # Attention from result position
        
        print(f"Layer {layer_idx+1}: a={result_attn[0]:.3f}, b={result_attn[1]:.3f}, ={result_attn[2]:.3f}")
    
    return attention_weights

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.num_samples:,} samples, seq_len {config.max_seq_len}")

    # Generate data
    sequences, num_to_token, token_to_num, EQUALS = generate_arithmetic_data(config)
    dataset = ArithmeticDataset(sequences)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")

    # Test the model on specific examples
    print(f"\n🧪 TESTING MODEL:")
    model.eval()
    device = next(model.parameters()).device
    
    test_cases = [
        (5, 3),
        (10, -7),
        (-15, 25),
        (0, 42),
        (-8, -12),
        (99, 1),
        (50, 50),
        (-100, 100)
    ]
    
    correct = 0
    for a, b in test_cases:
        expected = a + b
        expected = max(-100, min(100, expected))  # Clamp to valid range
        
        # Create input sequence: [a, b, =]
        input_seq = torch.tensor([num_to_token(a), num_to_token(b), EQUALS], dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_seq)
            # Get prediction for the last position (after =)
            predicted_token = logits[0, -1].argmax().item()
            predicted_num = token_to_num(predicted_token)
            
        is_correct = predicted_num == expected
        if is_correct:
            correct += 1
            
        print(f"   {a:4d} + {b:4d} = {expected:4d} | Predicted: {predicted_num:4d} {'✓' if is_correct else '✗'}")
    
    print(f"   Test Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.1f}%")

    # Interactive mode
    print(f"\n🎮 INTERACTIVE MODE:")
    print("Enter two numbers to test addition (or 'quit' to exit)")
    print("Example: 15 -7")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            parts = user_input.split()
            if len(parts) != 2:
                print("Please enter exactly two numbers separated by space")
                continue
                
            a, b = int(parts[0]), int(parts[1])
            
            # Clamp inputs to valid range
            a = max(-100, min(100, a))
            b = max(-100, min(100, b))
            
            expected = a + b
            expected = max(-100, min(100, expected))
            
            # Create input sequence: [a, b, =]
            input_seq = torch.tensor([num_to_token(a), num_to_token(b), EQUALS], dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(input_seq)
                predicted_token = logits[0, -1].argmax().item()
                predicted_num = token_to_num(predicted_token)
                
                # Also show top 3 predictions
                top_tokens = logits[0, -1].topk(3)
                top_predictions = [(token_to_num(t.item()), prob.item()) for t, prob in zip(top_tokens.indices, torch.softmax(top_tokens.values, dim=0))]
            
            is_correct = predicted_num == expected
            print(f"   {a} + {b} = {expected} | Model: {predicted_num} {'✓' if is_correct else '✗'}")
            print(f"   Top predictions: {top_predictions}")
            
        except ValueError:
            print("Please enter valid integers")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Goodbye!")
    
    # ========================================
    # ATTENTION PATTERN ANALYSIS
    # ========================================
    print("\n" + "="*50)
    print("🔍 ATTENTION PATTERN ANALYSIS")
    print("="*50)
    
    # 1. Visualize attention patterns for different categories
    print("\n1. Visualizing attention patterns...")
    visualize_attention_patterns(model, test_cases, token_to_num, num_to_token, EQUALS)
    
    # 2. Compute attention statistics
    print("\n2. Computing attention statistics...")
    attention_stats = analyze_attention_stats(model, val_loader, config)
    
    # 3. Analyze head specialization
    print("\n3. Analyzing head specialization...")
    head_specialization = analyze_head_specialization(model, config, num_to_token, token_to_num, EQUALS)
    
    # 4. Track attention flow for a specific example
    print("\n4. Tracking attention flow...")
    example_input = torch.tensor([num_to_token(15), num_to_token(-7), EQUALS], dtype=torch.long).unsqueeze(0)
    print(f"Example: 15 + (-7) = 8")
    attention_flow = track_attention_flow(model, example_input, config)
    
    # 5. Additional analysis: Check if model attends differently to correct vs incorrect predictions
    print("\n5. Analyzing attention for correct vs incorrect predictions...")
    correct_cases = []
    incorrect_cases = []
    
    for a, b in test_cases:
        expected = max(-100, min(100, a + b))
        input_seq = torch.tensor([num_to_token(a), num_to_token(b), EQUALS], dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_seq)
            predicted_token = logits[0, -1].argmax().item()
            predicted_num = token_to_num(predicted_token)
            
        if predicted_num == expected:
            correct_cases.append((a, b, input_seq))
        else:
            incorrect_cases.append((a, b, input_seq))
    
    if correct_cases and incorrect_cases:
        print(f"Analyzing {len(correct_cases)} correct vs {len(incorrect_cases)} incorrect cases...")
        
        # Average attention for correct cases
        correct_attn_avg = None
        for a, b, seq in correct_cases:
            attn_weights = get_attention_weights(model, seq)
            if correct_attn_avg is None:
                correct_attn_avg = [torch.zeros_like(attn) for attn in attn_weights]
            for i, attn in enumerate(attn_weights):
                correct_attn_avg[i] += attn
        
        for i in range(len(correct_attn_avg)):
            correct_attn_avg[i] /= len(correct_cases)
        
        # Average attention for incorrect cases
        incorrect_attn_avg = None
        for a, b, seq in incorrect_cases:
            attn_weights = get_attention_weights(model, seq)
            if incorrect_attn_avg is None:
                incorrect_attn_avg = [torch.zeros_like(attn) for attn in attn_weights]
            for i, attn in enumerate(attn_weights):
                incorrect_attn_avg[i] += attn
        
        for i in range(len(incorrect_attn_avg)):
            incorrect_attn_avg[i] /= len(incorrect_cases)
        
        # Compare attention patterns
        print("\nAttention comparison (result position attending to [a, b, =]):")
        for layer_idx in range(len(correct_attn_avg)):
            correct_result_attn = correct_attn_avg[layer_idx][0].mean(dim=0)[-1, :].cpu().numpy()
            incorrect_result_attn = incorrect_attn_avg[layer_idx][0].mean(dim=0)[-1, :].cpu().numpy()
            
            print(f"Layer {layer_idx+1}:")
            print(f"  Correct:   a={correct_result_attn[0]:.3f}, b={correct_result_attn[1]:.3f}, ={correct_result_attn[2]:.3f}")
            print(f"  Incorrect: a={incorrect_result_attn[0]:.3f}, b={incorrect_result_attn[1]:.3f}, ={incorrect_result_attn[2]:.3f}")
            print(f"  Difference: a={correct_result_attn[0]-incorrect_result_attn[0]:+.3f}, b={correct_result_attn[1]-incorrect_result_attn[1]:+.3f}, ={correct_result_attn[2]-incorrect_result_attn[2]:+.3f}")
    
    print("\n" + "="*50)
    print("✅ ATTENTION ANALYSIS COMPLETE")
    print("="*50)