import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
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
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 5000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

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
	
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
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
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

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
        
        # Store attention weights for analysis
        self.last_attn_weights = None

    def forward(self, x, return_attention=False):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        # Compute attention scores manually to capture weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(scores.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn_weights.detach()
        
        # Apply dropout and compute output
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        output = self.w_o(attn_output)
        
        if return_attention:
            return output, self.last_attn_weights
        return output

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

    def forward(self, x):
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
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if return_attention:
                attention_weights.append({
                    'layer': i,
                    'weights': block.attention.last_attn_weights
                })

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        if return_attention:
            return logits, attention_weights
        return logits

class AttentionAnalyzer:
    """Framework for analyzing attention patterns in the transformer"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def get_attention_weights(self, text, return_tokens=True):
        """Extract attention weights for a given text"""
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True, 
                                     max_length=512, truncation=True)
        input_ids = torch.tensor([tokens]).to(self.device)
        
        # Forward pass with attention capture
        with torch.no_grad():
            logits, attention_weights = self.model(input_ids, return_attention=True)
        
        if return_tokens:
            token_strings = [self.tokenizer.decode([t]) for t in tokens]
            return attention_weights, tokens, token_strings
        return attention_weights

def analyze_positional_decay(analyzer, text):
    """Hypothesis 1: Attention strength decays with distance"""
    weights, tokens, token_strings = analyzer.get_attention_weights(text)
    
    results = defaultdict(list)
    
    for layer_data in weights:
        layer_idx = layer_data['layer']
        attn = layer_data['weights']  # [batch, heads, seq, seq]
        
        if attn is None:
            continue
        
        # Average over batch and heads
        attn_avg = attn.mean(dim=(0, 1)).cpu().numpy()
        
        # Calculate average attention by distance
        seq_len = attn_avg.shape[0]
        distance_attention = defaultdict(list)
        
        for i in range(seq_len):
            for j in range(i):
                distance = i - j
                attention_value = attn_avg[i, j]
                distance_attention[distance].append(attention_value)
        
        # Average by distance
        avg_by_distance = {}
        for dist, values in distance_attention.items():
            if dist <= 10:  # Focus on nearby tokens
                avg_by_distance[dist] = np.mean(values)
        
        results[f'layer_{layer_idx}'] = avg_by_distance
    
    return results

def analyze_attention_entropy(analyzer, texts):
    """Hypothesis 3: Attention entropy changes across layers"""
    entropy_by_layer = defaultdict(list)
    
    for text in texts:
        weights, tokens, token_strings = analyzer.get_attention_weights(text)
        
        for layer_data in weights:
            layer_idx = layer_data['layer']
            attn = layer_data['weights']
            
            if attn is None:
                continue
            
            # Calculate entropy for each position
            attn_cpu = attn.cpu().numpy()
            batch_size, n_heads, seq_len, _ = attn_cpu.shape
            
            for b in range(batch_size):
                for h in range(n_heads):
                    for i in range(seq_len):
                        # Entropy of attention distribution at position i
                        attn_dist = attn_cpu[b, h, i, :i+1]  # Only look at valid positions
                        if len(attn_dist) > 0:
                            ent = entropy(attn_dist + 1e-10)
                            entropy_by_layer[layer_idx].append(ent)
    
    # Average entropy by layer
    avg_entropy = {}
    for layer, entropies in entropy_by_layer.items():
        avg_entropy[layer] = {
            'mean': np.mean(entropies),
            'std': np.std(entropies),
            'median': np.median(entropies)
        }
    
    return avg_entropy

def find_induction_heads(analyzer, text_with_repetitions):
    """Hypothesis 4: Some heads act as induction heads"""
    weights, tokens, token_strings = analyzer.get_attention_weights(text_with_repetitions)
    
    induction_scores = defaultdict(list)
    
    # Find repeated tokens
    token_positions = defaultdict(list)
    for i, token in enumerate(tokens):
        token_positions[token].append(i)
    
    # Find tokens that appear multiple times
    repeated_tokens = {tok: positions for tok, positions in token_positions.items()
                      if len(positions) > 1}
    
    for layer_data in weights:
        layer_idx = layer_data['layer']
        attn = layer_data['weights']
        
        if attn is None:
            continue
        
        batch_size, n_heads, seq_len, _ = attn.shape
        
        for head in range(n_heads):
            induction_score = 0
            count = 0
            
            for token, positions in repeated_tokens.items():
                if len(positions) < 2:
                    continue
                
                # For each repetition after the first
                for i in range(1, len(positions)):
                    curr_pos = positions[i]
                    prev_pos = positions[i-1]
                    
                    if prev_pos + 1 < seq_len and curr_pos < seq_len:
                        # Check if current position attends to token after previous occurrence
                        attention_to_next = attn[0, head, curr_pos, prev_pos + 1].item()
                        induction_score += attention_to_next
                        count += 1
            
            if count > 0:
                induction_scores[f'layer_{layer_idx}_head_{head}'] = induction_score / count
    
    return induction_scores

def analyze_special_token_attention(analyzer, texts):
    """Hypothesis 5: Special tokens aggregate information"""
    special_tokens = {'.', ',', '!', '?', ';', ':', '[SEP]', '[CLS]', '<eos>', '<pad>'}
    
    attention_to_special = defaultdict(lambda: defaultdict(list))
    
    for text in texts:
        weights, tokens, token_strings = analyzer.get_attention_weights(text)
        
        # Identify special token positions
        special_positions = []
        for i, token_str in enumerate(token_strings):
            if any(special in token_str for special in special_tokens):
                special_positions.append(i)
        
        for layer_data in weights:
            layer_idx = layer_data['layer']
            attn = layer_data['weights']
            
            if attn is None or len(special_positions) == 0:
                continue
            
            # Average attention to special tokens
            attn_avg = attn.mean(dim=(0, 1)).cpu().numpy()
            
            for i in range(attn_avg.shape[0]):
                for special_pos in special_positions:
                    if special_pos < i:  # Can only attend to previous positions
                        attention_value = attn_avg[i, special_pos]
                        attention_to_special[layer_idx]['to_special'].append(attention_value)
                
                # Compare with attention to regular tokens
                regular_positions = [p for p in range(i) if p not in special_positions]
                if regular_positions:
                    avg_regular = np.mean([attn_avg[i, p] for p in regular_positions])
                    attention_to_special[layer_idx]['to_regular'].append(avg_regular)
    
    # Compute statistics
    results = {}
    for layer in attention_to_special:
        special_attn = attention_to_special[layer]['to_special']
        regular_attn = attention_to_special[layer]['to_regular']
        
        results[layer] = {
            'special_mean': np.mean(special_attn) if special_attn else 0,
            'regular_mean': np.mean(regular_attn) if regular_attn else 0,
            'special_std': np.std(special_attn) if special_attn else 0,
            'regular_std': np.std(regular_attn) if regular_attn else 0,
        }
        
        if special_attn and regular_attn:
            results[layer]['ratio'] = results[layer]['special_mean'] / (results[layer]['regular_mean'] + 1e-10)
    
    return results

def analyze_head_specialization(analyzer, texts, n_samples=10):
    """Analyze if different heads specialize in different patterns"""
    head_statistics = defaultdict(lambda: defaultdict(list))
    
    for text in texts[:n_samples]:
        weights, tokens, token_strings = analyzer.get_attention_weights(text)
        
        for layer_data in weights:
            layer_idx = layer_data['layer']
            attn = layer_data['weights']
            
            if attn is None:
                continue
            
            batch_size, n_heads, seq_len, _ = attn.shape
            
            for head in range(n_heads):
                head_attn = attn[0, head].cpu().numpy()
                
                # Compute various statistics for this head
                # 1. Average attention distance
                distances = []
                for i in range(seq_len):
                    for j in range(i):
                        distances.append((i - j) * head_attn[i, j])
                avg_distance = np.mean(distances) if distances else 0
                
                # 2. Attention entropy
                entropies = []
                for i in range(seq_len):
                    if i > 0:
                        ent = entropy(head_attn[i, :i] + 1e-10)
                        entropies.append(ent)
                avg_entropy = np.mean(entropies) if entropies else 0
                
                # 3. Diagonal attention (attending to previous token)
                diagonal_attn = np.mean([head_attn[i, i-1] for i in range(1, seq_len)])
                
                head_statistics[f'L{layer_idx}_H{head}']['avg_distance'].append(avg_distance)
                head_statistics[f'L{layer_idx}_H{head}']['entropy'].append(avg_entropy)
                head_statistics[f'L{layer_idx}_H{head}']['diagonal'].append(diagonal_attn)
    
    # Compute summary statistics
    head_summary = {}
    for head_key, stats in head_statistics.items():
        head_summary[head_key] = {
            'avg_distance': np.mean(stats['avg_distance']),
            'entropy': np.mean(stats['entropy']),
            'diagonal': np.mean(stats['diagonal']),
            'distance_std': np.std(stats['avg_distance']),
            'entropy_std': np.std(stats['entropy']),
        }
    
    return head_summary

def visualize_attention_matrix(attention_weights, tokens, layer=0, head=0, 
                              figsize=(12, 10), save_path=None):
    """Visualize attention matrix as heatmap"""
    # Get attention for specific layer and head
    attn = attention_weights[layer]['weights'][0, head].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                cmap='Blues', cbar_kws={'label': 'Attention Weight'})
    
    ax.set_title(f'Attention Matrix - Layer {layer}, Head {head}')
    ax.set_xlabel('Keys (Attended to)')
    ax.set_ylabel('Queries (Attending from)')
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_hypothesis_results(results, hypothesis_name, save_path=None):
    """Generic plotting function for hypothesis results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(results, dict):
        if all(isinstance(v, dict) for v in results.values()):
            # Multi-dimensional results
            layers = sorted([int(k.split('_')[-1]) for k in results.keys() if 'layer' in k])
            metrics = list(next(iter(results.values())).keys())
            
            for metric in metrics:
                values = [results[f'layer_{l}'].get(metric, 0) for l in layers]
                ax.plot(layers, values, marker='o', label=metric)
            
            ax.set_xlabel('Layer')
            ax.set_ylabel('Value')
            ax.legend()
        else:
            # Simple key-value results
            keys = list(results.keys())
            values = list(results.values())
            ax.bar(keys, values)
            ax.set_xlabel('Category')
            ax.set_ylabel('Value')
            plt.xticks(rotation=45, ha='right')
    
    ax.set_title(f'Results: {hypothesis_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def run_attention_experiments(model, tokenizer, sample_texts):
    """Run all hypothesis tests and visualizations"""
    analyzer = AttentionAnalyzer(model, tokenizer)
    
    results = {}
    
    print("🔍 Testing Hypothesis 1: Positional Decay...")
    h1_results = analyze_positional_decay(analyzer, sample_texts[0])
    results['positional_decay'] = h1_results
    
    print("🔍 Testing Hypothesis 3: Attention Entropy...")
    h3_results = analyze_attention_entropy(analyzer, sample_texts[:5])
    results['entropy'] = h3_results
    
    print("🔍 Testing Hypothesis 4: Induction Heads...")
    repetitive_text = "The cat sat on the mat. The dog played with the ball. The cat sat on the mat again."
    h4_results = find_induction_heads(analyzer, repetitive_text)
    results['induction'] = h4_results
    
    print("🔍 Testing Hypothesis 5: Special Token Attention...")
    h5_results = analyze_special_token_attention(analyzer, sample_texts[:5])
    results['special_tokens'] = h5_results
    
    return results, analyzer

def integrate_analysis(model, tokenizer, texts):
    """Main analysis integration function"""
    print("\n" + "="*50)
    print("🧠 ATTENTION ANALYSIS")
    print("="*50)
    
    # Run experiments
    results, analyzer = run_attention_experiments(model, tokenizer, texts)
    
    # Analyze head specialization
    print("\n🔍 Analyzing head specialization...")
    head_specs = analyze_head_specialization(analyzer, texts)
    
    # Print summary
    print("\n📊 ANALYSIS SUMMARY:")
    print("-" * 40)
    
    # Find most specialized heads
    print("\n🎯 Most Specialized Heads:")
    for metric in ['avg_distance', 'entropy', 'diagonal']:
        sorted_heads = sorted(head_specs.items(),
                             key=lambda x: x[1][metric],
                             reverse=True)[:3]
        print(f"\n  {metric.upper()}:")
        for head, stats in sorted_heads:
            print(f"    {head}: {stats[metric]:.4f}")
    
    # Print hypothesis results
    print("\n🔬 Hypothesis Test Results:")
    print("-" * 40)
    
    if 'entropy' in results:
        print("\n  Entropy by Layer:")
        for layer in sorted(results['entropy'].keys()):
            print(f"    Layer {layer}: {results['entropy'][layer]['mean']:.4f}")
    
    if 'special_tokens' in results:
        print("\n  Special Token Attention Ratio:")
        for layer in sorted(results['special_tokens'].keys()):
            ratio = results['special_tokens'][layer].get('ratio', 0)
            print(f"    Layer {layer}: {ratio:.4f}")
    
    if 'induction' in results:
        print("\n  Top Induction Heads:")
        sorted_induction = sorted(results['induction'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        for head, score in sorted_induction:
            print(f"    {head}: {score:.4f}")
    
    return results, head_specs

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
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

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

    # Run comprehensive attention analysis
    print(f"\n🔬 Running post-training attention analysis...")
    sample_texts = texts[:20]  # Use first 20 texts for analysis
    
    try:
        analysis_results, head_specs = integrate_analysis(model, tokenizer, sample_texts)
        
        # Create some visualizations
        print(f"\n📊 Creating attention visualizations...")
        analyzer = AttentionAnalyzer(model, tokenizer)
        
        # Test with a simple sentence
        test_sentence = "The quick brown fox jumps over the lazy dog."
        weights, tokens, token_strings = analyzer.get_attention_weights(test_sentence)
        
        # Visualize attention matrix for first layer, first head
        if len(weights) > 0 and weights[0]['weights'] is not None:
            print(f"📈 Visualizing attention matrix for Layer 0, Head 0...")
            visualize_attention_matrix(weights, token_strings, layer=0, head=0, 
                                     save_path="attention_matrix_l0_h0.png")
        
        # Plot entropy results
        if 'entropy' in analysis_results:
            print(f"📈 Plotting entropy analysis...")
            plot_hypothesis_results(analysis_results['entropy'], 
                                   "Attention Entropy by Layer",
                                   save_path="entropy_analysis.png")
        
        # Save analysis results
        analysis_data = {
            'final_metrics': final_metrics,
            'attention_analysis': analysis_results,
            'head_specialization': head_specs,
            'config': config
        }
        
        with open('attention_analysis_results.pkl', 'wb') as f:
            pickle.dump(analysis_data, f)
        
        print(f"💾 Saved attention analysis results to 'attention_analysis_results.pkl'")
        
    except Exception as e:
        print(f"⚠️ Attention analysis failed: {e}")
        print("Continuing without analysis...")