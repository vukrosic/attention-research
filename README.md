# Attention Mechanism Tutorial: Adding Two Numbers

This tutorial demonstrates how an attention mechanism works by building a simple model that adds two numbers using Key-Query-Value attention, followed by a neural network.

## 🎯 What You'll Learn

- How Query, Key, and Value projections work
- What attention weights represent and how they're computed
- How the model decides what to "pay attention to"
- Step-by-step visualization of the attention process

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the basic attention model
python simple_attention.py

# Create visualizations
python visualize_attention.py

# Run the complete tutorial
python attention_tutorial.py
```

## 📚 Tutorial Overview

### The Problem
We want to add two numbers `[3, 5]` using an attention mechanism. While this seems overkill for simple addition, it demonstrates the core concepts that scale to complex tasks like language translation or image recognition.

### The Architecture

```
Input [3, 5] 
    ↓
┌─────────────────────────────────────┐
│         Attention Mechanism         │
│  ┌─────┐  ┌─────┐  ┌─────┐         │
│  │  Q  │  │  K  │  │  V  │         │
│  └─────┘  └─────┘  └─────┘         │
│      ↓        ↓        ↓           │
│  Attention = softmax(Q×K^T) × V     │
└─────────────────────────────────────┘
    ↓
Neural Network
    ↓
Output
```

## 🔍 Step-by-Step Breakdown

### Step 1: Input Numbers
- Start with two numbers: `[3, 5]`
- Goal: Learn to add them using attention

### Step 2: Generate Queries (Q)
- **What it does**: Creates "questions" about what to focus on
- **Formula**: `Q = W_q × Input`
- **Intuition**: "What am I looking for in the input?"

### Step 3: Generate Keys (K)
- **What it does**: Creates "labels" for each input position
- **Formula**: `K = W_k × Input`  
- **Intuition**: "What information is available at each position?"

### Step 4: Generate Values (V)
- **What it does**: Creates the actual "content" to be used
- **Formula**: `V = W_v × Input`
- **Intuition**: "What content should I extract from each position?"

### Step 5: Compute Attention Scores
- **What it does**: Measures compatibility between queries and keys
- **Formula**: `Scores = Q × K^T`
- **Intuition**: "How much should each query attend to each key?"

### Step 6: Apply Softmax (Attention Weights)
- **What it does**: Normalizes scores to probabilities
- **Formula**: `Weights = softmax(Scores)`
- **Intuition**: "Convert raw scores to attention probabilities"

### Step 7: Apply Attention to Values
- **What it does**: Weighted combination of values
- **Formula**: `Output = Weights × V`
- **Intuition**: "Use attention weights to combine value information"

### Step 8: Final Sum
- **What it does**: Sums the attended outputs
- **Result**: A learned combination of the input numbers

## 🧠 What is the Model Paying Attention To?

The attention mechanism learns to focus on different parts of the input based on:

1. **Query-Key Similarity**: Higher similarity = more attention
2. **Learned Transformations**: W_q, W_k, W_v matrices learn optimal projections
3. **Context**: In our simple case, both numbers are equally important for addition

### Attention Options Available:
- **Position 1**: First number (3)
- **Position 2**: Second number (5)
- **Attention Weights**: Show how much to focus on each position
- **Current Pattern**: The model learns to weight both positions for addition

## 📊 Understanding the Visualizations

When you run `attention_tutorial.py`, you'll see:

1. **Input Visualization**: The two numbers being processed
2. **Q, K, V Projections**: How the input is transformed
3. **Attention Scores**: Raw compatibility scores
4. **Attention Weights**: Normalized probabilities (sum to 1)
5. **Attended Output**: Weighted combination of values
6. **Final Result**: Comparison with expected sum

## 🎓 Key Insights

- **Attention is Selective**: The model learns what to focus on
- **Weights are Learned**: Through training, optimal attention patterns emerge
- **Scalable**: Same mechanism works for sequences of any length
- **Interpretable**: Attention weights show model's focus

## 🔧 Customization

Try modifying:
- Input numbers in `simple_attention.py`
- Weight initialization in `TutorialAttention`
- Visualization parameters in `attention_tutorial.py`

## 📈 Next Steps

This simple example demonstrates attention fundamentals. Real applications use:
- Multi-head attention (multiple Q, K, V projections)
- Positional encoding for sequence order
- Layer normalization and residual connections
- Transformer architectures for complex tasks

The core principle remains the same: **learn to focus on relevant information through Query-Key-Value interactions**.

---

# 🧠 Advanced: LLM Training with Attention Analysis

This repository also includes a comprehensive attention analysis framework for analyzing transformer language models (`llm.py`).

## 🔬 Attention Analysis Framework

The framework tests multiple hypotheses about how transformer attention patterns work:

### 📊 Hypotheses Tested

1. **Positional Decay**: Attention strength decays with token distance
2. **Layer Evolution**: Early layers focus locally, later layers globally  
3. **Attention Entropy**: Attention becomes more focused in middle layers
4. **Induction Heads**: Some heads learn to copy/repeat patterns
5. **Special Token Aggregation**: Punctuation acts as information sinks
6. **Head Specialization**: Different heads learn different patterns

### 🚀 Quick Start with LLM Analysis

1. **Run Training with Analysis**:
   ```bash
   python llm.py
   ```
   This will train a transformer model and automatically run attention analysis.

2. **Interactive Analysis Demo**:
   ```bash
   python attention_analysis_demo.py
   ```
   Test the analysis framework with a small model.

3. **Generate HTML Report**:
   ```bash
   python generate_attention_report.py
   ```
   Creates a comprehensive HTML report of analysis results.

### 📈 Analysis Features

- **Attention Weight Extraction**: Capture attention matrices from all layers
- **Positional Analysis**: How attention varies with token distance
- **Entropy Analysis**: Measure attention focus/distribution
- **Head Specialization**: Identify what different heads learn
- **Induction Head Detection**: Find heads that copy patterns
- **Visualization**: Attention heatmaps and pattern plots
- **Comprehensive Reporting**: HTML reports with all results

### 🔍 Example Analysis Output

```
🧠 ATTENTION ANALYSIS
==================================================
🔍 Testing Hypothesis 1: Positional Decay...
🔍 Testing Hypothesis 3: Attention Entropy...
🔍 Testing Hypothesis 4: Induction Heads...
🔍 Testing Hypothesis 5: Special Token Attention...

📊 ANALYSIS SUMMARY:
🎯 Most Specialized Heads:
  L2_H3: avg_distance=4.2341
  L1_H0: entropy=1.8765
  L0_H2: diagonal=0.3452

🔬 Hypothesis Test Results:
  Entropy by Layer:
    Layer 0: 2.1234
    Layer 1: 1.8765
    Layer 2: 2.0123
```

### 📁 Output Files

- `attention_analysis_results.pkl`: Complete analysis data
- `attention_matrix_l0_h0.png`: Attention heatmap visualization
- `entropy_analysis.png`: Entropy progression plot
- `attention_analysis_report.html`: Comprehensive HTML report

### 🛠️ Understanding the Results

- **High Entropy**: Distributed, unfocused attention
- **Low Entropy**: Focused, concentrated attention  
- **Induction Score**: How well heads copy previous patterns
- **Distance Decay**: How attention decreases with token distance
- **Head Specialization**: Different roles for different attention heads

This advanced analysis helps understand what your transformer model is learning and how attention patterns evolve during training!