#!/usr/bin/env python3
"""
Test script for attention analysis framework
"""

import torch
import sys
import traceback

def test_imports():
    """Test if all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        from llm import (
            AttentionAnalyzer, analyze_positional_decay, analyze_attention_entropy,
            find_induction_heads, analyze_special_token_attention, 
            analyze_head_specialization, MinimalLLM, ModelConfig
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation with attention analysis support"""
    print("\n🔍 Testing model creation...")
    
    try:
        config = ModelConfig(
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            vocab_size=100,
            max_seq_len=32
        )
        
        model = MinimalLLM(config)
        print("✅ Model created successfully")
        
        # Test forward pass with attention
        x = torch.randint(0, 100, (1, 10))
        logits, attention_weights = model(x, return_attention=True)
        
        print(f"✅ Forward pass successful: logits shape {logits.shape}")
        print(f"✅ Attention weights captured: {len(attention_weights)} layers")
        
        return True, model, config
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_attention_analysis(model, config):
    """Test attention analysis functions"""
    print("\n🔍 Testing attention analysis...")
    
    try:
        # Create dummy tokenizer
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = config.vocab_size
                
            def encode(self, text, add_special_tokens=True, max_length=512, truncation=True):
                words = text.split()[:max_length]
                return [hash(word) % self.vocab_size for word in words]
            
            def decode(self, tokens):
                if isinstance(tokens, list) and len(tokens) == 1:
                    return f"token_{tokens[0]}"
                return f"tokens_{len(tokens)}"
        
        tokenizer = DummyTokenizer()
        analyzer = AttentionAnalyzer(model, tokenizer, 'cpu')
        
        # Test attention extraction
        test_text = "The cat sat on the mat"
        weights, tokens, token_strings = analyzer.get_attention_weights(test_text)
        print(f"✅ Attention extraction: {len(weights)} layers, {len(tokens)} tokens")
        
        # Test positional decay analysis
        decay_results = analyze_positional_decay(analyzer, test_text)
        print(f"✅ Positional decay analysis: {len(decay_results)} layers")
        
        # Test entropy analysis
        entropy_results = analyze_attention_entropy(analyzer, [test_text])
        print(f"✅ Entropy analysis: {len(entropy_results)} layers")
        
        # Test induction heads
        induction_text = "The cat sat. The cat ran."
        induction_results = find_induction_heads(analyzer, induction_text)
        print(f"✅ Induction head analysis: {len(induction_results)} heads")
        
        # Test special token analysis
        special_results = analyze_special_token_attention(analyzer, [test_text])
        print(f"✅ Special token analysis: {len(special_results)} layers")
        
        # Test head specialization
        head_results = analyze_head_specialization(analyzer, [test_text], n_samples=1)
        print(f"✅ Head specialization: {len(head_results)} heads")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention analysis failed: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """Test if visualization libraries are available"""
    print("\n🔍 Testing visualization libraries...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Matplotlib and seaborn available")
        return True
    except ImportError as e:
        print(f"⚠️ Visualization libraries not available: {e}")
        print("   Install with: pip install matplotlib seaborn")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Attention Analysis Framework")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Cannot continue.")
        sys.exit(1)
    
    # Test model creation
    success, model, config = test_model_creation()
    if not success:
        print("\n❌ Model creation test failed. Cannot continue.")
        sys.exit(1)
    
    # Test attention analysis
    if not test_attention_analysis(model, config):
        print("\n❌ Attention analysis test failed.")
        sys.exit(1)
    
    # Test visualization (optional)
    test_visualization()
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Attention analysis framework is working correctly")
    print("\nNext steps:")
    print("1. Run 'python llm.py' to train a model with analysis")
    print("2. Run 'python attention_analysis_demo.py' for interactive demo")
    print("3. Run 'python generate_attention_report.py' to create HTML report")

if __name__ == "__main__":
    main()