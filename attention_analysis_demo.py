#!/usr/bin/env python3
"""
Attention Analysis Demo Script

This script demonstrates how to use the attention analysis framework
with a trained model. Run this after training your model.
"""

import torch
import pickle
import matplotlib.pyplot as plt
from llm import *

def quick_attention_test():
    """Quick test of attention analysis functionality"""
    print("🚀 Quick Attention Analysis Test")
    print("="*40)
    
    # Load a small config for testing
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        batch_size=2,
        max_seq_len=64,
        vocab_size=1000
    )
    
    # Create a small model for testing
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create a dummy tokenizer-like object for testing
    class DummyTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            
        def encode(self, text, add_special_tokens=True, max_length=512, truncation=True):
            # Simple word-based tokenization for demo
            words = text.split()
            return [hash(word) % 1000 for word in words[:max_length]]
        
        def decode(self, tokens):
            if isinstance(tokens, list) and len(tokens) == 1:
                return f"token_{tokens[0]}"
            return f"tokens_{len(tokens)}"
    
    tokenizer = DummyTokenizer()
    
    # Test sentences
    test_sentences = [
        "The cat sat on the mat",
        "The dog ran in the park quickly",
        "Machine learning is fascinating and complex",
        "The cat sat on the mat. The cat was happy.",
        "Python is a programming language. Python is popular."
    ]
    
    print(f"📝 Testing with {len(test_sentences)} sentences...")
    
    # Run analysis
    try:
        analyzer = AttentionAnalyzer(model, tokenizer, device)
        
        # Test basic attention extraction
        print("\n🔍 Testing attention extraction...")
        weights, tokens, token_strings = analyzer.get_attention_weights(test_sentences[0])
        print(f"✅ Successfully extracted attention for {len(weights)} layers")
        
        # Test positional decay analysis
        print("\n🔍 Testing positional decay analysis...")
        decay_results = analyze_positional_decay(analyzer, test_sentences[0])
        print(f"✅ Positional decay analysis completed for {len(decay_results)} layers")
        
        # Test entropy analysis
        print("\n🔍 Testing entropy analysis...")
        entropy_results = analyze_attention_entropy(analyzer, test_sentences[:3])
        print(f"✅ Entropy analysis completed")
        for layer, stats in entropy_results.items():
            print(f"   Layer {layer}: entropy = {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        # Test induction head analysis
        print("\n🔍 Testing induction head analysis...")
        induction_results = find_induction_heads(analyzer, test_sentences[3])
        print(f"✅ Found {len(induction_results)} potential induction heads")
        
        # Test head specialization
        print("\n🔍 Testing head specialization analysis...")
        head_results = analyze_head_specialization(analyzer, test_sentences[:3], n_samples=3)
        print(f"✅ Analyzed {len(head_results)} heads")
        
        # Show top specialized heads
        print("\n🎯 Top 3 heads by average distance:")
        sorted_heads = sorted(head_results.items(), 
                            key=lambda x: x[1]['avg_distance'], 
                            reverse=True)[:3]
        for head, stats in sorted_heads:
            print(f"   {head}: {stats['avg_distance']:.3f}")
        
        print("\n✅ All attention analysis tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_saved_model():
    """Analyze a saved model if available"""
    try:
        # Try to load saved analysis results
        with open('attention_analysis_results.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print("📊 Loaded saved attention analysis results!")
        print("="*50)
        
        analysis_results = data['attention_analysis']
        head_specs = data['head_specialization']
        
        # Display results
        if 'entropy' in analysis_results:
            print("\n📈 Entropy Analysis Results:")
            for layer, stats in analysis_results['entropy'].items():
                print(f"   Layer {layer}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if 'induction' in analysis_results:
            print(f"\n🔄 Top 5 Induction Heads:")
            sorted_induction = sorted(analysis_results['induction'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
            for head, score in sorted_induction:
                print(f"   {head}: {score:.4f}")
        
        if head_specs:
            print(f"\n🎯 Head Specialization Summary:")
            print(f"   Total heads analyzed: {len(head_specs)}")
            
            # Find most specialized heads
            for metric in ['avg_distance', 'entropy', 'diagonal']:
                sorted_heads = sorted(head_specs.items(),
                                    key=lambda x: x[1][metric],
                                    reverse=True)[:3]
                print(f"\n   Top 3 by {metric}:")
                for head, stats in sorted_heads:
                    print(f"     {head}: {stats[metric]:.4f}")
        
        return True
        
    except FileNotFoundError:
        print("📁 No saved analysis results found. Run the main training script first.")
        return False
    except Exception as e:
        print(f"❌ Error loading saved results: {e}")
        return False

def interactive_analysis():
    """Interactive analysis mode"""
    print("\n🎮 Interactive Analysis Mode")
    print("="*30)
    
    while True:
        print("\nChoose an option:")
        print("1. Quick attention test")
        print("2. Analyze saved model")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            quick_attention_test()
        elif choice == '2':
            analyze_saved_model()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    print("🧠 Attention Analysis Framework Demo")
    print("="*40)
    
    # Check if we have matplotlib available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Visualization libraries available")
    except ImportError:
        print("⚠️ Visualization libraries not available. Install matplotlib and seaborn for full functionality.")
    
    # Run interactive analysis
    interactive_analysis()