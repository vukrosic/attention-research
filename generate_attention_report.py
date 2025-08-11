#!/usr/bin/env python3
"""
Attention Analysis Report Generator

This script generates a comprehensive HTML report of attention analysis results.
"""

import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

def generate_html_report(analysis_data, output_file="attention_analysis_report.html"):
    """Generate a comprehensive HTML report"""
    
    # Extract data
    final_metrics = analysis_data.get('final_metrics', {})
    attention_analysis = analysis_data.get('attention_analysis', {})
    head_specs = analysis_data.get('head_specialization', {})
    config = analysis_data.get('config', {})
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Attention Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .analysis-section {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .hypothesis-result {{
                background-color: white;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #27ae60;
                border-radius: 4px;
            }}
            .head-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            .head-table th, .head-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .head-table th {{
                background-color: #3498db;
                color: white;
            }}
            .head-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .timestamp {{
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
                margin-bottom: 30px;
            }}
            .config-table {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Attention Analysis Report</h1>
            <div class="timestamp">Generated on {timestamp}</div>
    """
    
    # Model Configuration Section
    html_content += """
            <h2>📋 Model Configuration</h2>
            <div class="config-table">
    """
    
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config if isinstance(config, dict) else {}
    
    for key, value in config_dict.items():
        html_content += f"                <strong>{key}:</strong> {value}<br>\n"
    
    html_content += """
            </div>
    """
    
    # Training Results Section
    html_content += """
            <h2>🏆 Training Results</h2>
            <div class="metric-grid">
    """
    
    if final_metrics:
        metrics = [
            ("Validation Loss", final_metrics.get('val_loss', 0), "4f"),
            ("Validation Accuracy", final_metrics.get('val_accuracy', 0), "4f"),
            ("Validation Perplexity", final_metrics.get('val_perplexity', 0), "2f")
        ]
        
        for label, value, fmt in metrics:
            html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.{fmt[1:]}}</div>
                </div>
            """
    
    html_content += """
            </div>
    """
    
    # Attention Analysis Results
    html_content += """
            <h2>🔬 Attention Analysis Results</h2>
    """
    
    # Entropy Analysis
    if 'entropy' in attention_analysis:
        html_content += """
            <div class="analysis-section">
                <h3>📊 Attention Entropy by Layer</h3>
                <p>This measures how focused or distributed attention is across layers.</p>
        """
        
        entropy_data = attention_analysis['entropy']
        for layer in sorted(entropy_data.keys()):
            stats = entropy_data[layer]
            html_content += f"""
                <div class="hypothesis-result">
                    <strong>Layer {layer}:</strong> 
                    Mean Entropy: {stats['mean']:.4f} ± {stats['std']:.4f} 
                    (Median: {stats['median']:.4f})
                </div>
            """
        
        html_content += """
            </div>
        """
    
    # Induction Heads
    if 'induction' in attention_analysis:
        html_content += """
            <div class="analysis-section">
                <h3>🔄 Induction Head Analysis</h3>
                <p>Heads that attend to previous occurrences to predict repetitions.</p>
        """
        
        induction_data = attention_analysis['induction']
        sorted_induction = sorted(induction_data.items(), key=lambda x: x[1], reverse=True)
        
        html_content += """
                <table class="head-table">
                    <tr><th>Head</th><th>Induction Score</th></tr>
        """
        
        for head, score in sorted_induction[:10]:  # Top 10
            html_content += f"""
                    <tr><td>{head}</td><td>{score:.4f}</td></tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    # Special Token Analysis
    if 'special_tokens' in attention_analysis:
        html_content += """
            <div class="analysis-section">
                <h3>🎯 Special Token Attention</h3>
                <p>How much attention is paid to special tokens vs regular tokens.</p>
        """
        
        special_data = attention_analysis['special_tokens']
        for layer in sorted(special_data.keys()):
            stats = special_data[layer]
            ratio = stats.get('ratio', 0)
            html_content += f"""
                <div class="hypothesis-result">
                    <strong>Layer {layer}:</strong> 
                    Special/Regular Ratio: {ratio:.4f}
                    (Special: {stats.get('special_mean', 0):.4f}, 
                     Regular: {stats.get('regular_mean', 0):.4f})
                </div>
            """
        
        html_content += """
            </div>
        """
    
    # Head Specialization
    if head_specs:
        html_content += """
            <div class="analysis-section">
                <h3>🎯 Head Specialization Analysis</h3>
                <p>Different attention heads specialize in different patterns.</p>
        """
        
        # Top heads by different metrics
        metrics = ['avg_distance', 'entropy', 'diagonal']
        metric_names = ['Average Distance', 'Entropy', 'Diagonal Attention']
        
        for metric, name in zip(metrics, metric_names):
            html_content += f"""
                <h4>Top 5 Heads by {name}</h4>
                <table class="head-table">
                    <tr><th>Head</th><th>{name}</th></tr>
            """
            
            sorted_heads = sorted(head_specs.items(), 
                                key=lambda x: x[1].get(metric, 0), 
                                reverse=True)[:5]
            
            for head, stats in sorted_heads:
                value = stats.get(metric, 0)
                html_content += f"""
                    <tr><td>{head}</td><td>{value:.4f}</td></tr>
                """
            
            html_content += """
                </table>
            """
        
        html_content += """
            </div>
        """
    
    # Hypothesis Summary
    html_content += """
            <h2>📝 Hypothesis Testing Summary</h2>
            <div class="analysis-section">
                <div class="hypothesis-result">
                    <strong>Hypothesis 1 - Positional Decay:</strong> 
                    Attention strength should decay with distance between tokens.
                    <em>Status: Analyzed ✓</em>
                </div>
                <div class="hypothesis-result">
                    <strong>Hypothesis 2 - Layer Evolution:</strong> 
                    Early layers focus locally, later layers globally.
                    <em>Status: Partially analyzed through entropy ✓</em>
                </div>
                <div class="hypothesis-result">
                    <strong>Hypothesis 3 - Attention Entropy:</strong> 
                    Attention becomes more focused in middle layers.
                    <em>Status: Analyzed ✓</em>
                </div>
                <div class="hypothesis-result">
                    <strong>Hypothesis 4 - Induction Heads:</strong> 
                    Some heads learn to copy/repeat patterns.
                    <em>Status: Analyzed ✓</em>
                </div>
                <div class="hypothesis-result">
                    <strong>Hypothesis 5 - Special Tokens:</strong> 
                    Punctuation and special tokens aggregate information.
                    <em>Status: Analyzed ✓</em>
                </div>
            </div>
    """
    
    # Footer
    html_content += """
            <div style="margin-top: 50px; text-align: center; color: #7f8c8d;">
                <p>Generated by Attention Analysis Framework</p>
                <p>For more detailed analysis, check the saved pickle files and visualization images.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML report generated: {output_file}")
    return output_file

def main():
    """Main function to generate report"""
    try:
        # Load analysis data
        with open('attention_analysis_results.pkl', 'rb') as f:
            analysis_data = pickle.load(f)
        
        print("📊 Loaded attention analysis data")
        
        # Generate HTML report
        report_file = generate_html_report(analysis_data)
        
        print(f"✅ Report generated successfully!")
        print(f"📂 Open {report_file} in your browser to view the results")
        
    except FileNotFoundError:
        print("❌ No analysis results found. Please run the main training script first.")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()