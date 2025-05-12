import json
import os
import pandas as pd

def load_metrics_file(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

# Define models and their respective metrics files
models = {
    'BERT': '../results/bert_deployability_metrics.json',
    'RoBERTa': '../results/roberta_deployability_metrics.json',
    'DistilBERT': '../results/distilbert_deployability_metrics.json',
    'ALBERT': '../results/albert_deployability_metrics.json',
    'SBERT': '../results/sbert_deployability_metrics.json'
}

# Create a directory to hold the results
os.makedirs('../results', exist_ok=True)

# Load all metrics
results = {}
for model_name, metrics_file in models.items():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, metrics_file)
    metrics = load_metrics_file(full_path)
    if metrics:
        results[model_name] = metrics

# Convert to DataFrame for easy visualization
if results:
    df = pd.DataFrame({
        'Model': [],
        'Parameters': [],
        'Model Size (MB)': [],
        'Inference Latency (ms)': [],
        'Throughput (samples/s)': [],
        'Memory Usage (MB)': []
    })
    
    for model_name, metrics in results.items():
        new_row = {
            'Model': model_name,
            'Parameters': f"{metrics['model_size_parameters']:,}",
            'Model Size (MB)': metrics['model_size_mb'],
            'Inference Latency (ms)': metrics['inference_latency_ms'],
            'Throughput (samples/s)': metrics['throughput_samples_per_second'],
            'Memory Usage (MB)': metrics['memory_usage_mb']
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save as CSV
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/deployability_comparison.csv')
    df.to_csv(output_file, index=False)
    print(f"Deployability metrics comparison saved to {output_file}")
    
    # Print table
    print("\nDeployability Metrics Comparison:")
    print(df.to_string(index=False))
else:
    print("No metrics files found. Run the model training scripts first.") 