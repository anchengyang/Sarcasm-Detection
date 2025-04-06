import argparse
import torch
import pandas as pd
import time
import os
import gc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.profiler import profile, record_function, ProfilerActivity


class TextDataset(Dataset):
    def __init__(self, texts, targets, max_length=180, tokenizer_name='bert-base-uncased'):
        self.texts = texts
        self.targets = targets
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        return {
            'input_ids': torch.as_tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.as_tensor(attention_mask, dtype=torch.long),
            'targets': torch.as_tensor(target, dtype=torch.long),
            'text': text
        }


def get_model_size(model_path):
    """Calculate the model size in MB"""
    total_size = 0
    for path, dirs, files in os.walk(model_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB

def run_profiled_inference(model, dataloader, device, profile_path="profiler_output"):
    """Run inference with PyTorch profiler"""
    all_preds = []
    all_labels = []
    
    activities = [ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    # Create directory for profile outputs if it doesn't exist
    os.makedirs(profile_path, exist_ok=True)
    
    # Reset peak memory stats before profiling
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    # Use the profiler to profile the inference
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path)
    ) as prof:
        with torch.no_grad():
            for batch in dataloader:
                with record_function("batch_processing"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    targets = batch['targets'].to(device)
                    
                    with record_function("model_inference"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    with record_function("post_processing"):
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_labels.extend(batch['targets'].cpu().numpy())
                
                prof.step()

    return all_preds, all_labels, prof

def measure_inference_performance(model_path, test_data_path, test_labels_path, batch_sizes=[1, 16, 32, 64], parent_dir="."):
    """Measure inference performance metrics for the model"""
    # Load test data
    test_df = pd.read_csv(test_data_path)
    test_labels = pd.read_csv(test_labels_path)
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get model size before loading
    model_size_mb = get_model_size(model_path)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    performance_metrics = []
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing with batch size: {batch_size} ---")

        # Create profile directory for this batch size
        profile_dir = f"{parent_dir}/profile_batch_{batch_size}"
        
        # Create dataset and dataloader
        test_dataset = TextDataset(
            texts=test_df['headline'].values,
            targets=test_labels['is_sarcastic'].values,
            tokenizer_name=model_path
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated()
            print(f"Initial GPU memory: {initial_gpu_memory / (1024 * 1024):.2f} MB")
        
        # Warm-up, since models in high throughput environments usually remain loaded and warm
        print("Warming up...")
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                break
        
        # Perform garbage collection after warmup
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            # Reset peak memory stats after warmup
            torch.cuda.reset_peak_memory_stats()
            print(f"GPU memory after warmup: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        
        # Measure start time
        start_time = time.time()
        
        # Run profiled inference
        all_preds, all_labels, prof = run_profiled_inference(
            model=model, 
            dataloader=test_dataloader, 
            device=device, 
            profile_path=profile_dir
        )
        
        # Measure end time
        end_time = time.time()

         # Extract and print key profiler statistics
        print("\nProfiler Stats Summary:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        # Get memory stats from profiler
        profiler_stats = {
            'cpu_memory_usage': 0,
            'cuda_memory_usage': 0,
            'cpu_time': 0,
            'cuda_time': 0
        }
        
        for event in prof.key_averages():
            if event.key == "model_inference":
                profiler_stats['cpu_time'] = event.cpu_time * 1e-6  # Convert microseconds to seconds
                profiler_stats['cpu_memory_usage'] = event.self_cpu_memory_usage / (1024 * 1024)  # Convert to MB
                if device == 'cuda':
                    profiler_stats['cuda_time'] = event.device_time * 1e-6  # Convert microseconds to seconds
                    profiler_stats['cuda_memory_usage'] = (torch.cuda.max_memory_allocated() - initial_gpu_memory) / (1024 * 1024)  # Convert to MB
        
        # Get accurate GPU memory usage from PyTorch
        if device == 'cuda':
            # This gives us the peak memory during our profiling session
            gpu_memory_peak = torch.cuda.max_memory_allocated()
            gpu_memory_used = (gpu_memory_peak - initial_gpu_memory) / (1024 * 1024)  # Convert to MB
            print(f"GPU peak memory: {gpu_memory_peak / (1024 * 1024):.2f} MB")
            print(f"GPU memory used: {gpu_memory_used:.2f} MB")
        else:
            gpu_memory_used = 0

        memory_report = torch.cuda.memory_summary()
        print("Memory summary:")
        print(memory_report)


        
        # Calculate metrics
        total_inference_time = end_time - start_time
        samples_processed = len(test_dataset)
        throughput = samples_processed / total_inference_time
        
        # From profiler data
        avg_latency_per_batch = profiler_stats['cpu_time']  # Already converted to seconds
        avg_latency_per_sample = avg_latency_per_batch / batch_size
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        
        print(f"Total inference time: {total_inference_time:.4f} seconds")
        print(f"Average latency per batch: {avg_latency_per_batch*1000:.2f} ms")
        print(f"Average latency per sample: {avg_latency_per_sample*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/second")
        print(f"CPU Memory usage (profiler): {profiler_stats['cpu_memory_usage']:.2f} MB")
        
        if device == 'cuda':
            print(f"CUDA Time (profiler): {profiler_stats['cuda_time']*1000:.2f} ms")
            print(f"CUDA Memory usage (profiler): {profiler_stats['cuda_memory_usage']:.2f} MB")
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Generate trace file location
        trace_file = os.path.join(os.path.abspath(profile_dir), "trace.json")
        print(f"Full profiler trace saved to: {trace_file}")
        print(f"View with TensorBoard: tensorboard --logdir={profile_dir}")
        
        performance_metrics.append({
            'batch_size': batch_size,
            'total_inference_time_s': total_inference_time,
            'avg_latency_per_batch_ms': avg_latency_per_batch * 1000,
            'avg_latency_per_sample_ms': avg_latency_per_sample * 1000,
            'throughput_samples_per_second': throughput,
            'cpu_memory_usage_mb': profiler_stats['cpu_memory_usage'],
            'gpu_memory_usage_mb': gpu_memory_used if device == 'cuda' else 0,
            'profiler_cuda_memory_mb': profiler_stats['cuda_memory_usage'] if device == 'cuda' else 0,
            'model_size_mb': model_size_mb,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })


    # Save results to CSV
    results_df = pd.DataFrame(performance_metrics)
    results_df.to_csv(f"{parent_dir}/profiler_inference_metrics.csv", index=False)
    print(f"\nResults saved to {parent_dir}/profiler_inference_metrics.csv")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use. Example: 'bert'"
    )
    args = parser.parse_args()

    model = args.model
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Construct absolute paths
    test_data_path = os.path.join(BASE_DIR, '../data/test.csv')
    test_labels_path = os.path.join(BASE_DIR, '../data/test_labels.csv')

    model_path = os.path.join(BASE_DIR, f'../models/fine_tuned_{model}')  # Path to saved model

    os.makedirs(os.path.join(BASE_DIR, '../profiler_results'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, f'../profiler_results/{model}'), exist_ok=True)
    parent_dir = os.path.abspath(os.path.join(BASE_DIR, f'../profiler_results/{model}'))
    
    # Test with different batch sizes to measure performance characteristics
    batch_sizes = [32, 64, 128, 256]
    
    
    results = measure_inference_performance(
        model_path=model_path,
        test_data_path=test_data_path,
        test_labels_path=test_labels_path,
        batch_sizes=batch_sizes,
        parent_dir=parent_dir
    )
    
    # Print summary of results
    print("\n=== SUMMARY ===")
    print(f"Model size: {results['model_size_mb'].iloc[0]:.2f} MB")
    
    # Find best throughput configuration
    best_throughput_row = results.loc[results['throughput_samples_per_second'].idxmax()]
    print(f"\nBest throughput configuration:")
    print(f"  Batch size: {best_throughput_row['batch_size']}")
    print(f"  Throughput: {best_throughput_row['throughput_samples_per_second']:.2f} samples/second")
    print(f"  Latency per sample: {best_throughput_row['avg_latency_per_sample_ms']:.2f} ms")
    print(f"  CPU Memory usage: {best_throughput_row['cpu_memory_usage_mb']:.2f} MB")
    if best_throughput_row['gpu_memory_usage_mb'] > 0:
        print(f"  GPU Memory usage: {best_throughput_row['gpu_memory_usage_mb']:.2f} MB")
    
    # Find lowest latency configuration
    best_latency_row = results.loc[results['avg_latency_per_sample_ms'].idxmin()]
    print(f"\nBest latency configuration:")
    print(f"  Batch size: {best_latency_row['batch_size']}")
    print(f"  Latency per sample: {best_latency_row['avg_latency_per_sample_ms']:.2f} ms")
    print(f"  Throughput: {best_latency_row['throughput_samples_per_second']:.2f} samples/second")
    print(f"  CPU Memory usage: {best_latency_row['cpu_memory_usage_mb']:.2f} MB")
    if best_latency_row['gpu_memory_usage_mb'] > 0:
        print(f"  GPU Memory usage: {best_latency_row['gpu_memory_usage_mb']:.2f} MB")
