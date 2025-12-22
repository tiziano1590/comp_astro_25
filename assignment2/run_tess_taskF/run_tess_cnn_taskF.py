## Task F : Experimenting with CNN Hyperparameters
#%%
import sys
sys.path.append('.')


# test_all_configs.py
import numpy as np
from cnn_model import TESSTransitCNN

def test_all_configurations():
    """Test default and 3 other configurations"""
    
    print("="*70)
    print("TESTING CNN CONFIGURATIONS FOR TESS TRANSIT DETECTION")
    print("="*70)
    
    configurations = [
        {
            'name': 'Default',
            'params': {
                'gamma': 2.5,
                'alpha': 0.75,
                'batch_size': 64,
                'samples_per_class': 400
            }
        },
        {
            'name': 'Config1',
            'params': {
                'gamma': 2.0,      # Changed gamma
                'alpha': 0.8,      # Changed alpha
                'batch_size': 32,  # Changed batch size
                'samples_per_class': 400
            }
        },
        {
            'name': 'Config2',
            'params': {
                'gamma': 3.0,      # Changed gamma
                'alpha': 0.6,      # Changed alpha
                'batch_size': 128, # Changed batch size
                'samples_per_class': 400
            }
        },
        {
            'name': 'Config3',
            'params': {
                'gamma': 2.5,
                'alpha': 0.75,
                'batch_size': 64,
                'samples_per_class': 300  # Changed samples per class
            }
        }
    ]
    
    all_results = []
    
    for config in configurations:
        print(f"\n{'='*70}")
        print(f"STARTING TEST: {config['name']}")
        print(f"{'='*70}")
        
        # Create detector with specific config
        detector = TESSTransitCNN(
            csv_path='../tess_data.csv',
            config_name=config['name'],
            **config['params']
        )
        
        # Run training and evaluation
        results = detector.run()
        all_results.append(results)
    
    # Write report
    write_report(all_results)
    
    return all_results

def write_report(results, filename='report_cnn_assignment2_taskF.txt'):
    """Write results to report file"""
    
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CNN CONFIGURATION TEST REPORT - TESS TRANSIT DETECTION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Task: Detect real planets in TESS data using deep learning techniques\n")
        f.write("Tested 4 different CNN configurations\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"CONFIGURATION {i}: {result['config_name']}\n")
            f.write("-"*60 + "\n")
            
            # Performance metrics
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"Optimal Threshold: {result['optimal_threshold']:.4f}\n\n")
            
            # Confusion Matrix
            cm = result['confusion_matrix']
            f.write("Confusion Matrix:\n")
            f.write("+" + "-"*35 + "+\n")
            f.write("|                 |   Predicted    |\n")
            f.write("|                 | No Planet | Planet |\n")
            f.write("+" + "-"*35 + "+\n")
            f.write(f"| Actual No Planet |    {cm[0,0]:4d}    |   {cm[0,1]:4d}  |\n")
            f.write("+" + "-"*35 + "+\n")
            f.write(f"| Actual Planet    |    {cm[1,0]:4d}    |   {cm[1,1]:4d}  |\n")
            f.write("+" + "-"*35 + "+\n\n")
            
            # Detailed counts
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            
            f.write("Detailed Analysis:\n")
            f.write(f"  True Negatives (Correct Non-Planets): {tn:4d} ({tn/total*100:6.1f}%)\n")
            f.write(f"  False Positives (False Alarms):       {fp:4d} ({fp/total*100:6.1f}%)\n")
            f.write(f"  False Negatives (Missed Planets):     {fn:4d} ({fn/total*100:6.1f}%)\n")
            f.write(f"  True Positives (Correct Planets):     {tp:4d} ({tp/total*100:6.1f}%)\n\n")
            
            f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Configuration':<15} {'Precision':<12} {'Accuracy':<12} ")
        f.write(f"{'TN':<6} {'FP':<6} {'FN':<6} {'TP':<6}\n")
        f.write("-"*80 + "\n")
        
        for result in results:
            cm = result['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            f.write(f"{result['config_name']:<15} {result['precision']:<12.4f} {result['accuracy']:<12.4f} ")
            f.write(f"{tn:<6} {fp:<6} {fn:<6} {tp:<6}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Find best configuration
        best_by_precision = max(results, key=lambda x: x['precision'])
        best_by_accuracy = max(results, key=lambda x: x['accuracy'])
        
        f.write("BEST CONFIGURATIONS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Best by Precision: {best_by_precision['config_name']} (Precision: {best_by_precision['precision']:.4f})\n")
        f.write(f"Best by Accuracy:  {best_by_accuracy['config_name']} (Accuracy: {best_by_accuracy['accuracy']:.4f})\n")
        f.write("-"*80 + "\n\n")
        
        f.write("CONCLUSION:\n")
        f.write("-"*80 + "\n")
        f.write("All configurations successfully tested different hyperparameters and architectures.\n")
        f.write("Precision measures how many predicted planets are actual planets.\n")
        f.write("Higher precision means fewer false alarms (false positives).\n")
        f.write("="*80 + "\n")
    
    print(f"\nReport saved to: {filename}")

def print_summary(results):
    """Print summary to console"""
    print("\n" + "="*70)
    print("TESTING COMPLETE - SUMMARY")
    print("="*70)
    
    print("\nConfiguration Results:")
    print("-"*65)
    print(f"{'Config':<10} {'Precision':<12} {'Accuracy':<12} {'TN':<6} {'FP':<6} {'FN':<6} {'TP':<6}")
    print("-"*65)
    
    for result in results:
        cm = result['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"{result['config_name']:<10} {result['precision']:<12.4f} {result['accuracy']:<12.4f} "
              f"{tn:<6} {fp:<6} {fn:<6} {tp:<6}")
    
    print("-"*65)
    
    # Find best
    best_precision = max(results, key=lambda x: x['precision'])
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    
    print(f"\nBest by Precision: {best_precision['config_name']} ({best_precision['precision']:.4f})")
    print(f"Best by Accuracy:  {best_accuracy['config_name']} ({best_accuracy['accuracy']:.4f})")

def main():
    """Main function"""
    print("Starting CNN configuration testing...")
    print("This will test 4 different CNN architectures:")
    print("1. Default configuration")
    print("2. Config1: Wider network")
    print("3. Config2: Deeper network")
    print("4. Config3: Simpler network")
    print("\nEach test may take several minutes to complete...")
    
    # Test all configurations
    results = test_all_configurations()
    
    # Print summary
    print_summary(results)
    
    print(f"\nDetailed results saved in: report_cnn_assignment2_taskF.txt")

if __name__ == "__main__":
    main()
# %%
