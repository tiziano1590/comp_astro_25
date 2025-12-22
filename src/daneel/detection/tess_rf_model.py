# daneel - Random Forest Detector Module
import sys
import os

# Add path to your main repo if needed
main_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, main_repo_path)

from Tess_models import TESSModels

class RandomForestDetector:
    def __init__(self, parameters):
        """
        Initialize from parameters dict.
        
        Args:
            parameters: Dictionary loaded from parameters.yaml
        """
        self.params = parameters
        
        # Extract key parameters with defaults
        self.csv_path = parameters.get('dataset_path', 'tess_data.csv')
        self.n_bins = parameters.get('n_bins', 1000)
        self.use_scaler = parameters.get('use_scaler', True)
        self.samples_per_class = parameters.get('samples_per_class', 350)
        self.random_state = parameters.get('random_state', 42)
        
        # Random Forest specific parameters
        self.rf_params = parameters.get('random_forest', {})
    
    def detect(self):
        """Run Random Forest detection on TESS data using TESSModels."""
    
        # Initialize pipeline
        print(f"Initializing TESSModels pipeline with dataset: {self.csv_path}")
        pipeline = TESSModels(
            csv_path=self.csv_path,
            n_bins=self.n_bins,
            use_scaler=self.use_scaler,
            samples_per_class=self.samples_per_class,
            random_state=self.random_state
        )
    
        # Run Random Forest with RF-specific parameters
        rf, y_pred, proba, best_thresh, _ = pipeline.run_random_forest(**self.rf_params)
        
        # Save trained model and threshold
        pipeline.save_results_and_samples(rf, best_thresh, save_prefix='rf')
        
        print("Random Forest detection completed successfully.")
        
        # Return results for further use if needed
        return rf, y_pred, proba, best_thresh
