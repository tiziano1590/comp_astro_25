## mahdis: model for CNN 
## task F : model for tess transit light curves classification by deeep learning

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import IPython
    working_directory = "/".join(
            IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]
        )
    print("Setting working directory to: ", working_directory)
    print(os.chdir(working_directory))
except Exception as e:
    print("It was impossible to set your directory as the current one because of the following message")
    print(e)
    print("The working directory is: ", os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve ,precision_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("FINAL TESS CLASSIFICATION")
print("="*70)

# Optional: make TF less eager to pre-allocate all GPU memory (if using GPU)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
except Exception as e:
    print("GPU setup note:", e)

print("TF version:", tf.__version__)

class TESSTransitCNN:
    """
    CNN-based exoplanet detector for TESS light curves.
    Refactored from transit_analysis_deep_learning.ipynb.
    """

    def __init__(self,csv_path = 'tess_data.csv',
                n_bins = 1000 , gamma = 2.5, alpha = 0.75,
                batch_size = 64, samples_per_class = 400,
                config_name='Default'):
        # TessTransitCNN(params**)
        self.csv_path = csv_path
        self.n_bins = n_bins
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.config_name = config_name

        self.model = None
        self.scaler = None
        self.threshold = 0.5
        self.history = None
        self.results = {}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metadata_test = None
        self.kernel_size = 3
    
    def set_params(self,params):
        self.gamma = params['gamma']
        self.n_bins = params['n_bins']
        self.alpha = params['alpha']
        self.batch_size = params['batch_size']
        self.samples_per_class = params['samples_per_class']
        self.kernel_size = params['kernel_size']
        self.threshold = params['threshold']
        
        
    def focal_loss(self):
        gamma = self.gamma
        alpha = self.alpha

        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            focal_weight = alpha_factor * K.pow(1 - pt, gamma)
            bce = -K.log(pt)
            return K.mean(focal_weight * bce)
        
        return focal_loss_fixed
    


    def create_balanced_dataset(self, X, y):
    
        print("\n" + "="*70)
        print("CREATING BALANCED DATASET")
        print("="*70)

        samples_per_class = self.samples_per_class
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
    
        print(f"Original - Class 0: {len(X_class0)}, Class 1: {len(X_class1)}")
    
        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]
        
            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                # number we still need (cap to avoid oversampling too big chunks)
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
                
                aug_type = np.random.rand()
                if aug_type < 0.25:
                    # Additive Gaussian noise
                    X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    # Multiplicative scaling
                    scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    # Circular shift (time shift)
                    shifts = np.random.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    # Mild combo: small scale + small noise
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)
            
                X_result.append(X_aug)
        
            X_final = np.vstack(X_result)
            return X_final[:n_target]
    
        X0_bal = augment_to_target(X_class0, samples_per_class)
        X1_bal = augment_to_target(X_class1, samples_per_class)
    
        print(f"Balanced - Class 0: {len(X0_bal)}, Class 1: {len(X1_bal)}")
    
        X_balanced = np.vstack([X0_bal, X1_bal])
        y_balanced = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
    
        # Shuffle
        idx = np.arange(len(X_balanced))
        np.random.shuffle(idx)
    
        return X_balanced[idx], y_balanced[idx]

    def load_data(self):

        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        csv_path = self.csv_path
        n_bins = self.n_bins

        ## Load data
        df = pd.read_csv(csv_path)
        print(f"Dataset: {df.shape[0]} samples")
        

        ## Extract features and labels (x = light curve fluxes , x_err = flux errors, y = labels (planet , non-planet))
        flux_cols = [f'flux_{i:04d}' for i in range(n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values
        
        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]
        
        print("\nOriginal distribution:")
        print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        if (y==0).sum() > 0:
            print(f"  Ratio: {(y==1).sum() / (y==0).sum():.2f}:1")
        
        # Train/test split (keep errors aligned; stratify to preserve class ratio)
        X_train, X_test, y_train, y_test, X_err_train, X_err_test, idx_train, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"\nInitial split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Balance training set
        X_train, y_train = self.create_balanced_dataset(X_train, y_train)
        
        # Standardize (fit on train, apply to test)
        print("\n" + "="*70)
        print("STANDARDIZATION")
        print("="*70)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Train: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
        print(f"Test:  mean={X_test.mean():.6f}, std={X_test.std():.6f}")
        
        # Reshape for Conv1D: (samples, timesteps, channels)
        X_train = X_train.reshape(-1, n_bins, 1)
        X_test = X_test.reshape(-1, n_bins, 1)
        
        metadata_test = metadata.iloc[idx_test].reset_index(drop=True)
        
        print(f"\nFinal - X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Train dist: 0={( y_train==0).sum()}, 1={(y_train==1).sum()}")
        
        # Return standardized test for model input, but also return the standardized
        # copy (X_test_orig) so we can inverse-transform for plotting with error bars.
        return X_train, X_test, y_train, y_test, metadata_test, X_test.copy(), X_err_test, scaler
    
    def build_cnn_from_config(self, config_type='default'):
        """Build CNN based on configuration type"""
        if config_type == 'default':
            return self.build_default_cnn()
        elif config_type == 'config1':
            return self.build_config1_cnn()
        elif config_type == 'config2':
            return self.build_config2_cnn()
        elif config_type == 'config3':
            return self.build_config3_cnn()
        else:
            return self.build_default_cnn()
    
    def build_default_cnn(self):
        """Your original CNN"""
        n_bins = self.n_bins
        focal_loss = self.focal_loss()
        
        model = models.Sequential([
            layers.Input(shape=(n_bins, 1)),
            layers.Conv1D(64, kernel_size=self.kernel_size, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=self.kernel_size, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, kernel_size=self.kernel_size, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=focal_loss,
            metrics=['accuracy', keras.metrics.Precision(name='precision')]
        )
        
        print(f"\nDefault CNN Architecture:")
        model.summary()
        print(f"Using Focal Loss (gamma={self.gamma}, alpha={self.alpha})")

        return model
    
    def build_config1_cnn(self):
        """Configuration 1: Wider network"""
        n_bins = self.n_bins
        focal_loss = self.focal_loss()
        
        model = models.Sequential([
            layers.Input(shape=(n_bins, 1)),
            layers.Conv1D(128, kernel_size=7, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.4),
            
            layers.Conv1D(256, kernel_size=5, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.4),
            
            layers.Conv1D(512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.5),
            
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=focal_loss,
            metrics=['accuracy', keras.metrics.Precision(name='precision')]
        )
        
        return model
    
    def build_config2_cnn(self):
        """Configuration 2: Deeper network"""
        n_bins = self.n_bins
        focal_loss = self.focal_loss()
        
        model = models.Sequential([
            layers.Input(shape=(n_bins, 1)),
            layers.Conv1D(32, kernel_size=5, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=focal_loss,
            metrics=['accuracy', keras.metrics.Precision(name='precision')]
        )
        
        return model
    
    def build_config3_cnn(self):
        """Configuration 3: Simpler network"""
        n_bins = self.n_bins
        focal_loss = self.focal_loss()
        
        model = models.Sequential([
            layers.Input(shape=(n_bins, 1)),
            layers.Conv1D(32, kernel_size=7, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.4),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss,
            metrics=['accuracy', keras.metrics.Precision(name='precision')]
        )
        
        return model
    
    def build_simple_cnn(self):
        """Build CNN based on configuration name"""
        print("\n" + "="*70)
        print(f"BUILDING CNN - Configuration: {self.config_name}")
        print("="*70)
        
        if self.config_name == 'Default':
            return self.build_default_cnn()
        elif self.config_name == 'Config1':
            return self.build_config1_cnn()
        elif self.config_name == 'Config2':
            return self.build_config2_cnn()
        elif self.config_name == 'Config3':
            return self.build_config3_cnn()
        else:
            return self.build_default_cnn()

    
    def train_model(self, epochs=100):

        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_test
        y_val = self.y_test
        model = self.model
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                'best_model_final.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        return history
    

    def evaluate_with_optimal_threshold(self):

        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("="*70)

        X_test = self.X_test
        y_test = self.y_test
        model = self.model
        
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\nOptimal threshold: {optimal_threshold:.4f} (default=0.5)")
        print(f"  At this threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")
        
        # Predictions with optimal vs default thresholds
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        acc_optimal = accuracy_score(y_test, y_pred_optimal)
        acc_default = accuracy_score(y_test, y_pred_default)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\nResults:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Accuracy (default threshold=0.5): {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"  Accuracy (optimal threshold={optimal_threshold:.4f}): {acc_optimal:.4f} ({acc_optimal*100:.2f}%)")
        
        print("\nWith optimal threshold:")
        print(classification_report(y_test, y_pred_optimal,
                                    target_names=['Non-Planet', 'Planet'],
                                    digits=4,
                                    zero_division=0))
        
        print("\nPrediction distribution (optimal threshold):")
        print(f"  Predicted 0: {(y_pred_optimal == 0).sum()}")
        print(f"  Predicted 1: {(y_pred_optimal == 1).sum()}")
        print("True distribution:")
        print(f"  True 0: {(y_test == 0).sum()}")
        print(f"  True 1: {(y_test == 1).sum()}")
        
        return y_pred_optimal, y_pred_proba, optimal_threshold
    
    
    def plot_lightcurves_with_predictions(self,X_test_orig, X_err_test, y_pred, y_pred_proba,
                                          metadata_test, scaler, threshold, n_samples=6,
                                          save_path='sample_lightcurves_predictions.png'):
        
        print("\n" + "="*70)
        print(f"PLOTTING LIGHTCURVES WITH PREDICTIONS (n={n_samples})")
        print("="*70)

        y_test = self.y_test
       

        
        n_samples = min(n_samples, len(X_test_orig))
        
        # Select diverse samples: correct/incorrect for both classes
        correct_planet = np.where((y_test == 1) & (y_pred == 1))[0]
        incorrect_planet = np.where((y_test == 1) & (y_pred == 0))[0]
        correct_nonplanet = np.where((y_test == 0) & (y_pred == 0))[0]
        incorrect_nonplanet = np.where((y_test == 0) & (y_pred == 1))[0]
        
        selected_idx = []
        per_category = max(1, n_samples // 4)
        
        for idx_list in [correct_planet, incorrect_planet, correct_nonplanet, incorrect_nonplanet]:
            if len(idx_list) > 0:
                n_select = min(per_category, len(idx_list))
                selected_idx.extend(np.random.choice(idx_list, n_select, replace=False))
        
        while len(selected_idx) < n_samples:
            remaining = list(set(range(len(y_test))) - set(selected_idx))
            if remaining:
                selected_idx.append(np.random.choice(remaining))
            else:
                break
        
        selected_idx = np.array(selected_idx[:n_samples])
        
        # Figure layout
        n_cols = 2
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for plot_i, idx in enumerate(selected_idx):
            ax = axes[plot_i]
            
            # Inverse transform to original scale for plotting
            flux_norm = X_test_orig[idx].flatten()
            flux_err = X_err_test[idx]
            flux_original = scaler.inverse_transform(flux_norm.reshape(1, -1)).flatten()
            
            time_bins = np.arange(len(flux_original))
            
            # Metadata
            toi_name = metadata_test.loc[idx, 'toi_name']
            tic = metadata_test.loc[idx, 'tic']
            disp = metadata_test.loc[idx, 'disp']
            sector = metadata_test.loc[idx, 'sector']
            
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            pred_prob = y_pred_proba[idx]
            
            is_correct = (true_label == pred_label)
            true_str = 'Transit' if true_label == 1 else 'Non-Transit'
            pred_str = 'Transit' if pred_label == 1 else 'Non-Transit'
            
            # Errorbar plot
            ax.errorbar(time_bins, flux_original, yerr=flux_err, fmt='o', markersize=2,
                        ecolor='gray', elinewidth=0.5, capsize=0, alpha=0.6, label='Data')
            
            # Baseline median
            baseline = np.median(flux_original)
            ax.axhline(baseline, linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
            
            ax.set_xlabel('Time Bin', fontsize=10, fontweight='bold')
            ax.set_ylabel('Flux (original scale)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            status_symbol = '✓' if is_correct else '✗'
            color = 'green' if is_correct else 'red'
            title = (f'TOI {toi_name} (TIC {tic}, {disp}) - TESS Sector {sector}\n'
                    f'True: {true_str} | Pred: {pred_str} (p={pred_prob:.3f}) {status_symbol}')
            ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)
        
        # Hide unused axes
        for j in range(n_samples, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f'Sample Light-curve Predictions (Threshold={threshold:.3f})',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


    def plot_all(self, y_pred, y_pred_proba,
                X_test_orig=None, X_err_test=None, scaler=None):

        print("\n" + "="*70)
        print("VISUALIZATIONS")
        print("="*70)

        y_test = self.y_test
        metadata_test = self.metadata_test
        history = self.history
        threshold = self.threshold
        
        # Confusion matrix (Matplotlib-only)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            xlabel='Predicted', ylabel='True',
            title=f'Confusion Matrix (threshold={threshold:.3f})')
        
        # Add counts and percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center', color='black', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_final.png', dpi=300)
        print("Saved: confusion_matrix_final.png")
        plt.close()
        
        # Training history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [('loss', 'Loss'), ('accuracy', 'Accuracy'),
                ('auc', 'AUC'), ('recall', 'Recall')]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric in history.history and f'val_{metric}' in history.history:
                ax.plot(history.history[metric], label='Train', linewidth=2)
                ax.plot(history.history[f'val_{metric}'], label='Val', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.set_title(f'{title} vs Epoch', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.suptitle('Training History - Final Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history_final.png', dpi=300)
        print("Saved: training_history_final.png")
        plt.close()
        
        # Optional: light-curve panel
        if X_test_orig is not None and X_err_test is not None and scaler is not None:
            self.plot_lightcurves_with_predictions(X_test_orig, X_err_test, y_pred, 
                                            y_pred_proba, metadata_test, scaler, threshold, n_samples=6)
        
    def run(self):
        # Load and preprocess data
        (self.X_train, self.X_test,
         self.y_train, self.y_test,
         self.metadata_test,
         X_test_orig, X_err_test,
         self.scaler ) = self.load_data()
        
        
        # Build model
        config_type = self.config_name.lower()
        self.model = self.build_cnn_from_config(config_type)
        
        # Train model
        self.history = self.train_model(epochs=100)
        
        # Evaluate model with optimal threshold
        y_pred, y_pred_proba, optimal_threshold = self.evaluate_with_optimal_threshold()
        self.threshold = optimal_threshold
        
        # Store results for reporting

        precision = precision_score(self.y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(self.y_test, y_pred) 
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.results = {
            'config_name': self.config_name,
            'precision': precision,
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'optimal_threshold': optimal_threshold,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Plot results (only for default config to avoid too many plots)
        if self.config_name == 'Default':
            self.plot_all(y_pred, y_pred_proba, X_test_orig, X_err_test, self.scaler)
        
        # Save model
        self.model.save(f'tess_model_{self.config_name}.keras')
        print("="*70)
        
        return self.results



def main():

    detector = TESSTransitCNN()
    detector.run()

# %%
