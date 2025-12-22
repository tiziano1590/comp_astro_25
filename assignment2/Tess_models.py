## mahdis:this is model file for random forest, linear model and SVM model
## task D - random forest
## task E - linear regression
## task E - support vector machine

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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

import joblib

##mahdis nots: why we do not split to the validation set here?
## here, they probably simplified the pipeline because Random Forest is already stable and this is more of a baseline experiment, not a full hyperparameter optimization.

print("Environment ready.")

class TESSModels:
    def __init__(self, csv_path='tess_data.csv', n_bins=1000, use_scaler=False,
                 samples_per_class=350, random_state=42):
        self.csv_path = csv_path
        self.n_bins = n_bins
        self.use_scaler = use_scaler
        self.samples_per_class = samples_per_class
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.lm = None
        self.scaler = None
        self.best_thresh = None

    ## balanced dataset function
    def create_balanced_dataset(self, X, y, samples_per_class=None):
        print("\n" + "="*70)
        print("CREATING BALANCED DATASET")
        print("="*70)

        if samples_per_class is None:
            samples_per_class = self.samples_per_class
        
        X0 = X[y == 0]
        X1 = X[y == 1]
        print(f"Original - Class 0: {len(X0)}, Class 1: {len(X1)}")

        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]
        
            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
                aug_type = np.random.rand()

                if aug_type < 0.25:
                    X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    shifts = np.random.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)
                X_result.append(X_aug)
            X_final = np.vstack(X_result)
            return X_final[:n_target]
    
        X0_bal = augment_to_target(X0, samples_per_class)
        X1_bal = augment_to_target(X1, samples_per_class)
        print(f"Balanced - Class 0: {len(X0_bal)}, Class 1: {len(X1_bal)}")
    
        X_bal = np.vstack([X0_bal, X1_bal])
        y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
    
        idx = np.arange(len(X_bal))
        np.random.shuffle(idx)
        return X_bal[idx], y_bal[idx]
    
    ## load data function
    def load_data(self, force_reload=False):

        ## return chached data if already loaded
        if hasattr(self, "cached_data") and not force_reload:
            return self.cached_data

        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)

        df = pd.read_csv(self.csv_path)
        print(f"Dataset: {df.shape[0]} samples")

        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values

        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]

        print("Original distribution:")
        print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        if (y==0).sum() > 0:
            print(f"  Ratio: {(y==1).sum() / (y==0).sum():.2f}:1")

        X_train, X_test, y_train, y_test, X_err_train, X_err_test, idx_train, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)), 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=y
        )
        print(f"Initial split - Train: {len(X_train)}, Test: {len(X_test)}")

        ## balancing training set
        X_train, y_train = self.create_balanced_dataset(X_train, y_train)

        if self.use_scaler:
            print("\n" + "="*70)
            print("STANDARDIZATION")
            print("="*70)
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            print(f"Train: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
            print(f"Test:  mean={X_test.mean():.6f}, std={X_test.std():.6f}")

        metadata_test = metadata.iloc[idx_test].reset_index(drop=True)
        print(f"Final - X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Train dist: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")

        self.cached_data = (
            X_train,       # scaled & balanced training data
            X_test,        # scaled test data
            y_train,       # balanced labels
            y_test,        # original test labels
            metadata_test, # test metadata
            X_test.copy(), # raw X_test copy for plotting
            X_err_test,    # test errors
            self.scaler
        )

        return self.cached_data
    ## built random forest model function
    def build_random_forest(self, n_estimators=500, max_depth=None, min_samples_leaf=1,
                            max_features='sqrt', bootstrap=True, class_weight=None,
                            n_jobs=-1, oob_score=True):
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            bootstrap=bootstrap, class_weight=class_weight,
            random_state=self.random_state, n_jobs=n_jobs, oob_score=oob_score
        )
        return rf

    ## biuld linear model function
    def build_linear_model(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=1000):
        lm = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=self.random_state
        )
        return lm
    
    ## biuld SVM model function
    def build_svm_model(self, kernel='rbf', C=1.0, gamma='scale', probability=True):
        """
        Build a Support Vector Classifier.
        Args:
            kernel: 'linear', 'rbf', etc.
            C: regularization parameter
            gamma: kernel coefficient
            probability: if True, enables predict_proba
        """
        svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=self.random_state)
        return svm

    ## train model function
    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            print(f"OOB Score: {model.oob_score_:.4f}")
        return model     
      
    ## evaluate function
    def evaluate_with_optimal_threshold(self, model, X_test, y_test):
        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("="*70)
    
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
    
        y_pred_default = (proba >= 0.5).astype(int)
        y_pred_best = (proba >= best_thresh).astype(int)
    
        auc = roc_auc_score(y_test, proba)
        acc_default = accuracy_score(y_test, y_pred_default)
        acc_best = accuracy_score(y_test, y_pred_best)
    
        print(f"Optimal threshold: {best_thresh:.4f} (default=0.5)")
        print(f"  At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"Accuracy @{best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)")
    
        print("\nClassification report (optimal threshold):")
        print(classification_report(y_test, y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0))
    
        self.best_thresh = best_thresh
        return y_pred_best, proba, best_thresh, (fpr, tpr, thresholds)


    ## plotting functions (1. confusion matrix, 2. roc curve, 3. pr curve, 4. probability histograms)
    def plot_confusion_matrix_image(self, y_true, y_pred, threshold, save_prefix ='model'):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
               yticks=np.arange(2),
               xticklabels=['Non-Planet', 'Planet'],
               yticklabels=['Non-Planet', 'Planet'],
               xlabel='Predicted',
               ylabel='True',
               title=f'{save_prefix} - Confusion Matrix (threshold={threshold:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
        plt.tight_layout()

        save_path = f'confusion_matrix_{save_prefix}.png'

        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_roc_curve(self, y_true, proba, save_prefix ='model'):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={auc:.4f})')
        plt.plot([0,1],[0,1], linestyle='--',label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{save_prefix} - ROC Curve (AUC={auc:.4f})')
        plt.grid(alpha=0.3)
        plt.legend()
        save_path=f'roc_curve_{save_prefix}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_pr_curve(self, y_true, proba,save_prefix ='model'):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{save_prefix} - Precision-Recall Curve (AP={ap:.4f})')
        plt.grid(alpha=0.3)
        save_path=f'pr_curve_{save_prefix}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_probability_histograms(self, y_true, proba, save_prefix ='model'):
        plt.figure(figsize=(6,5))
        plt.hist(proba[y_true==0], bins=30, alpha=0.6, label='Non-Planet', density=True)
        plt.hist(proba[y_true==1], bins=30, alpha=0.6, label='Planet', density=True)
        plt.xlabel('Predicted Probability (class=1)')
        plt.ylabel('Density')
        plt.title(f'{save_prefix} - Predicted Probability Distributions by Class')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path=f'probability_hist_{save_prefix}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_lightcurve_sample(self, idx, X_test_standardized, X_err_test, metadata_test, scaler=None, proba=None, y_true=None, y_pred=None, save_prefix='model'):
        x_std = X_test_standardized[idx].reshape(1, -1)
        if scaler is not None:
            x_orig = scaler.inverse_transform(x_std).flatten()
        else:
            x_orig = x_std.flatten()
        yerr = X_err_test[idx]

        fig, ax = plt.subplots(figsize=(10,4))
        ax.errorbar(np.arange(len(x_orig)), x_orig, yerr=yerr, fmt='o', markersize=2, alpha=0.6)
        ax.axhline(np.median(x_orig), linestyle='--', linewidth=1)
        ax.set_xlabel('Time Bin')
        ax.set_ylabel('Flux')

        toi = metadata_test.loc[idx, 'toi_name']
        tic = metadata_test.loc[idx, 'tic']
        disp = metadata_test.loc[idx, 'disp']
        sector = metadata_test.loc[idx, 'sector']

        tstr = f'TOI {toi} (TIC {tic}, {disp}) - Sector {sector}'
        if proba is not None and y_true is not None and y_pred is not None:
            pred_str = 'Transit' if y_pred[idx]==1 else 'Non-Transit'
            true_str = 'Transit' if y_true[idx]==1 else 'Non-Transit'
            tstr += f'\nTrue: {true_str} | Pred: {pred_str} (p={proba[idx]:.3f})'

        ax.set_title(tstr)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = f"sample lightcurves_{save_prefix}_{idx}.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
        plt.close(fig)
    ## run random forest model function
    def run_random_forest(self, n_estimators=500, max_depth=None, min_samples_leaf=1,
            max_features='sqrt', bootstrap=True, class_weight=None,
            n_jobs=-1, oob_score=True, preview_samples=4):
        self.X_train, self.X_test, self.y_train, self.y_test, self.metadata_test, \
        self.X_test_std_copy, self.X_err_test, self.scaler = self.load_data()

        self.rf = self.build_random_forest(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            bootstrap=bootstrap, class_weight=class_weight,
            n_jobs=n_jobs, oob_score=oob_score
        )
        self.rf = self.train_model(self.rf, self.X_train, self.y_train)

        self.y_pred_opt, self.proba_test, self.best_thresh, _ = \
            self.evaluate_with_optimal_threshold(self.rf, self.X_test, self.y_test)

        # Plots
        self.plot_confusion_matrix_image(self.y_test, self.y_pred_opt, self.best_thresh, save_prefix='rf')
        self.plot_roc_curve(self.y_test, self.proba_test, save_prefix='rf')
        self.plot_pr_curve(self.y_test, self.proba_test, save_prefix='rf')
        self.plot_probability_histograms(self.y_test, self.proba_test, save_prefix='rf')

        # Optional preview of light curves can be added here
        num_samples = min(preview_samples, len(self.X_test_std_copy))
        idxs = np.random.choice(len(self.X_test_std_copy), num_samples, replace=False)
        for i in idxs:
            self.plot_lightcurve_sample(
                i, self.X_test_std_copy, self.X_err_test, self.metadata_test,
                scaler=self.scaler, proba=self.proba_test, y_true=self.y_test,
                y_pred=self.y_pred_opt, save_prefix='rf'
            )
            print("\nRandom Forest run completed!")
        return self.rf, self.y_pred_opt, self.proba_test, self.best_thresh, _

    ## run function LINEAR MODEL
    def run_linear_model(self, n_bins=1000, use_scaler=False, samples_per_class=350,
                     random_state=42):
    
        # Load data
        X_train, X_test, y_train, y_test, metadata_test, X_test_std_copy, X_err_test, scaler = self.load_data()
    
        # logistic regression model
        lm = LogisticRegression(random_state=random_state, max_iter=1000)
    
        # train model
        lm = self.train_model(lm, X_train, y_train)
    
        # evaluate model
        y_pred_opt, proba_test, best_thresh, roc_tuple = self.evaluate_with_optimal_threshold(lm, X_test, y_test)
    
        #visualizations (plots)
        self.plot_confusion_matrix_image(y_test, y_pred_opt, best_thresh, save_prefix='lm')
        self.plot_roc_curve(y_test, proba_test, save_prefix='lm')
        self.plot_pr_curve(y_test, proba_test, save_prefix='lm')
        self.plot_probability_histograms(y_test, proba_test, save_prefix='lm')

        num_samples = min(4, len(X_test_std_copy))
        idxs = np.random.choice(len(X_test_std_copy), num_samples, replace=False)
        for i in idxs:
            self.plot_lightcurve_sample(
                i, X_test_std_copy, X_err_test, metadata_test,
                scaler=scaler, proba=proba_test, y_true=y_test, y_pred=y_pred_opt,
                save_prefix='lm'
            )

        print("\nLinear model run completed!")
        return lm, y_pred_opt, proba_test, best_thresh, roc_tuple
    
    ## run SVM model function
    def run_svm_model(self, n_bins=1000, use_scaler=False, samples_per_class=350,
                  random_state=42, kernel='rbf', C=1.0, gamma='scale'):
        # Load data
        X_train, X_test, y_train, y_test, metadata_test, X_test_std_copy, X_err_test, scaler = self.load_data()

        # SVM model
        svm_model = SVC(random_state=random_state, kernel=kernel, C=C, gamma=gamma, probability=True)
    
        # Train model
        svm_model = self.train_model(svm_model, X_train, y_train)
    
        # Evaluate model
        y_pred_opt, proba_test, best_thresh, roc_tuple = self.evaluate_with_optimal_threshold(svm_model, X_test, y_test)
    
        # Visualizations
        self.plot_confusion_matrix_image(y_test, y_pred_opt, best_thresh, save_prefix='svm')
        self.plot_roc_curve(y_test, proba_test, save_prefix='svm')
        self.plot_pr_curve(y_test, proba_test, save_prefix='svm')
        self.plot_probability_histograms(y_test, proba_test, save_prefix='svm')
    
        # Optional: sample light curves
        num_samples = min(4, len(X_test_std_copy))
        idxs = np.random.choice(len(X_test_std_copy), num_samples, replace=False)
        for i in idxs:
            self.plot_lightcurve_sample(
                i, X_test_std_copy, X_err_test, metadata_test,
                scaler=scaler, proba=proba_test, y_true=y_test, y_pred=y_pred_opt,
                save_prefix='svm'
            )
    
        print("\nSVM model run completed!")
        return svm_model, y_pred_opt, proba_test, best_thresh, roc_tuple

    ## save results function
    def save_results_and_samples(self, model, best_thresh, save_prefix='model'):
        # Persist the trained model and optimal threshold
        joblib.dump(model, f'tess_{save_prefix}_model.joblib')
        np.save(f'{save_prefix}_optimal_threshold.npy', best_thresh)
        print(f"Saved: tess_{save_prefix}_model.joblib, {save_prefix}_optimal_threshold.npy")
    

 
def main(csv_path='tess_data.csv', n_bins=1000, use_scaler=True, samples_per_class=None):
    # Initialize the linear model pipeline
    pipeline = TESSModels(
        csv_path=csv_path,
        n_bins=n_bins,
        use_scaler=use_scaler,
        samples_per_class=samples_per_class if samples_per_class is not None else 350
    )
    # Run the random forest model
    rf, y_pred_rf, proba_rf, best_thresh_rf = pipeline.run_random_forest()
    pipeline.save_results_and_samples(rf, best_thresh_rf, save_prefix='rf')
    # Run the linear model
    lm, y_pred_lm, proba_lm, best_thresh_lm, roc_lm = pipeline.run_linear_model()
    pipeline.save_results_and_samples(lm, best_thresh_lm, save_prefix='lm')

    # Run the SVM model
    svm, y_pred_svm, proba_svm, best_thresh_svm, roc_svm = pipeline.run_svm_model()
    pipeline.save_results_and_samples(svm, best_thresh_svm, save_prefix='svm')
    
# main()

    
# %%
