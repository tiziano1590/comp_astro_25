#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class TESSRandomForest:
    def __init__(self, csv_path='tess_data.csv', n_bins=1000, use_scaler=False,
                 samples_per_class=350, random_state=42):
        self.csv_path = csv_path
        self.n_bins = n_bins
        self.use_scaler = use_scaler
        self.samples_per_class = samples_per_class
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.rf = None
        self.scaler = None
        self.best_thresh = None

    # Balanced dataset
    def create_balanced_dataset(self, X, y, samples_per_class=None):
        if samples_per_class is None:
            samples_per_class = self.samples_per_class
        
        X0 = X[y == 0]
        X1 = X[y == 1]

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
        
        X_bal = np.vstack([X0_bal, X1_bal])
        y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
        idx = np.arange(len(X_bal))
        np.random.shuffle(idx)
        return X_bal[idx], y_bal[idx]

    # Load data 
    def load_data(self):
        df = pd.read_csv(self.csv_path)
        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values

        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]

        X_train, X_test, y_train, y_test, X_err_train, X_err_test, idx_train, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)), test_size=0.2, random_state=self.random_state, stratify=y
        )

        X_train, y_train = self.create_balanced_dataset(X_train, y_train)

        if self.use_scaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        metadata_test = metadata.iloc[idx_test].reset_index(drop=True)

        return X_train, X_test, y_train, y_test, metadata_test, X_test.copy(), X_err_test, self.scaler

    #Build Random Forest
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

    # Train model
    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            print(f"OOB Score: {model.oob_score_:.4f}")
        return model

    # Evaluate
    def evaluate_with_optimal_threshold(self, model, X_test, y_test):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        y_pred_best = (proba >= best_thresh).astype(int)

        print(f"Optimal threshold: {best_thresh:.4f}")
        print(classification_report(y_test, y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0))
        self.best_thresh = best_thresh
        return y_pred_best, proba, best_thresh, (fpr, tpr, thresholds)

    # Plotting methods 
    def plot_confusion_matrix_image(self, y_true, y_pred, threshold, save_path='confusion_matrix_rf.png'):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2), yticks=np.arange(2),
               xticklabels=['Non-Planet', 'Planet'], yticklabels=['Non-Planet', 'Planet'],
               xlabel='Predicted', ylabel='True', title=f'Confusion Matrix (threshold={threshold:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def plot_roc_curve(self, y_true, proba, save_path='roc_curve_rf.png'):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={auc:.4f})')
        plt.plot([0,1],[0,1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC={auc:.4f})')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_pr_curve(self, y_true, proba, save_path='pr_curve_rf.png'):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AP={ap:.4f})')
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_probability_histograms(self, y_true, proba, save_path='probability_hist_RF.png'):
        plt.figure(figsize=(6,5))
        plt.hist(proba[y_true==0], bins=30, alpha=0.6, label='Non-Planet', density=True)
        plt.hist(proba[y_true==1], bins=30, alpha=0.6, label='Planet', density=True)
        plt.xlabel('Predicted Probability (class=1)')
        plt.ylabel('Density')
        plt.title('Predicted Probability Distributions by Class')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def plot_lightcurve_sample(self, idx, X_test_standardized, X_err_test, metadata_test,
                           scaler=None, proba=None, y_true=None, y_pred=None,
                           save_prefix='sample_lightcurve_RF'):
        # Make a single-figure plot for one sample (no subplots)
        x_std = X_test_standardized[idx].reshape(1, -1)
        if scaler is not None:
            x_orig = scaler.inverse_transform(x_std).flatten()
        else:
            x_orig = x_std.flatten()
        yerr = X_err_test[idx]

        fig, ax = plt.subplots(figsize=(10, 4))
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
            pred_str = 'Transit' if y_pred[idx] == 1 else 'Non-Transit'
            true_str = 'Transit' if y_true[idx] == 1 else 'Non-Transit'
            tstr += f'\nTrue: {true_str} | Pred: {pred_str} (p={proba[idx]:.3f})'

        ax.set_title(tstr)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = f"{save_prefix}_{idx}.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
        plt.close(fig)


    #Run the full pipeline
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
        self.plot_confusion_matrix_image(self.y_test, self.y_pred_opt, self.best_thresh)
        self.plot_roc_curve(self.y_test, self.proba_test)
        self.plot_pr_curve(self.y_test, self.proba_test)
        self.plot_probability_histograms(self.y_test, self.proba_test)

        # Optional preview of light curves can be added here
        num_samples = min(preview_samples, len(self.X_test_std_copy))
        idxs = np.random.choice(len(self.X_test_std_copy), num_samples, replace=False)
        for i in idxs:
            self.plot_lightcurve_sample(
                i, self.X_test_std_copy, self.X_err_test, self.metadata_test,
                scaler=self.scaler, proba=self.proba_test, y_true=self.y_test,
                y_pred=self.y_pred_opt, save_prefix='sample_lightcurve_RF'
            )
            print("\nRandom Forest run completed!")
            return self.rf, self.y_pred_opt, self.proba_test, self.best_thresh

    # Save results
    def save_results_and_samples(self, model, y_pred, proba, best_thresh, save_prefix='rf'):
        # Save a few single-sample light curves
        num_samples = min(4, len(self.X_test_std_copy))
        idxs = np.random.choice(len(self.X_test_std_copy), num_samples, replace=False)
        for i in idxs:
            self.plot_lightcurve_sample(
                i, self.X_test_std_copy, self.X_err_test, self.metadata_test,
                scaler=self.scaler, proba=proba, y_true=self.y_test, y_pred=y_pred,
                save_prefix=f'sample_lightcurve_{save_prefix}'
            )

        # Persist the trained model and optimal threshold
        joblib.dump(model, f'tess_{save_prefix}_model.joblib')
        np.save(f'{save_prefix}_optimal_threshold.npy', best_thresh)
        print(f"Saved: tess_{save_prefix}_model.joblib, {save_prefix}_optimal_threshold.npy")





# %%
