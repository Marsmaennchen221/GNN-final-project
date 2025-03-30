import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import seaborn as sns
from scipy import stats

class WGANEvaluator:
    def __init__(self, wgan_model, X_train_normalized, train_labels):
        """
        Initialize evaluator with the trained WGAN model and reference data
        
        Parameters:
        -----------
        wgan_model : ClassConditionalWGAN
            The trained WGAN model
        X_train_normalized : numpy.ndarray
            Normalized training data of shape (n_samples, 61, 21)
        train_labels : numpy.ndarray
            Class labels for training data of shape (n_samples,)
        """
        self.wgan = wgan_model
        self.X_train = X_train_normalized
        self.train_labels = train_labels
        
        # Extract shape information
        self.n_samples, self.n_timesteps, self.n_features = X_train_normalized.shape
        
    def generate_samples_for_class(self, class_idx, n_samples=100):
        """
        Generate samples for a specific class
        
        Parameters:
        -----------
        class_idx : int
            Class index to generate samples for
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        numpy.ndarray
            Generated samples of shape (n_samples, n_timesteps, n_features)
        """
        return self.wgan.generate_samples(class_idx, n_samples)
    
    def get_real_samples_for_class(self, class_idx):
        """
        Get real samples for a specific class
        
        Parameters:
        -----------
        class_idx : int
            Class index to get samples for
            
        Returns:
        --------
        numpy.ndarray
            Real samples of shape (n_class_samples, n_timesteps, n_features)
        """
        class_mask = (self.train_labels == class_idx)
        return self.X_train[class_mask]
    
    def plot_time_series_comparison(self, class_idx, feature_idx, n_samples=5, figsize=(15, 10)):
        """
        Plot comparison between real and generated time series for a specific feature
        
        Parameters:
        -----------
        class_idx : int
            Class index to compare
        feature_idx : int
            Feature index to plot (0-20)
        n_samples : int
            Number of samples to plot
        figsize : tuple
            Figure size
        """
        # Get real samples
        real_samples = self.get_real_samples_for_class(class_idx)
        
        # Generate fake samples
        fake_samples = self.generate_samples_for_class(class_idx, n_samples)
        
        # Prepare figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot real samples
        ax = axes[0]
        for i in range(min(n_samples, len(real_samples))):
            ax.plot(real_samples[i, :, feature_idx], alpha=0.7, linewidth=1)
        ax.set_title(f'Real Samples - Class {class_idx} - Feature {feature_idx}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Plot generated samples
        ax = axes[1]
        for i in range(n_samples):
            ax.plot(fake_samples[i, :, feature_idx], alpha=0.7, linewidth=1)
        ax.set_title(f'Generated Samples - Class {class_idx} - Feature {feature_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compute_standard_deviation_comparison(self, class_idx):
        """
        Compare standard deviations between real and generated data
        
        Parameters:
        -----------
        class_idx : int
            Class index to compare
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with standard deviation comparison
        """
        # Get real samples
        real_samples = self.get_real_samples_for_class(class_idx)
        
        # Generate fake samples with similar count
        n_real = len(real_samples)
        fake_samples = self.generate_samples_for_class(class_idx, n_real)
        
        # Calculate standard deviations for each feature across time
        real_std = np.std(real_samples, axis=(0, 1))  # Std across all samples and time
        fake_std = np.std(fake_samples, axis=(0, 1))
        
        # Calculate std across time for each sample, then average
        real_std_time = np.mean(np.std(real_samples, axis=1), axis=0)
        fake_std_time = np.mean(np.std(fake_samples, axis=1), axis=0)
        
        # Create DataFrame for comparison
        df = pd.DataFrame({
            'Feature': np.arange(self.n_features),
            'Real_Std_Overall': real_std,
            'Generated_Std_Overall': fake_std,
            'Std_Diff_Overall': np.abs(real_std - fake_std),
            'Real_Std_Temporal': real_std_time,
            'Generated_Std_Temporal': fake_std_time,
            'Std_Diff_Temporal': np.abs(real_std_time - fake_std_time)
        })
        
        return df
    
    def plot_standard_deviation_comparison(self, class_idx, figsize=(15, 10)):
        """
        Plot standard deviation comparison
        
        Parameters:
        -----------
        class_idx : int
            Class index to compare
        figsize : tuple
            Figure size
        """
        df = self.compute_standard_deviation_comparison(class_idx)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot overall std comparison
        ax = axes[0]
        ax.bar(df['Feature'] - 0.2, df['Real_Std_Overall'], width=0.4, alpha=0.7, label='Real')
        ax.bar(df['Feature'] + 0.2, df['Generated_Std_Overall'], width=0.4, alpha=0.7, label='Generated')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Standard Deviation')
        ax.set_title(f'Overall Standard Deviation Comparison - Class {class_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot temporal std comparison
        ax = axes[1]
        ax.bar(df['Feature'] - 0.2, df['Real_Std_Temporal'], width=0.4, alpha=0.7, label='Real')
        ax.bar(df['Feature'] + 0.2, df['Generated_Std_Temporal'], width=0.4, alpha=0.7, label='Generated')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Temporal Standard Deviation')
        ax.set_title(f'Temporal Standard Deviation Comparison - Class {class_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def compute_euclidean_distances(self, class_idx, n_samples=100):
        """
        Compute Euclidean distances between real and generated samples
        
        Parameters:
        -----------
        class_idx : int
            Class index to compare
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        tuple
            (intra_real_distances, intra_generated_distances, inter_distances)
        """
        # Get real samples
        real_samples = self.get_real_samples_for_class(class_idx)
        
        # Limit the number of real samples for computation efficiency
        if len(real_samples) > n_samples:
            indices = np.random.choice(len(real_samples), n_samples, replace=False)
            real_samples = real_samples[indices]
        
        # Generate fake samples
        fake_samples = self.generate_samples_for_class(class_idx, n_samples)
        
        # Reshape samples for distance calculation
        real_flat = real_samples.reshape(len(real_samples), -1)
        fake_flat = fake_samples.reshape(n_samples, -1)
        
        # Calculate distances
        intra_real_dist = euclidean_distances(real_flat)
        intra_fake_dist = euclidean_distances(fake_flat)
        inter_dist = euclidean_distances(real_flat, fake_flat)
        
        return intra_real_dist, intra_fake_dist, inter_dist
    
    def plot_euclidean_distance_distributions(self, class_idx, n_samples=100, figsize=(15, 5)):
        """
        Plot distributions of Euclidean distances
        
        Parameters:
        -----------
        class_idx : int
            Class index to compare
        n_samples : int
            Number of samples to generate
        figsize : tuple
            Figure size
        """
        intra_real, intra_fake, inter = self.compute_euclidean_distances(class_idx, n_samples)
        
        # Extract upper triangular part to avoid redundancy and self-distances
        intra_real_upper = intra_real[np.triu_indices_from(intra_real, k=1)]
        intra_fake_upper = intra_fake[np.triu_indices_from(intra_fake, k=1)]
        
        # Flatten inter distances
        inter_flat = inter.flatten()
        
        plt.figure(figsize=figsize)
        plt.hist(intra_real_upper, bins=30, alpha=0.7, label='Real-Real', density=True)
        plt.hist(intra_fake_upper, bins=30, alpha=0.7, label='Generated-Generated', density=True)
        plt.hist(inter_flat, bins=30, alpha=0.7, label='Real-Generated', density=True)
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Density')
        plt.title(f'Euclidean Distance Distributions - Class {class_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Return summary statistics
        stats_df = pd.DataFrame({
            'Distance_Type': ['Real-Real', 'Generated-Generated', 'Real-Generated'],
            'Mean': [np.mean(intra_real_upper), np.mean(intra_fake_upper), np.mean(inter_flat)],
            'Median': [np.median(intra_real_upper), np.median(intra_fake_upper), np.median(inter_flat)],
            'Std': [np.std(intra_real_upper), np.std(intra_fake_upper), np.std(inter_flat)],
            'Min': [np.min(intra_real_upper), np.min(intra_fake_upper), np.min(inter_flat)],
            'Max': [np.max(intra_real_upper), np.max(intra_fake_upper), np.max(inter_flat)]
        })
        
        return stats_df
    
    def compute_cdf_comparison(self, class_idx, feature_idx, n_samples=100):
        """
        Compare CDFs between real and generated samples for a specific feature
        
        Parameters:
        -----------
        class_idx : int
            Class index to compare
        feature_idx : int
            Feature index to compare
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        tuple
            (ks_statistic, p_value, real_cdf_x, real_cdf_y, fake_cdf_x, fake_cdf_y)
        """
        # Get real samples
        real_samples = self.get_real_samples_for_class(class_idx)
        
        # Generate fake samples
        fake_samples = self.generate_samples_for_class(class_idx, n_samples)
        
        # Extract feature values
        real_values = real_samples[:, :, feature_idx].flatten()
        fake_values = fake_samples[:, :, feature_idx].flatten()
        
        # Perform Kolmogorov-Smirnov test
        ks_result = ks_2samp(real_values, fake_values)
        
        # Compute empirical CDFs
        real_sorted = np.sort(real_values)
        real_cdf_y = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        
        fake_sorted = np.sort(fake_values)
        fake_cdf_y = np.arange(1, len(fake_sorted) + 1) / len(fake_sorted)
        
        return ks_result.statistic, ks_result.pvalue, real_sorted, real_cdf_y, fake_sorted, fake_cdf_y
    
    def plot_cdf_comparison(self, class_idx, feature_indices=None, n_samples=100, figsize=(15, 10)):
        """
        Plot CDF comparison between real and generated samples
        
        Parameters:
        -----------
        class_idx : int
            Class index to compare
        feature_indices : list
            List of feature indices to compare. If None, will select a few representative features.
        n_samples : int
            Number of samples to generate
        figsize : tuple
            Figure size
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with KS test results
        """
        if feature_indices is None:
            # Select a few representative features
            feature_indices = np.linspace(0, self.n_features - 1, 4, dtype=int)
        
        n_features = len(feature_indices)
        rows = int(np.ceil(n_features / 2))
        
        fig, axes = plt.subplots(rows, 2, figsize=figsize)
        if rows == 1 and n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # For storing KS test results
        ks_results = []
        
        for i, feature_idx in enumerate(feature_indices):
            if i < len(axes):
                ax = axes[i]
                
                # Compute CDF comparison
                ks_stat, p_value, real_x, real_cdf, fake_x, fake_cdf = self.compute_cdf_comparison(
                    class_idx, feature_idx, n_samples
                )
                
                # Plot CDFs
                ax.plot(real_x, real_cdf, label='Real', alpha=0.7)
                ax.plot(fake_x, fake_cdf, label='Generated', alpha=0.7)
                ax.set_title(f'Feature {feature_idx} (KS={ks_stat:.3f}, p={p_value:.3e})')
                ax.set_xlabel('Value')
                ax.set_ylabel('CDF')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Store results
                ks_results.append({
                    'Feature': feature_idx,
                    'KS_Statistic': ks_stat,
                    'p_value': p_value,
                    'Significant': p_value < 0.05
                })
        
        # Hide any unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame(ks_results)
    
    def generate_comprehensive_report(self, class_idx, n_samples=100, feature_sample=4):
        """
        Generate a comprehensive evaluation report for a specific class
        
        Parameters:
        -----------
        class_idx : int
            Class index to evaluate
        n_samples : int
            Number of samples to generate
        feature_sample : int
            Number of features to sample for detailed analysis
            
        Returns:
        --------
        dict
            Dictionary containing all evaluation results
        """
        # Select a sample of features for detailed analysis
        feature_indices = np.linspace(0, self.n_features - 1, feature_sample, dtype=int)
        
        # 1. Time series plots
        print(f"==== Time Series Comparison - Class {class_idx} ====")
        for feature_idx in feature_indices:
            self.plot_time_series_comparison(class_idx, feature_idx, n_samples=5)
        
        # 2. Standard deviation analysis
        print(f"\n==== Standard Deviation Analysis - Class {class_idx} ====")
        std_df = self.plot_standard_deviation_comparison(class_idx)
        print("Top 5 features with highest standard deviation difference:")
        print(std_df.sort_values('Std_Diff_Overall', ascending=False).head(5))
        
        # 3. Euclidean distance analysis
        print(f"\n==== Euclidean Distance Analysis - Class {class_idx} ====")
        dist_stats = self.plot_euclidean_distance_distributions(class_idx, n_samples)
        print(dist_stats)
        
        # 4. CDF comparison
        print(f"\n==== CDF Comparison - Class {class_idx} ====")
        ks_results = self.plot_cdf_comparison(class_idx, feature_indices, n_samples)
        print("KS Test Results:")
        print(ks_results)
        
        # 5. Summary statistics
        print(f"\n==== Summary Statistics - Class {class_idx} ====")
        overall_quality = 0.0
        
        # Compute average KS statistic (lower is better)
        avg_ks = ks_results['KS_Statistic'].mean()
        print(f"Average KS Statistic: {avg_ks:.4f} (lower is better)")
        
        # Count significant KS tests (fewer is better)
        sig_ks = ks_results['Significant'].sum()
        pct_sig = 100 * sig_ks / len(ks_results)
        print(f"Significant KS Tests: {sig_ks}/{len(ks_results)} ({pct_sig:.1f}%)")
        
        # Compute Euclidean distance ratio (closer to 1 is better)
        real_real_dist = dist_stats.iloc[0]['Mean']
        gen_gen_dist = dist_stats.iloc[1]['Mean']
        real_gen_dist = dist_stats.iloc[2]['Mean']
        
        dist_ratio = real_gen_dist / ((real_real_dist + gen_gen_dist) / 2)
        print(f"Distance Ratio: {dist_ratio:.4f} (closer to 1 is better)")
        
        # Compute standard deviation similarity (higher is better)
        std_similarity = 1 - std_df['Std_Diff_Overall'].mean() / std_df['Real_Std_Overall'].mean()
        print(f"Standard Deviation Similarity: {std_similarity:.4f} (higher is better)")
        
        # Compute overall quality score (0-100)
        # Convert metrics to 0-1 scale where 1 is best
        ks_score = 1 - min(avg_ks, 1.0)  # Lower KS is better
        sig_score = 1 - (sig_ks / len(ks_results))  # Fewer significant differences is better
        dist_score = 1 - min(abs(dist_ratio - 1), 1.0)  # Closer to 1 is better
        
        # Weighted average
        overall_quality = 100 * (0.3 * ks_score + 0.2 * sig_score + 0.3 * dist_score + 0.2 * std_similarity)
        print(f"Overall Quality Score: {overall_quality:.1f}/100")
        
        # Return all results
        return {
            'std_analysis': std_df,
            'euclidean_stats': dist_stats,
            'ks_results': ks_results,
            'overall_quality': overall_quality,
            'metrics': {
                'avg_ks': avg_ks,
                'sig_ks_pct': pct_sig,
                'dist_ratio': dist_ratio,
                'std_similarity': std_similarity
            }
        }
    
    def compare_all_classes(self, n_samples=100):
        """
        Compare all classes and rank them by generation quality
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate per class
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with quality metrics for all classes
        """
        class_metrics = []
        
        # Get unique classes
        unique_classes = np.unique(self.train_labels)
        
        for class_idx in unique_classes:
            # Skip classes with no samples
            class_mask = (self.train_labels == class_idx)
            if np.sum(class_mask) == 0:
                continue
                
            print(f"\nEvaluating Class {class_idx}...")
            try:
                # Calculate key metrics without full visualization
                # 1. Standard deviation analysis
                std_df = self.compute_standard_deviation_comparison(class_idx)
                std_similarity = 1 - std_df['Std_Diff_Overall'].mean() / std_df['Real_Std_Overall'].mean()
                
                # 2. Sample a few features for KS test
                feature_indices = np.linspace(0, self.n_features - 1, 4, dtype=int)
                ks_stats = []
                p_values = []
                
                for feature_idx in feature_indices:
                    ks_stat, p_value, _, _, _, _ = self.compute_cdf_comparison(
                        class_idx, feature_idx, n_samples
                    )
                    ks_stats.append(ks_stat)
                    p_values.append(p_value)
                
                avg_ks = np.mean(ks_stats)
                sig_ks = np.sum(np.array(p_values) < 0.05)
                pct_sig = 100 * sig_ks / len(p_values)
                
                # 3. Euclidean distance analysis
                _, _, inter_dist = self.compute_euclidean_distances(class_idx, min(n_samples, 50))
                avg_dist = np.mean(inter_dist)
                
                # 4. Calculate overall quality
                ks_score = 1 - min(avg_ks, 1.0)
                sig_score = 1 - (sig_ks / len(feature_indices))
                dist_score = 1 - min(avg_dist / 100, 1.0)  # Normalize distance
                
                overall_quality = 100 * (0.4 * ks_score + 0.3 * sig_score + 0.3 * std_similarity)
                
                # Store metrics
                class_metrics.append({
                    'Class': class_idx,
                    'Sample_Count': np.sum(class_mask),
                    'Avg_KS_Statistic': avg_ks,
                    'Pct_Significant_KS': pct_sig,
                    'Std_Similarity': std_similarity,
                    'Avg_Distance': avg_dist,
                    'Overall_Quality': overall_quality
                })
                
            except Exception as e:
                print(f"Error evaluating class {class_idx}: {str(e)}")
        
        # Create DataFrame and sort by quality
        metrics_df = pd.DataFrame(class_metrics)
        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values('Overall_Quality', ascending=False)
        
        return metrics_df