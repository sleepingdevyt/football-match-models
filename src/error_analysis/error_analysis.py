import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

class SoccerModelErrorAnalyzer:
    """
    A comprehensive Error Analysis pipeline for 1X2 Soccer Models.
    Designed to diagnose Reliability, Market Performance, Segmentation, and Consensus.
    """

    def __init__(self, val_df, y_true, model_preds_dict, outcomes_order=['Away', 'Draw', 'Home']):
        """
        :param val_df: DataFrame containing validation features and metadata (Tournament, Odds, etc.)
        :param y_true: Series or Array of actual outcomes (encoded as 0, 1, 2 corresponding to outcomes_order)
        :param model_preds_dict: Dictionary {'ModelName': probabilities_array (N_samples, 3)}
        :param outcomes_order: Order of columns in probabilities_array. Default: Away (0), Draw (1), Home (2)
        """
        self.val_df = val_df.copy()
        self.y_true = np.array(y_true)
        self.models = model_preds_dict
        self.outcomes = outcomes_order
        self.results = {}

        # Ensure y_true is integer encoded if it's not already
        if self.y_true.dtype == 'O':
            # rudimentary mapping, adjust if your encoding differs
            mapping = {k: i for i, k in enumerate(outcomes_order)}
            self.y_true = np.vectorize(mapping.get)(self.y_true)

    def _calculate_rps_single(self, probs, outcome_idx):
        """
        Calculates RPS for a single match.
        RPS = (1 / (r-1)) * sum((CDF_pred - CDF_obs)^2)
        For 3 classes (r=3), 1/(3-1) = 0.5
        """
        # CDF Prediction
        cdf_pred = np.cumsum(probs)

        # CDF Observed (Heaviside step function)
        cdf_obs = np.zeros(3)
        cdf_obs[outcome_idx:] = 1

        # We sum over r-1 (first two outcomes)
        rps = 0.5 * np.sum((cdf_pred[:2] - cdf_obs[:2]) ** 2)
        return rps

    def _get_vectorized_rps(self, probs_array, y_true):
        """Vectorized RPS calculation for an array of predictions."""
        n_samples = len(y_true)
        rps_values = []
        for i in range(n_samples):
            rps_values.append(self._calculate_rps_single(probs_array[i], y_true[i]))
        return np.array(rps_values)

    # ==============================================================================
    # TEST 1: The "Trust" Test (Reliability & Calibration)
    # ==============================================================================
    def test_reliability_calibration(self):
        print("\n" + "="*60)
        print("TEST 1: Reliability & Calibration Analysis")
        print("GOAL: Determine if '70% confidence' actually means 70% win rate.")
        print("="*60)

        plt.figure(figsize=(10, 6))

        # We focus on the 'Home' probability (index 2) for visualization
        home_idx = 2

        for model_name, preds in self.models.items():
            prob_home = preds[:, home_idx]
            y_home = (self.y_true == home_idx).astype(int)

            fraction_of_positives, mean_predicted_value = calibration_curve(y_home, prob_home, n_bins=10)

            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{model_name}')

            # Calculate Brier Score (MSE of probabilities) as a summary metric
            bs = brier_score_loss(y_home, prob_home)
            print(f"[{model_name}] Home Win Brier Score: {bs:.4f}")

        plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        plt.ylabel("Fraction of positives (Actual Home Win Rate)")
        plt.xlabel("Mean predicted value (Predicted Home Win Prob)")
        plt.title("Reliability Diagram (Home Wins)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def test_tournament_segmentation(self, group_col='tournament'):
        print("\n" + "="*60)
        print("TEST 3: Tournament Segmented Performance")
        print("GOAL: Identify which leagues/cups we fail to predict.")
        print("="*60)

        if group_col not in self.val_df.columns:
            print(f"Column '{group_col}' not found in DataFrame.")
            return

        # Use the best model (lowest RPS) or just the first one for this deep dive
        # Let's use the first model in the dict
        model_name = list(self.models.keys())[0]
        preds = self.models[model_name]

        rps_scores = self._get_vectorized_rps(preds, self.y_true)

        # Create a temp DF for aggregation
        analysis_df = self.val_df[[group_col]].copy()
        analysis_df['RPS'] = rps_scores
        analysis_df['Count'] = 1

        grouped = analysis_df.groupby(group_col).agg({'RPS': 'mean', 'Count': 'count'}).sort_values('RPS', ascending=False)

        print(f"Analysis using model: {model_name}")
        print(grouped.head(10))


    # ==============================================================================
    # TEST 4: The "Data Blindspots" Test (Feature Slicing)
    # ==============================================================================
    def test_feature_slicing(self):
        print("\n" + "="*60)
        print("TEST 4: Data Blindspots (Feature Slicing)")
        print("GOAL: Find non-linear weaknesses in specific scenarios.")
        print("="*60)

        # Features to analyze
        features_to_check = [
            'diff_days_rest',
            'home_elo_diff_home_away',
            'home_tactics_instability_index_last_10'
        ]

        # Get predictions from the first model
        model_name = list(self.models.keys())[0]
        preds = self.models[model_name]

        # Calculate RPS (returns a numpy array)
        rps_scores = self._get_vectorized_rps(preds, self.y_true)

        for feat in features_to_check:
            if feat not in self.val_df.columns:
                print(f"Skipping {feat}: Not found in DataFrame columns.")
                continue

            try:
                # 1. Create a temporary DataFrame to handle alignment and types safely
                # We use .values to ignore index mismatches between val_df and the rps array
                plot_df = pd.DataFrame({
                    'RPS': rps_scores,
                    'Feature_Value': self.val_df[feat].values
                })

                # 2. Create Bins (Deciles)
                # duplicates='drop' handles cases where many values are identical (e.g. 0 days rest)
                plot_df['bin'] = pd.qcut(plot_df['Feature_Value'], q=10, duplicates='drop')

                # 3. Group by the bin and calculate mean RPS
                mean_rps_by_bin = plot_df.groupby('bin')['RPS'].mean()

                # 4. Plot
                plt.figure(figsize=(10, 5))
                mean_rps_by_bin.plot(kind='bar', color='salmon', edgecolor='black', alpha=0.7)

                plt.title(f"Mean RPS by {feat} (Deciles)")
                plt.ylabel("Error (RPS)")
                plt.xlabel(f"{feat} Range")
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()

                print(f"-> Plotted error distribution for {feat}")

            except Exception as e:
                print(f"Could not analyze feature {feat}: {e}")


    # ==============================================================================
    # TEST 6: The "Consensus" Test (Ensemble Correlation)
    # ==============================================================================
    def test_ensemble_correlation(self):
        print("\n" + "="*60)
        print("TEST 6: Consensus & Ensemble Potential")
        print("GOAL: Check if models agree. Low correlation = High Ensemble Potential.")
        print("="*60)

        # Extract Home Win probabilities from all models
        home_probs = {}
        for name, preds in self.models.items():
            home_probs[name] = preds[:, 2] # Index 2 is Home

        corr_df = pd.DataFrame(home_probs)
        corr_matrix = corr_df.corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0.9, vmax=1.0)
        plt.title("Model Prediction Correlation (Home Win Prob)")
        plt.show()

        avg_corr = np.mean(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)])
        print(f"Average Correlation: {avg_corr:.4f}")

    def _calculate_ece(self, y_true_binary, y_prob, n_bins=10):
        """
        Calcula el Expected Calibration Error (ECE) para un problema binario.

        Parameters
        ----------
        y_true_binary : array-like of shape (n_samples,)
            Etiquetas binarias (0/1).
        y_prob : array-like of shape (n_samples,)
            Probabilidades predichas para la clase positiva.
        n_bins : int
            Número de bins de calibración.

        Returns
        -------
        float
            Valor del ECE.
        """
        y_true_binary = np.asarray(y_true_binary)
        y_prob = np.asarray(y_prob)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

        ece = 0.0
        n = len(y_prob)

        for b in range(n_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue

            bin_confidence = y_prob[mask].mean()
            bin_accuracy = y_true_binary[mask].mean()
            bin_weight = mask.sum() / n

            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def calculate_ece_by_category(self, category_col='tournament', n_bins=10, min_samples=30):
        """
        Calcula el ECE de cada modelo para cada categoría.

        Parameters
        ----------
        category_col : str
            Columna de self.val_df usada para segmentar.
        n_bins : int
            Número de bins para el cálculo del ECE.
        min_samples : int
            Mínimo de muestras requeridas por categoría.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas:
            [category_col, Model, Outcome, Samples, ECE]
        """
        if category_col not in self.val_df.columns:
            raise ValueError(f"Column '{category_col}' not found in val_df.")

        rows = []

        categories = self.val_df[category_col].dropna().unique()

        for category in categories:
            mask = self.val_df[category_col] == category
            n_samples = mask.sum()

            if n_samples < min_samples:
                continue

            y_cat = self.y_true[mask]

            for model_name, preds in self.models.items():
                preds_cat = preds[mask]

                # ECE por cada outcome (one-vs-rest)
                for outcome_idx, outcome_name in enumerate(self.outcomes):
                    y_binary = (y_cat == outcome_idx).astype(int)
                    y_prob = preds_cat[:, outcome_idx]

                    ece = self._calculate_ece(
                        y_true_binary=y_binary,
                        y_prob=y_prob,
                        n_bins=n_bins
                    )

                    rows.append({
                        category_col: category,
                        'Model': model_name,
                        'Outcome': outcome_name,
                        'Samples': n_samples,
                        'ECE': ece
                    })

                # ECE promedio del modelo en esa categoría
                ece_values = []
                for outcome_idx in range(len(self.outcomes)):
                    y_binary = (y_cat == outcome_idx).astype(int)
                    y_prob = preds_cat[:, outcome_idx]

                    ece_values.append(
                        self._calculate_ece(
                            y_true_binary=y_binary,
                            y_prob=y_prob,
                            n_bins=n_bins
                        )
                    )

                rows.append({
                    category_col: category,
                    'Model': model_name,
                    'Outcome': 'MacroAvg',
                    'Samples': n_samples,
                    'ECE': np.mean(ece_values)
                })

        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values(
            ['Model', category_col, 'Outcome']
        ).reset_index(drop=True)

        return result_df

    def plot_ece_by_category(self,
                             category_col='tournament',
                             outcome='MacroAvg',
                             n_bins=10,
                             min_samples=30,
                             top_n=20):
        """
        Grafica el ECE por categoría para cada modelo.
        """
        ece_df = self.calculate_ece_by_category(
            category_col=category_col,
            n_bins=n_bins,
            min_samples=min_samples
        )

        plot_df = ece_df[ece_df['Outcome'] == outcome].copy()

        # Seleccionar categorías con mayor ECE promedio
        top_categories = (
            plot_df.groupby(category_col)['ECE']
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        plot_df = plot_df[plot_df[category_col].isin(top_categories)]

        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=plot_df,
            x=category_col,
            y='ECE',
            hue='Model'
        )
        plt.xticks(rotation=45, ha='right')
        plt.title(f'ECE by {category_col} ({outcome})')
        plt.ylabel('Expected Calibration Error')
        plt.xlabel(category_col)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ==============================================================================
    # PIPELINE ORCHESTRATOR
    # ==============================================================================
    def run_pipeline(self):
        """Executes all selected tests in order."""
        print("STARTING ERROR ANALYSIS PIPELINE...")
        print(f"Validation Set Size: {len(self.val_df)} matches")

        # 1. Correlación entre modelos
        self.test_ensemble_correlation()

        # 2. Calibración global
        self.test_reliability_calibration()

        # 3. Segmentación por torneo
        self.test_tournament_segmentation()

        # 4. Feature slicing
        self.test_feature_slicing()

        # 5. ECE por categoría
        print("\n" + "=" * 60)
        print("TEST 5: Expected Calibration Error by Category")
        print("GOAL: Measure calibration quality for each model within each category.")
        print("=" * 60)

        try:
            ece_df = self.calculate_ece_by_category(
                category_col='tournament',
                n_bins=10,
                min_samples=30
            )

            if ece_df.empty:
                print("No categories with enough samples to compute ECE.")
            else:
                print("\nTop 20 worst calibrated segments (MacroAvg):")
                display(
                    ece_df[ece_df['Outcome'] == 'MacroAvg']
                    .sort_values('ECE', ascending=False)
                    .head(20)
                )

                # Visualización
                self.plot_ece_by_category(
                    category_col='tournament',
                    outcome='MacroAvg',
                    n_bins=10,
                    min_samples=30,
                    top_n=15
                )

                # Guardar resultados
                self.results['ece_by_category'] = ece_df

        except Exception as e:
            print(f"Could not compute ECE by category: {e}")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)