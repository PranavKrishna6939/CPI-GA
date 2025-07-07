import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

MAX_FEATURES_TO_SELECT = 100
MIN_FEATURES_TO_SELECT = 50
FEATURE_SELECTION_RATIO = 1/3

DEFAULT_LAG_PERIODS = [1, 12]

class TimeSeriesCPIPredictor:
    def __init__(self, data_path='raw_cpi_data.csv',
                 max_features=None, lag_periods=None):
        self.data_path = data_path
        self.target_column = 'Consumer_Food_Price_Index_Combined'
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.max_features = max_features if max_features is not None else MAX_FEATURES_TO_SELECT
        self.min_features = MIN_FEATURES_TO_SELECT
        self.feature_ratio = FEATURE_SELECTION_RATIO
        self.lag_periods = lag_periods if lag_periods is not None else DEFAULT_LAG_PERIODS
    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Target column: {self.target_column}")
        month_mapping = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        self.df['month_num'] = self.df['Month'].map(month_mapping)
        unmapped = self.df[self.df['month_num'].isna()]

        self.df['date'] = pd.to_datetime(
            self.df['Year'].astype(str) + '-' + self.df['month_num'].astype(int).astype(str) + '-01'
        )
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.numerical_cols = [col for col in self.df.columns
                              if col not in ['Year', 'Month', 'date', 'month_num']
                              and self.df[col].dtype in ['float64', 'int64']]

        self.feature_cols = [col for col in self.numerical_cols if col != self.target_column]

        self.impute_raw_data()
    def impute_raw_data(self):
        nan_counts = self.df[self.numerical_cols].isnull().sum()
        total_nans = nan_counts.sum()
        print(f"Total NaN values in raw data: {total_nans}")
        if total_nans == 0:
            print("No NaN values found in raw data.")
            return
        nan_columns = [col for col, count in nan_counts.items() if count > 0]
        total_imputed = 0
        for col in nan_columns:
            col_nans = self.df[col].isnull().sum()
            self.df[col] = self.df[col].fillna(method='ffill').fillna(method='bfill')
            if self.df[col].isnull().any():
                col_mean = self.df[col].mean()
                if pd.isna(col_mean):
                    self.df[col] = self.df[col].fillna(0)
                else:
                    self.df[col] = self.df[col].fillna(col_mean)
            total_imputed += col_nans
        print(f"Total values imputed: {total_imputed}")
        final_nans = self.df[self.numerical_cols].isnull().sum().sum()

    def create_lag_features(self, lag_periods=None):
        if lag_periods is None:
            lag_periods = self.lag_periods
        print(f"Creating lag features for periods: {lag_periods}")
        for lag in lag_periods:
            for col in self.numerical_cols:
                lag_col_name = f"{col}_lag_{lag}"
                self.df[lag_col_name] = self.df[col].shift(lag)
        self.lag_feature_cols = []
        for lag in lag_periods:
            for col in self.feature_cols:
                lag_col_name = f"{col}_lag_{lag}"
                self.lag_feature_cols.append(lag_col_name)
        max_lag = max(lag_periods)
        self.df = self.df.iloc[max_lag:].reset_index(drop=True)

    def prepare_modeling_data(self, test_size=0.1):

        self.create_lag_features()
        mask = ~self.df[self.target_column].isna()
        modeling_data = self.df[mask].copy()
        X = modeling_data[self.lag_feature_cols]
        y = modeling_data[self.target_column]
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        n_features_to_select = min(self.max_features,
                                 max(self.min_features,
                                     int(X.shape[1] * self.feature_ratio)))
        top_features = correlations.head(n_features_to_select).index.tolist()
        X = X[top_features]
        self.selected_features = top_features

        split_idx = int(len(X) * (1 - test_size))
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
    def train_models(self):

        model_configs = {
            'RandomForest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features=0.3,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=20,
                    learning_rate=0.05,
                    max_depth=2,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features=0.3,
                    subsample=0.7,
                    random_state=42
                ),
                'use_scaled': False
            },
            'ExtraTrees': {
                'model': ExtraTreesRegressor(
                    n_estimators=20,
                    max_depth=3,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features=0.3,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(
                    n_estimators=20,
                    learning_rate=0.05,
                    max_depth=2,
                    min_child_weight=10,
                    subsample=0.6,
                    colsample_bytree=0.3,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(
                    n_estimators=20,
                    learning_rate=0.05,
                    max_depth=2,
                    min_child_samples=25,
                    subsample=0.6,
                    colsample_bytree=0.3,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'use_scaled': False
            },
            'Ridge': {
                'model': Ridge(
                    alpha=10.0,
                    random_state=42
                ),
                'use_scaled': True
            },
            'Lasso': {
                'model': Lasso(
                    alpha=1.0,
                    random_state=42,
                    max_iter=2000
                ),
                'use_scaled': True
            }
        }
        self.results = {}
        for name, config in model_configs.items():
            print(f"Training {name}...")
            model = config['model']
            if config['use_scaled']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            if X_train_use.isnull().any().any():
                X_train_use = X_train_use.fillna(0)
            if X_test_use.isnull().any().any():
                X_test_use = X_test_use.fillna(0)
            try:
                model.fit(X_train_use, self.y_train)
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
            y_train_pred = model.predict(X_train_use)
            y_test_pred = model.predict(X_test_use)
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            self.results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            }
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X_train_use.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            print(f"  {name} - Test R²: {test_r2:.4f}, Test MAE: {test_mae:.4f}")
    def evaluate_models(self):
        print("\n" + "="*60)
        print("TIME SERIES MODEL PERFORMANCE")
        print("="*60)
        summary_data = []
        for name, results in self.results.items():
            summary_data.append({
                'Model': name,
                'Train R²': results['train_r2'],
                'Test R²': results['test_r2'],
                'Train MAE': results['train_mae'],
                'Test MAE': results['test_mae'],
                'Train RMSE': np.sqrt(results['train_mse']),
                'Test RMSE': np.sqrt(results['test_mse'])
            })
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.round(4))
        best_model_name = summary_df.loc[summary_df['Test R²'].idxmax(), 'Model']
        return summary_df, best_model_name
    
    def plot_results(self, best_model_name):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        ax1 = axes[0, 0]
        models = list(self.results.keys())
        test_r2_scores = [self.results[name]['test_r2'] for name in models]
        bars = ax1.bar(models, test_r2_scores, color='lightblue', alpha=0.7)
        ax1.set_title('Time Series Model Performance (Test R²)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        best_idx = models.index(best_model_name)
        bars[best_idx].set_color('green')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        ax2 = axes[0, 1]
        best_results = self.results[best_model_name]
        ax2.scatter(self.y_test, best_results['y_test_pred'], alpha=0.6, color='green', label='Predictions')
        ax2.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title(f'Time Series: Actual vs Predicted - {best_model_name}', fontsize=12, fontweight='bold')
        ax2.legend()
        ax3 = axes[1, 0]
        if best_model_name in self.feature_importance:
            top_features = self.feature_importance[best_model_name].head(15)
            ax3.barh(range(len(top_features)), top_features['importance'], color='lightgreen')
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'], fontsize=8)
            ax3.set_xlabel('Feature Importance')
            ax3.set_title(f'Time Series Top 15 Lag Features - {best_model_name}', fontsize=12, fontweight='bold')
            ax3.invert_yaxis()
        ax4 = axes[1, 1]
        test_dates = self.df.iloc[self.X_test.index]['date']
        ax4.plot(test_dates, self.y_test.values, label='Actual', linewidth=2, color='blue')
        ax4.plot(test_dates, best_results['y_test_pred'], label='Time Series Predicted',
                linewidth=2, color='green', linestyle='--')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Consumer Food Price Index')
        ax4.set_title(f'Time Series Prediction Timeline - {best_model_name}', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig('timeseries_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    def save_feature_importance(self, best_model_name):
        if best_model_name in self.feature_importance:
            self.feature_importance[best_model_name].to_csv(
                f'timeseries_feature_importance_{best_model_name}.csv', index=False
            )
            print(f"Time series feature importance saved to timeseries_feature_importance_{best_model_name}.csv")
    
    def run_complete_pipeline(self):

        self.load_and_prepare_data()
        self.prepare_modeling_data()
        self.train_models()
        summary_df, best_model_name = self.evaluate_models()
        self.plot_results(best_model_name)
        self.save_feature_importance(best_model_name)
        return self.results, summary_df, best_model_name
    
if __name__ == "__main__":
    print("="*60)
    print(f"Default max features: {MAX_FEATURES_TO_SELECT}")
    print(f"Default lag periods: {DEFAULT_LAG_PERIODS}")
    ts_predictor = TimeSeriesCPIPredictor()
    ts_results, ts_summary, ts_best = ts_predictor.run_complete_pipeline()
    print(f"- Lag features used: {len(ts_predictor.lag_feature_cols)}")
    print(f"- Selected features: {len(ts_predictor.selected_features)}")
    print(f"- Best time series model: {ts_best}")
    print(f"- Best time series R²: {ts_summary['Test R²'].max():.4f}")

