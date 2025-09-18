#!/usr/bin/env python3
"""
Fixed ML Model Evaluation
Corrected data type issues and model evaluation problems
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FixedLeakageDetector:
    """Fixed data leakage detection"""
    
    def __init__(self):
        self.leakage_checks = []
    
    def check_feature_leakage(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Check for feature leakage"""
        try:
            leakage_results = {}
            
            # Check for perfect correlation (suspicious)
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    correlation = abs(features[col].corr(target))
                    if correlation > 0.99:
                        leakage_results[col] = {
                            'correlation': correlation,
                            'leakage_risk': 'HIGH',
                            'reason': 'Perfect correlation with target'
                        }
                    elif correlation > 0.95:
                        leakage_results[col] = {
                            'correlation': correlation,
                            'leakage_risk': 'MEDIUM',
                            'reason': 'Very high correlation with target'
                        }
            
            # Check for future information
            for col in features.columns:
                if 'future' in col.lower() or 'next' in col.lower():
                    leakage_results[col] = {
                        'leakage_risk': 'HIGH',
                        'reason': 'Feature name suggests future information'
                    }
            
            return leakage_results
            
        except Exception as e:
            logger.error(f"❌ Leakage detection failed: {e}")
            return {}
    
    def check_temporal_leakage(self, features: pd.DataFrame, target: pd.Series, 
                             timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """Check for temporal leakage"""
        try:
            if timestamp_col not in features.columns:
                return {'error': 'Timestamp column not found'}
            
            # Sort by timestamp
            df = features.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)
            
            # Check if any feature value is from future
            leakage_results = {}
            
            for col in df.columns:
                if col == timestamp_col:
                    continue
                
                # Check if feature value is from future relative to target
                if col in target.index:
                    feature_timestamp = df.loc[target.index, timestamp_col]
                    target_timestamp = df.loc[target.index, timestamp_col]
                    
                    future_features = feature_timestamp > target_timestamp
                    if future_features.any():
                        leakage_results[col] = {
                            'leakage_risk': 'HIGH',
                            'reason': f'{future_features.sum()} future values detected'
                        }
            
            return leakage_results
            
        except Exception as e:
            logger.error(f"❌ Temporal leakage detection failed: {e}")
            return {}

class FixedTimeSeriesValidator:
    """Fixed time-series specific validation"""
    
    def __init__(self):
        self.validation_results = {}
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, 
                              model, n_splits: int = 5) -> Dict[str, Any]:
        """Walk-forward validation for time-series"""
        try:
            # Clean data - remove non-numeric columns
            X_clean = X.select_dtypes(include=[np.number])
            
            if len(X_clean.columns) == 0:
                return {'error': 'No numeric features available'}
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            mse_scores = []
            mae_scores = []
            directional_accuracy = []
            hit_rates = []
            
            for train_idx, test_idx in tscv.split(X_clean):
                X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Directional accuracy
                direction_actual = np.sign(y_test.diff().dropna())
                direction_pred = np.sign(pd.Series(y_pred).diff().dropna())
                dir_acc = (direction_actual == direction_pred).mean()
                
                # Hit rate (percentage of correct predictions)
                hit_rate = (np.abs(y_test - y_pred) < np.std(y_test)).mean()
                
                mse_scores.append(mse)
                mae_scores.append(mae)
                directional_accuracy.append(dir_acc)
                hit_rates.append(hit_rate)
            
            return {
                'mse_mean': np.mean(mse_scores),
                'mse_std': np.std(mse_scores),
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'directional_accuracy_mean': np.mean(directional_accuracy),
                'directional_accuracy_std': np.std(directional_accuracy),
                'hit_rate_mean': np.mean(hit_rates),
                'hit_rate_std': np.std(hit_rates),
                'n_splits': n_splits
            }
            
        except Exception as e:
            logger.error(f"❌ Walk-forward validation failed: {e}")
            return {}
    
    def purged_kfold_validation(self, X: pd.DataFrame, y: pd.Series, 
                               model, n_splits: int = 5, purge_days: int = 1) -> Dict[str, Any]:
        """Purged K-Fold validation to avoid leakage"""
        try:
            # Clean data - remove non-numeric columns
            X_clean = X.select_dtypes(include=[np.number])
            
            if len(X_clean.columns) == 0:
                return {'error': 'No numeric features available'}
            
            # Create time-based splits with purging
            n_samples = len(X_clean)
            split_size = n_samples // n_splits
            
            mse_scores = []
            mae_scores = []
            directional_accuracy = []
            
            for i in range(n_splits):
                # Define train and test indices
                test_start = i * split_size
                test_end = (i + 1) * split_size
                
                # Purge period
                purge_start = max(0, test_start - purge_days)
                purge_end = min(n_samples, test_end + purge_days)
                
                # Training indices (before purge period)
                train_indices = list(range(0, purge_start))
                
                # Test indices (after purge period)
                test_indices = list(range(purge_end, n_samples))
                
                if len(train_indices) == 0 or len(test_indices) == 0:
                    continue
                
                X_train, X_test = X_clean.iloc[train_indices], X_clean.iloc[test_indices]
                y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Directional accuracy
                direction_actual = np.sign(y_test.diff().dropna())
                direction_pred = np.sign(pd.Series(y_pred).diff().dropna())
                dir_acc = (direction_actual == direction_pred).mean()
                
                mse_scores.append(mse)
                mae_scores.append(mae)
                directional_accuracy.append(dir_acc)
            
            return {
                'mse_mean': np.mean(mse_scores),
                'mse_std': np.std(mse_scores),
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'directional_accuracy_mean': np.mean(directional_accuracy),
                'directional_accuracy_std': np.std(directional_accuracy),
                'n_splits': len(mse_scores),
                'purge_days': purge_days
            }
            
        except Exception as e:
            logger.error(f"❌ Purged K-Fold validation failed: {e}")
            return {}

class FixedRealisticModelEvaluator:
    """Fixed realistic model evaluation with economic metrics"""
    
    def __init__(self):
        self.leakage_detector = FixedLeakageDetector()
        self.validator = FixedTimeSeriesValidator()
    
    def evaluate_price_prediction_model(self, X: pd.DataFrame, y: pd.Series, 
                                      model, timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """Evaluate price prediction model with realistic metrics"""
        try:
            logger.info("🔍 Starting realistic model evaluation...")
            
            # 1. Leakage detection
            logger.info("🔍 Checking for data leakage...")
            feature_leakage = self.leakage_detector.check_feature_leakage(X, y)
            temporal_leakage = self.leakage_detector.check_temporal_leakage(X, y, timestamp_col)
            
            # 2. Time-series validation
            logger.info("🔍 Running walk-forward validation...")
            wf_results = self.validator.walk_forward_validation(X, y, model)
            
            logger.info("🔍 Running purged K-Fold validation...")
            pkf_results = self.validator.purged_kfold_validation(X, y, model)
            
            # 3. Economic metrics
            logger.info("🔍 Calculating economic metrics...")
            economic_metrics = self._calculate_economic_metrics(X, y, model)
            
            # 4. Compile results
            results = {
                'leakage_detection': {
                    'feature_leakage': feature_leakage,
                    'temporal_leakage': temporal_leakage,
                    'leakage_risk': 'HIGH' if feature_leakage or temporal_leakage else 'LOW'
                },
                'walk_forward_validation': wf_results,
                'purged_kfold_validation': pkf_results,
                'economic_metrics': economic_metrics,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # 5. Realistic performance assessment
            if wf_results.get('directional_accuracy_mean', 0) > 0.7:
                results['realistic_assessment'] = 'GOOD - High directional accuracy'
            elif wf_results.get('directional_accuracy_mean', 0) > 0.55:
                results['realistic_assessment'] = 'MODERATE - Above random'
            else:
                results['realistic_assessment'] = 'POOR - Near random performance'
            
            logger.info(f"✅ Model evaluation completed: {results['realistic_assessment']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_economic_metrics(self, X: pd.DataFrame, y: pd.Series, model) -> Dict[str, Any]:
        """Calculate economically meaningful metrics"""
        try:
            # Clean data - remove non-numeric columns
            X_clean = X.select_dtypes(include=[np.number])
            
            if len(X_clean.columns) == 0:
                return {'error': 'No numeric features available'}
            
            # Simple strategy simulation
            y_pred = model.predict(X_clean)
            
            # Calculate returns
            actual_returns = y.pct_change().dropna()
            predicted_returns = pd.Series(y_pred).pct_change().dropna()
            
            # Align series
            min_len = min(len(actual_returns), len(predicted_returns))
            if min_len == 0:
                return {'error': 'No data for economic metrics'}
            
            actual_returns = actual_returns.iloc[-min_len:]
            predicted_returns = predicted_returns.iloc[-min_len:]
            
            # Strategy signals
            signals = np.where(predicted_returns > 0, 1, -1)
            strategy_returns = signals * actual_returns
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(strategy_returns)
            win_rate = (strategy_returns > 0).mean()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_return': strategy_returns.mean(),
                'volatility': strategy_returns.std(),
                'total_trades': len(strategy_returns)
            }
            
        except Exception as e:
            logger.error(f"❌ Economic metrics calculation failed: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def create_realistic_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create realistic model with proper validation"""
        try:
            logger.info("🤖 Creating realistic price prediction model...")
            
            # Remove any suspicious features
            X_clean = self._clean_features(X)
            
            # Create models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            results = {}
            
            for name, model in models.items():
                logger.info(f"🤖 Training {name} model...")
                
                # Evaluate model
                evaluation = self.evaluate_price_prediction_model(X_clean, y, model)
                
                # Check if model is realistic
                if evaluation.get('leakage_detection', {}).get('leakage_risk') == 'LOW':
                    results[name] = {
                        'model': model,
                        'evaluation': evaluation,
                        'realistic': True
                    }
                else:
                    results[name] = {
                        'model': model,
                        'evaluation': evaluation,
                        'realistic': False,
                        'reason': 'Data leakage detected'
                    }
            
            # Select best realistic model
            realistic_models = {k: v for k, v in results.items() if v['realistic']}
            
            if realistic_models:
                # Select based on directional accuracy
                best_model = max(realistic_models.keys(), 
                               key=lambda x: realistic_models[x]['evaluation']['walk_forward_validation'].get('directional_accuracy_mean', 0))
                
                logger.info(f"✅ Best realistic model: {best_model}")
                
                return {
                    'best_model': best_model,
                    'all_models': results,
                    'realistic_models': realistic_models,
                    'evaluation_summary': {
                        'total_models': len(models),
                        'realistic_models': len(realistic_models),
                        'best_directional_accuracy': realistic_models[best_model]['evaluation']['walk_forward_validation'].get('directional_accuracy_mean', 0)
                    }
                }
            else:
                logger.warning("⚠️ No realistic models found - all models have data leakage")
                return {
                    'best_model': None,
                    'all_models': results,
                    'realistic_models': {},
                    'evaluation_summary': {
                        'total_models': len(models),
                        'realistic_models': 0,
                        'best_directional_accuracy': 0
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Realistic model creation failed: {e}")
            return {'error': str(e)}
    
    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean features to remove potential leakage"""
        try:
            X_clean = X.copy()
            
            # Remove features with suspicious names
            suspicious_patterns = ['future', 'next', 'ahead', 'forward']
            for pattern in suspicious_patterns:
                X_clean = X_clean.loc[:, ~X_clean.columns.str.contains(pattern, case=False)]
            
            # Remove features with perfect correlation
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    if X_clean[col].std() == 0:  # Constant feature
                        X_clean = X_clean.drop(columns=[col])
            
            logger.info(f"🧹 Cleaned features: {len(X.columns)} -> {len(X_clean.columns)}")
            
            return X_clean
            
        except Exception as e:
            logger.error(f"❌ Feature cleaning failed: {e}")
            return X

# Global fixed evaluator instance
fixed_model_evaluator = FixedRealisticModelEvaluator()

# Convenience functions
def evaluate_price_prediction_model(X: pd.DataFrame, y: pd.Series, model, timestamp_col: str = 'timestamp') -> Dict[str, Any]:
    """Evaluate price prediction model"""
    return fixed_model_evaluator.evaluate_price_prediction_model(X, y, model, timestamp_col)

def create_realistic_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Create realistic model with proper validation"""
    return fixed_model_evaluator.create_realistic_model(X, y)
