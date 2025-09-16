#!/usr/bin/env python3
"""
ML Model Integration System
Advanced machine learning models for trading predictions and market analysis
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLModelIntegration:
    """Advanced ML model integration for trading"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineering = FeatureEngineering()
        self.model_performance = {}
        
    def create_price_prediction_model(self) -> Dict[str, Any]:
        """Create price prediction model using ensemble methods"""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Generate synthetic training data (replace with real data)
            np.random.seed(42)
            n_samples = 10000
            
            # Create features
            features = np.random.randn(n_samples, 20)
            
            # Create target (price movement)
            target = np.sum(features[:, :5], axis=1) + np.random.randn(n_samples) * 0.1
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Create ensemble model
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            # Train models
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.model_performance[name] = {
                    'mse': mse,
                    'r2_score': r2,
                    'accuracy': max(0, r2) * 100
                }
                
                logger.info(f"✅ {name} model trained - R²: {r2:.3f}, Accuracy: {max(0, r2)*100:.1f}%")
            
            # Store best model
            best_model_name = max(self.model_performance.keys(), 
                                key=lambda x: self.model_performance[x]['r2_score'])
            self.models['price_prediction'] = models[best_model_name]
            
            return {
                'model_type': 'price_prediction',
                'best_model': best_model_name,
                'performance': self.model_performance[best_model_name],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Price prediction model creation failed: {e}")
            return {}
    
    def create_sentiment_analysis_model(self) -> Dict[str, Any]:
        """Create sentiment analysis model for market sentiment"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            
            # Generate synthetic sentiment data (replace with real news/social media data)
            np.random.seed(42)
            
            # Create sample text data
            positive_texts = [
                "Market is bullish and trending upward",
                "Strong earnings report drives stock higher",
                "Positive economic indicators show growth",
                "Investor confidence is high",
                "Bullish momentum continues"
            ] * 100
            
            negative_texts = [
                "Market crash concerns investors",
                "Bearish trend continues downward",
                "Economic uncertainty affects trading",
                "Negative sentiment prevails",
                "Market volatility increases"
            ] * 100
            
            neutral_texts = [
                "Market remains stable",
                "Trading volume is normal",
                "No significant market movements",
                "Sideways trading pattern",
                "Market consolidation continues"
            ] * 100
            
            # Combine data
            texts = positive_texts + negative_texts + neutral_texts
            labels = [1] * 500 + [0] * 500 + [2] * 500  # 1: positive, 0: negative, 2: neutral
            
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42
            )
            
            # Train model
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models['sentiment_analysis'] = model
            self.models['sentiment_vectorizer'] = vectorizer
            
            self.model_performance['sentiment_analysis'] = {
                'accuracy': accuracy * 100,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info(f"✅ Sentiment analysis model trained - Accuracy: {accuracy*100:.1f}%")
            
            return {
                'model_type': 'sentiment_analysis',
                'accuracy': accuracy * 100,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis model creation failed: {e}")
            return {}
    
    def create_volatility_prediction_model(self) -> Dict[str, Any]:
        """Create volatility prediction model"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Generate synthetic volatility data
            np.random.seed(42)
            n_samples = 5000
            
            # Create features (price, volume, time-based features)
            features = np.random.randn(n_samples, 15)
            
            # Create target (volatility)
            target = np.abs(features[:, 0]) + np.random.randn(n_samples) * 0.05
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models['volatility_prediction'] = model
            
            self.model_performance['volatility_prediction'] = {
                'mse': mse,
                'r2_score': r2,
                'accuracy': max(0, r2) * 100
            }
            
            logger.info(f"✅ Volatility prediction model trained - R²: {r2:.3f}")
            
            return {
                'model_type': 'volatility_prediction',
                'r2_score': r2,
                'accuracy': max(0, r2) * 100,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Volatility prediction model creation failed: {e}")
            return {}
    
    def create_market_regime_detection_model(self) -> Dict[str, Any]:
        """Create market regime detection model"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            
            # Generate synthetic market data
            np.random.seed(42)
            n_samples = 3000
            
            # Create features (price, volume, volatility, momentum)
            features = np.random.randn(n_samples, 10)
            
            # Add regime-specific patterns
            # Regime 1: Trending (high momentum, low volatility)
            features[:1000, 0] += 2  # High momentum
            features[:1000, 1] -= 1  # Low volatility
            
            # Regime 2: Volatile (high volatility, low momentum)
            features[1000:2000, 0] -= 1  # Low momentum
            features[1000:2000, 1] += 2  # High volatility
            
            # Regime 3: Sideways (low momentum, low volatility)
            features[2000:, 0] -= 0.5  # Low momentum
            features[2000:, 1] -= 0.5  # Low volatility
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Cluster for regime detection
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Evaluate clustering
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            
            self.models['market_regime_detection'] = kmeans
            self.models['regime_scaler'] = scaler
            
            self.model_performance['market_regime_detection'] = {
                'silhouette_score': silhouette_avg,
                'n_clusters': 3,
                'regime_labels': ['Trending', 'Volatile', 'Sideways']
            }
            
            logger.info(f"✅ Market regime detection model trained - Silhouette: {silhouette_avg:.3f}")
            
            return {
                'model_type': 'market_regime_detection',
                'silhouette_score': silhouette_avg,
                'n_regimes': 3,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Market regime detection model creation failed: {e}")
            return {}
    
    def predict_price_movement(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict price movement using trained model"""
        try:
            if 'price_prediction' not in self.models:
                return {'error': 'Price prediction model not trained'}
            
            model = self.models['price_prediction']
            prediction = model.predict(features.reshape(1, -1))[0]
            
            # Calculate confidence based on model performance
            confidence = self.model_performance.get('price_prediction', {}).get('accuracy', 0) / 100
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'direction': 'UP' if prediction > 0 else 'DOWN',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Price prediction failed: {e}")
            return {'error': str(e)}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of given text"""
        try:
            if 'sentiment_analysis' not in self.models:
                return {'error': 'Sentiment analysis model not trained'}
            
            model = self.models['sentiment_analysis']
            vectorizer = self.models['sentiment_vectorizer']
            
            # Vectorize text
            text_vector = vectorizer.transform([text])
            
            # Predict sentiment
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            
            sentiment_map = {0: 'NEGATIVE', 1: 'POSITIVE', 2: 'NEUTRAL'}
            sentiment = sentiment_map.get(prediction, 'UNKNOWN')
            
            confidence = max(probabilities)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    'negative': probabilities[0],
                    'positive': probabilities[1],
                    'neutral': probabilities[2]
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return {'error': str(e)}
    
    def detect_market_regime(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            if 'market_regime_detection' not in self.models:
                return {'error': 'Market regime detection model not trained'}
            
            kmeans = self.models['market_regime_detection']
            scaler = self.models['regime_scaler']
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Predict regime
            regime = kmeans.predict(features_scaled)[0]
            
            regime_labels = ['Trending', 'Volatile', 'Sideways']
            regime_name = regime_labels[regime]
            
            # Calculate distance to cluster center for confidence
            distances = kmeans.transform(features_scaled)[0]
            confidence = 1 / (1 + min(distances))
            
            return {
                'regime': regime_name,
                'regime_id': regime,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            return {'error': str(e)}
    
    def run_ml_pipeline(self) -> Dict[str, Any]:
        """Run complete ML pipeline"""
        logger.info("🚀 Starting ML model integration pipeline...")
        
        results = {}
        
        # Create all models
        models_to_create = [
            ('price_prediction', self.create_price_prediction_model),
            ('sentiment_analysis', self.create_sentiment_analysis_model),
            ('volatility_prediction', self.create_volatility_prediction_model),
            ('market_regime_detection', self.create_market_regime_detection_model)
        ]
        
        for model_name, create_func in models_to_create:
            logger.info(f"🤖 Creating {model_name} model...")
            results[model_name] = create_func()
        
        # Test models
        logger.info("🧪 Testing ML models...")
        
        # Test price prediction
        test_features = np.random.randn(20)
        price_pred = self.predict_price_movement(test_features)
        results['price_prediction_test'] = price_pred
        
        # Test sentiment analysis
        test_text = "Market is showing strong bullish momentum with positive indicators"
        sentiment_pred = self.analyze_sentiment(test_text)
        results['sentiment_analysis_test'] = sentiment_pred
        
        # Test market regime detection
        test_regime_features = np.random.randn(10)
        regime_pred = self.detect_market_regime(test_regime_features)
        results['market_regime_test'] = regime_pred
        
        logger.info("✅ ML pipeline completed successfully")
        
        return results

class FeatureEngineering:
    """Feature engineering for ML models"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        try:
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Technical indicators
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'] = df['ema_12'] - df['ema_26']
            df['bollinger_upper'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
            df['bollinger_lower'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)

def main():
    """Run ML model integration"""
    ml_system = MLModelIntegration()
    results = ml_system.run_ml_pipeline()
    
    print("\n" + "="*80)
    print("🤖 ML MODEL INTEGRATION RESULTS")
    print("="*80)
    
    print(f"\n📋 MODEL CREATION RESULTS:")
    for model_name, result in results.items():
        if 'test' not in model_name and result:
            print(f"\n   {model_name.upper()}:")
            for key, value in result.items():
                if key != 'timestamp':
                    print(f"     {key}: {value}")
    
    print(f"\n🧪 MODEL TESTING RESULTS:")
    for model_name, result in results.items():
        if 'test' in model_name and result:
            print(f"\n   {model_name.upper()}:")
            for key, value in result.items():
                if key != 'timestamp':
                    print(f"     {key}: {value}")
    
    print("\n" + "="*80)
    
    # Check if all models were created successfully
    success = all(
        result and 'error' not in result 
        for model_name, result in results.items() 
        if 'test' not in model_name
    )
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
