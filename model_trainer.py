# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

class CreditModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
    
    def train_models(self, X, y):
        print("🤖 Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📈 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        results = {}
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            start_time = time.time()
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            percentage_error = (mae / y_test.mean()) * 100
            
            training_time = time.time() - start_time
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'percentage_error': percentage_error,
                'training_time': training_time
            }
            
            print(f"✅ {name}:")
            print(f"   MAE: ₹{mae:,.0f} | RMSE: ₹{rmse:,.0f}")
            print(f"   R²: {r2:.4f} | Error: {percentage_error:.1f}%")
            print(f"   Time: {training_time:.2f}s")
        
        # Find best model
        self.best_model_name = min(results, key=lambda x: results[x]['mae'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n🎉 BEST MODEL: {self.best_model_name}")
        print(f"🏆 Performance: MAE ₹{results[self.best_model_name]['mae']:,.0f}, " 
              f"R² {results[self.best_model_name]['r2']:.4f}")
        
        return results, X_test_scaled, y_test, X_train.columns
    
    def save_models(self, results, feature_names):
        """Save trained models and metadata"""
        print("💾 Saving models...")
        
        # Save best model
        joblib.dump(self.best_model, 'best_credit_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'credit_scaler.pkl')
        
        # Save feature names
        joblib.dump(list(feature_names), 'feature_names.pkl')
        
        # Save model results
        results_summary = {
            'best_model': self.best_model_name,
            'models': {name: {k: v for k, v in info.items() if k != 'model'} 
                      for name, info in results.items()}
        }
        joblib.dump(results_summary, 'training_results.pkl')
        
        print("✅ Models saved successfully!")
        print("   - best_credit_model.pkl")
        print("   - credit_scaler.pkl") 
        print("   - feature_names.pkl")
        print("   - training_results.pkl")
    
    def plot_results(self, results, y_test):
        """Create performance visualization"""
        plt.figure(figsize=(12, 8))
        
        # Model comparison
        model_names = list(results.keys())
        maes = [results[name]['mae'] for name in model_names]
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(model_names, maes, color=['blue', 'green', 'orange', 'red'])
        plt.title('Model Performance (MAE)')
        plt.ylabel('Mean Absolute Error (₹)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, mae in zip(bars, maes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                    f'₹{mae:,.0f}', ha='center', va='bottom')
        
        # R² scores
        r2_scores = [results[name]['r2'] for name in model_names]
        
        plt.subplot(2, 2, 2)
        bars = plt.bar(model_names, r2_scores, color=['blue', 'green', 'orange', 'red'])
        plt.title('Model Performance (R² Score)')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, r2 in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{r2:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🤖 Starting Model Training...")
    
    # Load processed data
    df = pd.read_csv('processed_credit_data.csv')
    print(f"📁 Loaded processed data with {len(df)} records")
    
    # Prepare features and target
    exclude_cols = ['applicant_id', 'employment_type', 'recommended_credit_limit']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['recommended_credit_limit']
    
    print(f"📊 Features: {X.shape[1]}, Target: {y.shape[0]}")
    
    # Train models
    trainer = CreditModelTrainer()
    results, X_test, y_test, feature_names = trainer.train_models(X, y)
    
    # Save models
    trainer.save_models(results, feature_names)
    
    # Plot results
    trainer.plot_results(results, y_test)
    
    print(f"\n🎯 Training completed successfully!")
    print(f"📈 Best model: {trainer.best_model_name}")
    print(f"📊 Model performance saved to: model_performance.png")

if __name__ == "__main__":
    main()
