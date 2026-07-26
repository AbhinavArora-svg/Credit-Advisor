# feature_engineer.py
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class CreditFeatureEngineer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def engineer_features(self, df):
        print("🔧 Engineering features...")
        
        # Financial ratios
        df['emi_to_income_ratio'] = (df['monthly_emi'] * 12) / df['annual_income']
        df['savings_to_income_ratio'] = df['savings_balance'] / df['annual_income']
        df['investment_to_income_ratio'] = df['investment_value'] / df['annual_income']
        
        # Credit utilization metrics
        df['effective_utilization'] = df['credit_card_utilization'] * (1 + df['existing_loans'] * 0.1)
        
        # Stability scores
        df['employment_stability'] = np.log1p(df['years_at_job']) / 10
        df['income_consistency'] = df['income_stability'] * (1 + df['employment_stability'])
        
        # Behavioral metrics
        df['banking_behavior_score'] = 1 / (1 + df['bounced_checks'] + df['credit_inquiries'] * 0.1)
        df['financial_reserves'] = (df['savings_balance'] + df['investment_value']) / (df['monthly_emi'] * 6)
        
        # Risk flags
        df['high_dti_flag'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
        df['low_savings_flag'] = (df['savings_to_income_ratio'] < 0.3).astype(int)
        df['credit_seeker_flag'] = (df['credit_inquiries'] > 5).astype(int)
        
        # Employment type encoding
        df['is_salaried'] = (df['employment_type'] == 'Salaried').astype(int)
        df['is_self_employed'] = (df['employment_type'] == 'Self-Employed').astype(int)
        df['is_business'] = (df['employment_type'] == 'Business').astype(int)
        
        print(f"✅ Created {len([col for col in df.columns if col not in ['applicant_id', 'employment_type', 'recommended_credit_limit']])} features")
        
        return df

def main():
    print("🔧 Starting Feature Engineering...")
    
    # Load data
    df = pd.read_csv('credit_applicant_data.csv')
    print(f"📁 Loaded data with {len(df)} records")
    
    # Engineer features
    engineer = CreditFeatureEngineer()
    df_with_features = engineer.engineer_features(df)
    
    # Prepare features for modeling
    exclude_cols = ['applicant_id', 'employment_type', 'recommended_credit_limit']
    feature_cols = [col for col in df_with_features.columns if col not in exclude_cols]
    
    X = df_with_features[feature_cols]
    y = df_with_features['recommended_credit_limit']
    
    # Save processed data
    df_with_features.to_csv('processed_credit_data.csv', index=False)
    
    print(f"✅ Feature engineering completed!")
    print(f"📊 Final feature matrix: {X.shape}")
    print(f"🎯 Target variable range: ₹{y.min():,.0f} - ₹{y.max():,.0f}")
    print(f"📁 Saved to: processed_credit_data.csv")
    
    # Show feature summary
    print(f"\n📋 Feature summary:")
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature categories: Financial ratios, Risk flags, Behavioral metrics")

if __name__ == "__main__":
    main()
