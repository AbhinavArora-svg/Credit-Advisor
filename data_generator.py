# data_generator.py
import pandas as pd
import numpy as np
from faker import Faker
import random
import os

class FinancialDataGenerator:
    def __init__(self, seed=42):
        self.fake = Faker()
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_applicant_profile(self, num_applicants=800):
        print("📊 Generating financial applicant data...")
        
        applicants = []
        for i in range(num_applicants):
            # Basic demographics
            age = random.randint(25, 65)
            employment_type = random.choice(['Salaried', 'Self-Employed', 'Business'])
            years_at_job = random.randint(1, 30)
            
            # Income details
            if employment_type == 'Salaried':
                base_income = random.randint(300000, 2500000)
                income_stability = random.uniform(0.7, 1.0)
            else:
                base_income = random.randint(200000, 5000000)
                income_stability = random.uniform(0.5, 0.9)
            
            # Financial obligations
            existing_loans = random.randint(0, 3)
            monthly_emi = base_income * random.uniform(0.1, 0.4) / 12
            credit_card_utilization = random.uniform(0.1, 0.9)
            
            # Assets and savings
            savings_balance = base_income * random.uniform(0.5, 3.0)
            investment_value = base_income * random.uniform(0.2, 2.0)
            property_value = base_income * random.uniform(1.0, 10.0)
            
            # Banking behavior
            avg_account_balance = base_income / 12 * random.uniform(0.5, 2.0)
            bounced_checks = random.randint(0, 5)
            credit_inquiries = random.randint(0, 10)
            
            # Calculate risk score and credit limit
            debt_to_income_ratio = (monthly_emi * 12) / base_income
            risk_score = self.calculate_risk_score(
                age, employment_type, debt_to_income_ratio, 
                credit_card_utilization, bounced_checks
            )
            
            base_limit = base_income * random.uniform(0.2, 0.6)
            adjusted_limit = base_limit * risk_score
            
            applicant = {
                'applicant_id': f'APP_{i:04d}',
                'age': age,
                'employment_type': employment_type,
                'years_at_job': years_at_job,
                'annual_income': base_income,
                'income_stability': income_stability,
                'existing_loans': existing_loans,
                'monthly_emi': monthly_emi,
                'debt_to_income_ratio': debt_to_income_ratio,
                'credit_card_utilization': credit_card_utilization,
                'savings_balance': savings_balance,
                'investment_value': investment_value,
                'property_value': property_value,
                'avg_account_balance': avg_account_balance,
                'bounced_checks': bounced_checks,
                'credit_inquiries': credit_inquiries,
                'risk_score': risk_score,
                'recommended_credit_limit': max(50000, adjusted_limit),
            }
            applicants.append(applicant)
        
        df = pd.DataFrame(applicants)
        return df
    
    def calculate_risk_score(self, age, employment, dti, utilization, bounced_checks):
        score = 1.0
        
        if age < 35:
            score *= 0.8
        elif age > 55:
            score *= 1.1
        
        if employment == 'Salaried':
            score *= 1.0
        elif employment == 'Self-Employed':
            score *= 0.9
        else:
            score *= 0.8
        
        if dti > 0.4:
            score *= 0.7
        elif dti > 0.3:
            score *= 0.9
        
        if utilization > 0.7:
            score *= 0.8
        
        if bounced_checks > 2:
            score *= 0.7
        
        return max(0.1, min(1.0, score))

def main():
    print("🏦 Starting Data Generation...")
    
    # Generate data
    generator = FinancialDataGenerator()
    df = generator.generate_applicant_profile(800)
    
    # Save to CSV
    df.to_csv('credit_applicant_data.csv', index=False)
    
    print(f"✅ Generated {len(df)} applicant records")
    print(f"📁 Saved to: credit_applicant_data.csv")
    print(f"📊 Data shape: {df.shape}")
    print(f"💰 Credit limit range: ₹{df['recommended_credit_limit'].min():,.0f} - ₹{df['recommended_credit_limit'].max():,.0f}")
    
    # Show sample
    print("\n📋 Sample data:")
    print(df[['applicant_id', 'annual_income', 'risk_score', 'recommended_credit_limit']].head())

if __name__ == "__main__":
    main()
