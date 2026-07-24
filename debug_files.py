# debug_files.py - Check what files exist
import os
import joblib
import pandas as pd

print("🔍 DEBUG: Checking project files...")
print("=" * 50)

# List all files in current directory
files = os.listdir('.')
print("📁 Files in current directory:")
for file in sorted(files):
    if file.endswith(('.pkl', '.csv', '.py')):
        print(f"   {file}")

print("\n" + "=" * 50)

# Check specific model files
model_files = ['credit_model.pkl', 'best_credit_model.pkl', 'feature_names.pkl']
data_files = ['credit_applicant_data.csv', 'processed_credit_data.csv']

print("🔧 Model Files Status:")
for file in model_files:
    if os.path.exists(file):
        print(f"   ✅ {file} - EXISTS")
        try:
            if file.endswith('.pkl'):
                obj = joblib.load(file)
                print(f"        Type: {type(obj).__name__}")
                if hasattr(obj, 'shape'):
                    print(f"        Shape: {obj.shape}")
                elif isinstance(obj, list):
                    print(f"        Length: {len(obj)}")
        except Exception as e:
            print(f"        Error loading: {e}")
    else:
        print(f"   ❌ {file} - MISSING")

print("\n📊 Data Files Status:")
for file in data_files:
    if os.path.exists(file):
        print(f"   ✅ {file} - EXISTS")
        try:
            df = pd.read_csv(file)
            print(f"        Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"        Columns: {list(df.columns)}")
        except Exception as e:
            print(f"        Error reading: {e}")
    else:
        print(f"   ❌ {file} - MISSING")

print("\n" + "=" * 50)
print("🎯 Next steps based on what's available:")
if 'credit_model.pkl' in files and 'credit_applicant_data.csv' in files:
    print("✅ Ready to run explainability.py!")
else:
    if 'credit_applicant_data.csv' not in files:
        print("1. Run: python data_generator.py")
    if 'credit_model.pkl' not in files:
        print("2. Run: python model_trainer.py")
