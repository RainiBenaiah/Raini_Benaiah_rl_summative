import os

def list_model_files():
    project_root = "C:/Users/USER/Documents/last_rl_summative/Raini_Benaiah_rl_summative"
    models_pg_path = os.path.join(project_root, "models/pg")
    models_path = os.path.join(project_root, "models")

    print("Checking directory: models/pg")
    if os.path.exists(models_pg_path):
        print("Files found in models/pg:")
        for file in os.listdir(models_pg_path):
            full_path = os.path.join(models_pg_path, file)
            print(f"  - {file} (Full path: {full_path})")
    else:
        print("Directory models/pg does not exist")

    print("\nChecking directory: models")
    if os.path.exists(models_path):
        print("Files found in models:")
        for file in os.listdir(models_path):
            full_path = os.path.join(models_path, file)
            print(f"  - {file} (Full path: {full_path})")
    else:
        print("Directory models does not exist")

if __name__ == "__main__":
    list_model_files()