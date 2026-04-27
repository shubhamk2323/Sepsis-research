import zipfile
import os

data_dir = "c:/Users/shubh/Desktop/Model/Sepsis_Data"
zip_files = ["training_setA.zip", "training_setB.zip"]

for zf in zip_files:
    zip_path = os.path.join(data_dir, zf)
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted {zf} successfully!")
    except FileNotFoundError:
        print(f"Wait, {zf} is not downloaded yet or couldn't be found.")
    except Exception as e:
        print(f"Error extracting {zf}: {e}")

print("Done! Check Sepsis_Data folder.")
