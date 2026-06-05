import os

def rename_files(directory, prefix):
    if not os.path.exists(directory): return
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for i, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}_{i+1}{ext}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        if not os.path.exists(new_path) or old_path == new_path:
            os.rename(old_path, new_path)
        else:
            temp_path = os.path.join(directory, f"temp_{prefix}_{i+1}{ext}")
            os.rename(old_path, temp_path)
            os.rename(temp_path, new_path)

if __name__ == "__main__":
    rename_files("offline_preprocessing/test_cloth", "cloth")
    rename_files("offline_preprocessing/test_person", "person")
    print("Renamed files successfully!")
