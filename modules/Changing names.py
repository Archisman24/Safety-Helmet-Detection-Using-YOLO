import os
def rename_files(directory_path, new_name_prefix):
    # Get a list of all files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through each file and rename it
    for old_name in file_list:
        # Construct the new file name with the provided prefix
        new_name = f"{new_name_prefix}{old_name}"

        # Join the full paths for the old and new names
        old_path = os.path.join(directory_path, old_name)
        new_path = os.path.join(directory_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

# Provide the directory path and the new name prefix
directory_path = "Dataset(in XML format)/annotations"
new_name_prefix = "yolo"

# Call the function to rename files in the specified directory
rename_files(directory_path, new_name_prefix)
