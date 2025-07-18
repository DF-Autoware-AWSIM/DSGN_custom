import os
import shutil

def copy_file_n_times(src_file, n):
    base_dir = os.path.dirname(src_file)
    base_name = os.path.splitext(os.path.basename(src_file))[0]
    ext = os.path.splitext(src_file)[1]

    for i in range(1, n + 1):
        new_name = f"{i:06d}{ext}"  # e.g., 000001.txt, 000002.txt, ...
        dst_file = os.path.join(base_dir, new_name)
        shutil.copy(src_file, dst_file)
        print(f"Copied to {dst_file}")

# Example usage
copy_file_n_times("/home/arka/DSGN/data/awsim/training/calib/000000.txt", 282)
