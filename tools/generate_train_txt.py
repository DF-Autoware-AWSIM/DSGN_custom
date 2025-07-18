import os

# Paths to your KITTI folders
#train_img_dir = '/home/arka/DSGN/data/kitti/training/image_2'
#test_img_dir = '/home/arka/DSGN/data/kitti/testing/image_2'

train_img_dir = '/home/arka/DSGN/data/awsim/training/image_2'
test_img_dir = '/home/arka/DSGN/data/awsim/testing/image_2'
# Output files
train_txt = 'train.txt'
val_txt = 'val.txt'
test_txt = 'test.txt'

# Helper function to get sorted image IDs
def get_image_ids(img_dir):
    return sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.endswith('.png')
    ])

# Get all train image IDs
train_ids_all = get_image_ids(train_img_dir)
train_ids = train_ids_all[:272]
val_ids = train_ids_all[272:]

# Get test image IDs
test_ids = get_image_ids(test_img_dir)

# Write train.txt
with open(train_txt, 'w') as f:
    for img_id in train_ids:
        f.write(f"{img_id}\n")

# Write val.txt
with open(val_txt, 'w') as f:
    for img_id in val_ids:
        f.write(f"{img_id}\n")

# Write test.txt
with open(test_txt, 'w') as f:
    for img_id in test_ids:
        f.write(f"{img_id}\n")

print("Generated train.txt, val.txt, and test.txt successfully.")
