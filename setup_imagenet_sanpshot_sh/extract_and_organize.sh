cd /mnt/data

# Check the file is there
ls -lh ILSVRC2017_CLS-LOC.tar.gz

# Extract (this will take 30-60 minutes)
tar -xzf ILSVRC2017_CLS-LOC.tar.gz

# Create organized directory structure
mkdir imagenet

# Move train, val, test folders
mv ILSVRC/Data/CLS-LOC/train imagenet/
mv ILSVRC/Data/CLS-LOC/val imagenet/
mv ILSVRC/Data/CLS-LOC/test imagenet/

# Organize validation data into subdirectories
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valprep.sh imagenet/val/
cd /mnt/data/imagenet/val
sh valprep.sh
rm valprep.sh

# Verify everything
cd /mnt/data
ls -la imagenet/
ls imagenet/train/ | wc -l  # Should show 1000