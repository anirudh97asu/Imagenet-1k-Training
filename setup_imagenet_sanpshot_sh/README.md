# Simplified ImageNet Setup Guide for AWS EC2

## Overview
This guide walks you through downloading ImageNet once, creating a snapshot, and reusing it on GPU instances for training.

---

## Part 1: One-Time Setup (Download ImageNet)

### Step 1: Launch Setup Instance
1. Launch a **t3.xlarge** EC2 instance in **us-east-1**
2. Note your **instance ID** (e.g., `i-0123456789abcdef`)
3. Note your **availability zone** (e.g., `us-east-1c`)

### Step 2: Install Required Tools
SSH into your instance and run:
```bash
sudo apt-get update
sudo apt install python3-pip unzip aria2 -y
```

### Step 3: Configure AWS CLI
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Set credentials
export AWS_ACCESS_KEY_ID='YOUR_KEY'
export AWS_SECRET_ACCESS_KEY='YOUR_SECRET'
export AWS_DEFAULT_REGION='us-east-1'
```

### Step 4: Create and Attach Storage Volume
```bash
# Create 375GB volume (adjust zone to match your instance)
aws ec2 create-volume \
    --volume-type gp3 \
    --size 375 \
    --availability-zone ap-south-1 \
    --region ap-south-1 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=ImageNet-Data}]'
```

**Save the volume ID** from the output (e.g., `vol-0123456789abcdef`)

```bash
# Attach volume to instance
aws ec2 attach-volume \
    --volume-id vol-XXXXXXXXX \
    --instance-id i-XXXXXXXXX \
    --device /dev/xvdf \
    --region us-east-1

# Verify attachment
aws ec2 describe-volumes \
    --volume-ids vol-XXXXXXXXX \
    --region us-east-1 \
    --query 'Volumes[0].Attachments[0].State'
```

### Step 5: Format and Mount Volume
```bash
# Format the volume (try first command, if it fails use second)
sudo mkfs -t ext4 /dev/xvdf
# OR if above fails:
# sudo mkfs -t ext4 /dev/nvme1n1

# Create mount point and mount
sudo mkdir -p /mnt/data
sudo mount /dev/xvdf /mnt/data
# OR if device name differs:
# sudo mount /dev/nvme1n1 /mnt/data

# Set permissions
sudo chown -R ubuntu:ubuntu /mnt/data

# Verify mount
df -h
```

### Step 6: Download ImageNet
```bash
cd /mnt/data

# Download via torrent (runs in background)
MAGNET_LINK='magnet:?xt=urn:btih:943977d8c96892d24237638335e481f3ccd54cfb&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce'

aria2c \
    --dir=/mnt/data \
    --enable-rpc=false \
    --max-concurrent-downloads=1 \
    --continue=true \
    --seed-time=0 \
    "$MAGNET_LINK" &
```

**This will take several hours.** Monitor progress with `jobs` or `ps aux | grep aria2`

### Step 7: Extract and Organize Data
```bash
# Extract archive
tar -xzf ILSVRC2017_CLS-LOC.tar.gz

# Create organized structure
mkdir imagenet
mv ILSVRC/Data/CLS-LOC/train imagenet/
mv ILSVRC/Data/CLS-LOC/val imagenet/
mv ILSVRC/Data/CLS-LOC/test imagenet/

# Organize validation data into subdirectories
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valprep.sh imagenet/val/
cd /mnt/data/imagenet/val
sh valprep.sh
rm valprep.sh
```

### Step 8: Create Snapshot
```bash
# Create snapshot (replace vol-XXXXX with your volume ID)
aws ec2 create-snapshot \
    --volume-id vol-XXXXXXXXX \
    --description "ImageNet Dataset" \
    --region us-east-1 \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=ImageNet-Snapshot}]'
```

**Save the snapshot ID** from output (e.g., `snap-0123456789abcdef`)

```bash
# Monitor snapshot progress (takes 3-4 hours)
aws ec2 describe-snapshots \
    --snapshot-ids snap-XXXXXXXXX \
    --region us-east-1 \
    --query 'Snapshots[0].State'
```

Wait until state shows `completed`

### Step 9: Cleanup
```bash
# Stop the instance
aws ec2 stop-instances --instance-ids i-XXXXXXXXX --region us-east-1

# Delete volume (after instance stopped)
aws ec2 delete-volume --volume-id vol-XXXXXXXXX --region us-east-1

# Terminate instance (optional)
aws ec2 terminate-instances --instance-ids i-XXXXXXXXX --region us-east-1
```

---

## Part 2: Using ImageNet on GPU Instances

### Option A: User Data Script (Automatic on Launch)

When launching your GPU instance, add this as **User Data**:

```bash
#!/bin/bash

# Replace snap-XXXXX with your snapshot ID
SNAPSHOT_ID="snap-XXXXXXXXX"
REGION="us-east-1"

# Create volume from snapshot
VOLUME_ID=$(aws ec2 create-volume \
    --volume-type gp3 \
    --snapshot-id $SNAPSHOT_ID \
    --availability-zone $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone) \
    --region $REGION \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=ImageNet-Data}]' \
    --query 'VolumeId' \
    --output text)

# Wait for volume
aws ec2 wait volume-available --volume-ids $VOLUME_ID --region $REGION

# Get instance ID
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Attach volume
aws ec2 attach-volume \
    --volume-id $VOLUME_ID \
    --instance-id $INSTANCE_ID \
    --device /dev/xvdf \
    --region $REGION

# Wait and find device
sleep 30
DEVICE_NAME=$(lsblk -o NAME,MOUNTPOINT | grep -v 'MOUNTPOINT' | grep -E 'nvme|xvd' | awk '{print "/dev/"$1}' | tail -n1)

# Mount
mkdir -p /mnt/data
mount $DEVICE_NAME /mnt/data
chown -R ubuntu:ubuntu /mnt/data
```

### Option B: Manual Mount (After Instance Launch)

SSH into your GPU instance and run the commands from Option A manually.

### Step 10: Verify Data Access
```bash
# Check mount
df -h

# Verify ImageNet data
ls /mnt/data/imagenet/
# Should show: train/ val/ test/

# Check train classes
ls /mnt/data/imagenet/train/ | wc -l
# Should show: 1000
```

---

## Quick Reference

### Important IDs to Save
- **Snapshot ID**: `snap-XXXXXXXXX` (use this for all future GPU instances)
- **Volume size**: 375GB
- **Region**: us-east-1

### Data Location
Once mounted, ImageNet will be at: `/mnt/data/imagenet/`
- Training: `/mnt/data/imagenet/train/`
- Validation: `/mnt/data/imagenet/val/`
- Test: `/mnt/data/imagenet/test/`

### Tips
- Keep your snapshot ID safe - you'll need it for every training instance
- Launch GPU instances in the same region (us-east-1) as your snapshot
- The volume is automatically created and mounted via user data script
- No need to download ImageNet again!