#!/bin/bash
set -e

SNAPSHOT_ID="snap-059cb17b74ead62b0"
REGION="us-east-1"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== EBS Volume Mount Script ===${NC}"

# Get instance metadata with IMDSv2 token
echo "Fetching instance metadata..."
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
AZ=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id)

echo "Instance ID: $INSTANCE_ID"
echo "Availability Zone: $AZ"

# Check if volume already exists from this snapshot
echo "Checking for existing volumes..."
EXISTING_VOLUME=$(aws ec2 describe-volumes \
    --region $REGION \
    --filters "Name=snapshot-id,Values=$SNAPSHOT_ID" "Name=availability-zone,Values=$AZ" "Name=status,Values=available,in-use" \
    --query 'Volumes[0].VolumeId' \
    --output text 2>/dev/null || echo "None")

if [ "$EXISTING_VOLUME" != "None" ] && [ "$EXISTING_VOLUME" != "" ]; then
    echo -e "${YELLOW}Found existing volume: $EXISTING_VOLUME${NC}"
    read -p "Use existing volume? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        VOLUME_ID=$EXISTING_VOLUME
        # Check if already attached
        ATTACHMENT=$(aws ec2 describe-volumes \
            --volume-ids $VOLUME_ID \
            --region $REGION \
            --query 'Volumes[0].Attachments[0].State' \
            --output text)
        
        if [ "$ATTACHMENT" == "attached" ]; then
            echo "Volume already attached, skipping to mount..."
            sleep 2
            DEVICE_NAME=$(lsblk -o NAME,SERIAL -p | grep $(echo $VOLUME_ID | sed 's/-//g') | awk '{print $1}')
        else
            # Attach existing volume
            echo "Attaching existing volume..."
            aws ec2 attach-volume \
                --volume-id $VOLUME_ID \
                --instance-id $INSTANCE_ID \
                --device /dev/xvdf \
                --region $REGION
            sleep 15
            DEVICE_NAME=$(lsblk -o NAME,SERIAL -p | grep $(echo $VOLUME_ID | sed 's/-//g') | awk '{print $1}')
        fi
    else
        EXISTING_VOLUME="None"
    fi
fi

if [ "$EXISTING_VOLUME" == "None" ] || [ "$EXISTING_VOLUME" == "" ]; then
    echo "Creating new volume from snapshot..."
    
    # Create with higher IOPS for faster initialization
    VOLUME_ID=$(aws ec2 create-volume \
        --volume-type gp3 \
        --iops 16000 \
        --throughput 1000 \
        --snapshot-id $SNAPSHOT_ID \
        --availability-zone $AZ \
        --region $REGION \
        --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=ImageNet-Data}]' \
        --query 'VolumeId' \
        --output text)

    echo "Volume ID: $VOLUME_ID"
    echo "Waiting for volume to become available..."
    
    # Progress indicator for volume creation
    while true; do
        STATE=$(aws ec2 describe-volumes --volume-ids $VOLUME_ID --region $REGION --query 'Volumes[0].State' --output text)
        if [ "$STATE" == "available" ]; then
            echo -e "\n${GREEN}✓ Volume available${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done

    echo "Attaching volume to instance..."
    aws ec2 attach-volume \
        --volume-id $VOLUME_ID \
        --instance-id $INSTANCE_ID \
        --device /dev/xvdf \
        --region $REGION > /dev/null

    echo "Waiting for volume to attach..."
    sleep 15

    # Find the newly attached device
    DEVICE_NAME=$(lsblk -o NAME,SERIAL -p | grep $(echo $VOLUME_ID | sed 's/-//g') | awk '{print $1}')
fi

if [ -z "$DEVICE_NAME" ]; then
    echo "Could not find device by serial. Trying alternative method..."
    DEVICE_NAME=$(lsblk -o NAME,SIZE -p | grep "323.2G\|400G" | awk '{print $1}' | head -1)
fi

if [ -z "$DEVICE_NAME" ]; then
    echo "ERROR: Could not find device"
    lsblk
    exit 1
fi

echo "Device found: $DEVICE_NAME"

# Check if already mounted
if mountpoint -q /mnt/data 2>/dev/null; then
    echo -e "${GREEN}✓ Already mounted at /mnt/data${NC}"
    df -h /mnt/data
    ls -lh /mnt/data | head -10
    exit 0
fi

echo "Creating mount point..."
sudo mkdir -p /mnt/data

echo -e "${YELLOW}Mounting volume (this may take 2-5 minutes for large volumes)...${NC}"

# Run mount in background and show progress
sudo mount $DEVICE_NAME /mnt/data &
MOUNT_PID=$!

# Show spinner while mounting
spin='-\|/'
i=0
elapsed=0
while kill -0 $MOUNT_PID 2>/dev/null; do
    i=$(( (i+1) %4 ))
    printf "\r${spin:$i:1} Mounting... ${elapsed}s elapsed"
    sleep 1
    elapsed=$((elapsed + 1))
    
    # If taking too long, show message
    if [ $elapsed -eq 60 ]; then
        echo -e "\n${YELLOW}Mount is taking longer than expected. This is normal for large EBS volumes from snapshots.${NC}"
    fi
    if [ $elapsed -eq 120 ]; then
        echo -e "\n${YELLOW}Still mounting... You can check 'dmesg | tail' in another terminal for details.${NC}"
    fi
done

wait $MOUNT_PID
MOUNT_STATUS=$?

if [ $MOUNT_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}✓ Mount successful!${NC}"
    
    sudo chown -R ec2-user:ec2-user /mnt/data
    
    echo -e "\n${GREEN}=== Mount Information ===${NC}"
    df -h /mnt/data
    echo -e "\n${GREEN}=== Directory Contents (first 10 items) ===${NC}"
    ls -lh /mnt/data | head -10
    
    echo -e "\n${GREEN}Volume is ready at /mnt/data${NC}"
    echo -e "${YELLOW}Tip: First access to files will be slower due to snapshot lazy loading${NC}"
else
    echo -e "\n${RED}✗ Mount failed with status $MOUNT_STATUS${NC}"
    exit 1
fi