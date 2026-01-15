# See disks; you should see ~400G on xvdf or nvme1n1
lsblk
# Format the NEW data disk (run ONE of these; use the device that exists)
sudo mkfs -t ext4 /dev/xvdf || sudo mkfs -t ext4 /dev/nvme1n1
# Mount it and give yourself ownership
sudo mkdir -p /mnt/data
sudo mount /dev/xvdf /mnt/data || sudo mount /dev/nvme1n1 /mnt/data
sudo chown -R ec2-user:ec2-user /mnt/data
df -h /mnt/data