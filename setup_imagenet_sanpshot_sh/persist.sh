# Persist across reboots
DEV=$([ -b /dev/xvdf ] && echo /dev/xvdf || echo /dev/nvme1n1)
UUID=$(sudo blkid -s UUID -o value "$DEV")
echo "UUID=$UUID /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null

# get the UUID for /dev/nvme1n1
UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)
# add an fstab entry
echo "UUID=$UUID /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
# quick check (re-mount everything)
sudo mount -a
df -h /mnt/data