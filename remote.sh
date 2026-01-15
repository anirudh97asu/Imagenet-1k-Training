# # Replace with CURRENT IP from AWS console
# CURRENT_IP=35.153.207.20

# # Test 1: Can you reach it?
# ping -c 4 $CURRENT_IP

# # Test 2: Is port 22 open?
# nc -zv $CURRENT_IP 22

# # Or use telnet
# telnet $CURRENT_IP 22

# # Test 3: Try SSH with verbose output
# ssh -vvv -i "/home/anirudh97/Downloads/Image_Net_Training/tsai_imagenet.pem" ec2-user@$CURRENT_IP

# Check if you can reach the instance
ping 35.153.207.20

# Check if port 22 is accessible
nc -zv 35.153.207.20 22

# Check detailed SSH connection
ssh -i "/home/anirudh97/Downloads/Image_Net_Training/tsai_imagenet.pem" ec2-user@35.153.207.20