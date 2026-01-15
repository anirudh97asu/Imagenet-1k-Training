aws ec2 create-snapshot \
    --volume-id "volume-id"\
    --description "ImageNet Dataset" \
    --region us-east-2 \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=ImageNet-Snapshot}]'