# Start a new tmux session named "imagenet"
tmux new -s imagenet

# Now inside tmux, run:
cd /mnt/data

MAGNET_LINK='magnet:?xt=urn:btih:943977d8c96892d24237638335e481f3ccd54cfb&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce'

aria2c \
    --dir=/mnt/data \
    --enable-rpc=false \
    --max-concurrent-downloads=1 \
    --continue=true \
    --seed-time=0 \
    "$MAGNET_LINK"