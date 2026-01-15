cd /tmp
wget https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0.tar.gz
tar -xzf aria2-1.37.0.tar.gz
cd aria2-1.37.0

# Install build dependencies
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc-c++ libxml2-devel openssl-devel

# Compile and install
./configure
make
sudo make install

# Verify
aria2c --version