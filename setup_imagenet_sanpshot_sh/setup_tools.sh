sudo dnf -y update
sudo dnf -y install unzip tmux jq wget tar python3-pip
python3 -m pip install --user --upgrade kaggle
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json # paste your Kaggle token JSON
chmod 600 ~/.kaggle/kaggle.json
kaggle --version

mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'JSON'
{"username":"given","key":"given"}
JSON
chmod 600 ~/.kaggle/kaggle.json
# sanity check: list competition files (should NOT 403)
kaggle competitions files -c imagenet-object-localization-challenge | head