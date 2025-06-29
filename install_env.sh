conda create -n vagen_chenliang python=3.10 -y
conda activate vagen_chenliang

# verl
git clone https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

bash scripts/install.sh