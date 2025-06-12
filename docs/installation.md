conda create -n dawn python=3.8
conda activate dawn

cd data
git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin
sh install.sh

cd ../../