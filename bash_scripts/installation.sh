module load Anaconda3/5.0.1-fasrc02
# conda remove --name 2tf1.14_cuda10 --all
# conda create -n 2tf1.14_cuda10 python=3.6 numpy six wheel
source activate 2tf1.14_cuda10
pip install --upgrade tensorflow-gpu==1.14
pip install -r requirements.txt