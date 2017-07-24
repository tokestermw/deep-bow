sudo apt-get update && sudo apt-get install -y

sudo apt-get install -y \
    git \
    python-dev \
    python-pip \
    gfortran \
    g++ \
    cmake \
    wget \
    nano \
    tmux \
    htop \
    glances

sudo pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 

git clone https://github.com/pytorch/text.git
(cd text && sudo python setup.py install)

git clone https://github.com/tokestermw/deep-bow.git
(cd deep-bow && sudo pip install -r requirements.txt)

sudo apt-get clean
sudo apt-get autoremove
sudo rm -rf /var/lib/apt/lists/*