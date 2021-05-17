sudo pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
sudo pip3 install dgl
sudo pip3 install pyinstrument
sudo pip3 install torchbiggraph
sudo pip3 install sklearn
sudo pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
sudo pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
sudo pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
sudo pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
sudo /usr/bin/pip3 install --upgrade pip
sudo /usr/local/bin/pip3 install  torch-geometric
aws s3 cp  s3://graphframes-sh2/newdung.pem /home/hadoop
sudo chmod 400 /home/hadoop/newdung.pem

echo "IdentityFile ~/newdung.pem" > ~/.ssh/config
echo "IdentitiesOnly yes" >> ~/.ssh/config
sudo chmod 644 ~/.ssh/config

sudo yum install -y htop
sudo yum install -y git
sudo yum install -y wireshark
mkdir /mnt/workspace
mkdir ~/workspace
