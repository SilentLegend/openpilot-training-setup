#!/usr/bin/env bash
set -e

# 1. Systeem updaten
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential software-properties-common python3-dev python3-pip python3-venv

# 2. NVIDIA GPU, CUDA & cuDNN (optioneel)
if lspci | grep -i nvidia &> /dev/null; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update
  sudo apt install -y nvidia-driver-550 cuda-toolkit-12-4 libcudnn8 libcudnn8-dev
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# 3. Projectmap maken en virtualenv
PROJECT_DIR="$HOME/openpilot-training-setup"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
python3 -m venv openpilot-env
source openpilot-env/bin/activate
pip install --upgrade pip

# 4. Clone en build OpenPilot
git clone --recurse-submodules https://github.com/commaai/openpilot.git
cd openpilot
tools/ubuntu_setup.sh
scons -j"$(nproc)"
cd ..

# 5. ML-libraries installeren
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy matplotlib seaborn scikit-learn pandas jupyter notebook onnx onnxruntime-gpu capnp cereal

# 6. Voorbeeld-scripts neerzetten
cat > model_config.py << 'EOF'
import torch, torch.nn as nn
from torchvision.models import efficientnet_b2
class OpenPilotTrafficLightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b2(pretrained=True).features
        self.gru = nn.GRU(1408,512,batch_first=True)
        self.head = nn.Linear(512,4)
    def forward(self,x):
        f=self.backbone(x).view(x.size(0),-1)
        g,_=self.gru(f.unsqueeze(1))
        return self.head(g.squeeze(1))
def export_to_onnx(model,output="traffic_light_model.onnx"):
    model.eval()
    dummy=torch.randn(1,3,128,256)
    torch.onnx.export(model,dummy,output,export_params=True,opset_version=12,
                      input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{0:'batch'},'output':{0:'batch'}})
    print("Exported to",output)
EOF

cat > train_traffic_lights.py << 'EOF'
#!/usr/bin/env python3
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from model_config import OpenPilotTrafficLightModel
def train_model(model,train_loader,val_loader,epochs=50):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)
    crit=nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=1e-3)
    for e in range(epochs):
        model.train();tl=0
        for x,y in train_loader:
            x,y=x.to(dev),y.to(dev)
            opt.zero_grad()
            loss=crit(model(x),y);loss.backward();opt.step()
            tl+=loss.item()
        model.eval();vl=0;corr=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(dev),y.to(dev)
                out=model(x);vl+=crit(out,y).item()
                corr+=out.argmax(1).eq(y).sum().item()
        acc=100*corr/len(val_loader.dataset)
        echo "Epoch $e: Train $(echo "$tl/${#train_loader[@]}" | bc -l), Val $(echo "$vl/${#val_loader[@]}" | bc -l), Acc $acc%"
if __name__=="__main__":
    echo "Voeg eigen DataLoader toe en run train_model()"
EOF
chmod +x setup.sh train_traffic_lights.py

echo "Setup compleet. Gebruik 'source openpilot-env/bin/activate' voor omgeving."
