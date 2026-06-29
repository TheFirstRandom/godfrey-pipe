# Installation

**Install build dependencies**

```bash
sudo apt install pipx
pipx install uv
```

**Install godfrey-pipe and dependencies**

```bash
curl -fsSL https://ollama.com/install.sh | sh

sudo apt install python3-paho-mqtt

git clone https://github.com/TheFirstRandom/godfrey-pipe.git
cd godfrey-pipe
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
sudo chmod +x scripts/*
```

**Create the systemd service**

```bash
cat << EOF | sudo tee /etc/systemd/system/godfrey-server.service 
[Unit]
Description=Godfrey Server
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/godfrey-pipe
ExecStart=/home/user/godfrey-pipe/.venv/bin/python src/godfrey_server/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now godfrey-server.service
```