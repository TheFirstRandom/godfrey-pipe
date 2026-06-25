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
```