`````bash
sudo apt install portaudio19-dev

cd godfrey-pipe
hf download davidscripka/openwakeword --local-dir ../models/openwakeword

export OPENWAKEWORD_MODEL_PATH="/home/user/models/openwakeword"
`````