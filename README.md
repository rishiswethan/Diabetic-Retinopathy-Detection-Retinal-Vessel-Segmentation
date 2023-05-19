This project was tested on Ubuntu 22.04 LTS with Python 3.10

### How to run the program:
1. Install python 3.10
2) Install everything you need
   - `python -m venv venv`
   - Activate the virtual environment
     - Linux/MacOS: `source venv/bin/activate`
     - Windows: `venv\Scripts\activate`
   - `pip install -r requirements.txt`
   - To setup the GPU version of pytorch, follow the instructions in this [link](https://github.com/openai/whisper/discussions/47).
     A quick summary of the steps is given below:
       - `pip uninstall torch`
       - `pip cache purge`
       - `pip3 install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir`
         - There should be a download size of 2.3GB if it is downloading the GPU version correctly. If it's something like 200MB, that's the CPU version.
         - If that didn't work, simply install the CPU version for now. It'll slow down transcription but it'll work.
           - `pip install torch`
           
3) `python run.py`
    - You can follow the on screen instructions to run the program.