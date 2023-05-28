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
    - You can follow the on-screen instructions to run the program.
    - Input folder contains sample images to test the program.
   Try feeding the folder path as input to the program, and try to feed a single image path as input to the program.


### What the outputs from prediction mean:

        predict() of Predict class in predict.py file is the main function that is called to predict the condition of the retina for the given image(s).
        Predict the condition of the retina for the given image(s)

        Parameters
        ----------
        images: str or list of str
            Argument can be one of the following:
                - string path to a single image
                - string path to a folder containing images
                - list of string paths to images

        Returns
        -------
        outputs: dict of str: int
            You'll get a dictionary of image paths and their corresponding predicted labels.
            The labels are encoded as integers, so you'll need to decode them to get the actual labels given below.

                FULL_LABELS = {
                    0: 'No_DR',
                    1: 'Mild',
                    2: 'Moderate',
                    3: 'Severe',
                    4: 'Proliferate_DR',
                }
            
            Example:
                >>> Predict().predict(cf.INPUT_FOLDER)  # contains 3 images

                Output:
                    {'<cf.INPUT_FOLDER>/test_IDRiD_016-Severe-3.jpg': 'Severe',
                     '<cf.INPUT_FOLDER>/test_IDRiD_047-No_DR-0.jpg': 'No_DR',
                     '<cf.INPUT_FOLDER>/train_IDRiD_236-Moderate-2.jpg': 'Moderate'}


