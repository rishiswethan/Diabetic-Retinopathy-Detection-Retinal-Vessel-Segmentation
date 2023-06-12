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
       - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --force-reinstall`
         - There should be a download size of 2.3GB if it is downloading the GPU version correctly. If it's something like 600MB, that's the CPU version.
         - If that didn't work, simply install the CPU version for now. It'll slow down transcription but it'll work.
           - `pip install torch`
           
3) `python run.py`
    - You can follow the on-screen instructions to run the program.
    - Input folder contains sample images to test the program.
   Try feeding the folder path, and try to feed a single image path as input to the program.


### What the outputs and parameters from predict.py mean:

     - Predict class
        Class to help initialise the model and encapsulate the predict function. Use this class to use the predict function encapsulated within it.
        
        Parameters:
        ----------
        model_save_path: str
            path to the model to load
        device: str, optional
            device to run the model on. If None, will use cuda by default if available, else will use cpu.
            Options: "cpu" or "cuda".
        batch_size: int, optional
            batch size to use when predicting. Reduce if running out of memory when using cuda
        image_size: int, optional
            image size to use when predicting. Do not change unless you trained the model with a different image size
        verbose: bool, optional
            whether to print out information when predicting
        labels: list, optional
            list of labels to use when predicting. Do not change unless you trained the model with different labels
        best_hp_json_save_path: str, optional
            path to the json file containing the best hyperparameters from the tuning process
        num_classes: int, optional
            number of classes to use when predicting. Do not change unless you trained the model with a different number of classes

     - predict()
        function of Predict class in predict.py file is the main function that is called to predict the condition of the retina for the given image(s).
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
            
            Example:
                >>> Predict().predict(cf.INPUT_FOLDER)  # contains 3 images

                Output:
                    {'<cf.INPUT_FOLDER>/test_IDRiD_016-Severe-3.jpg': 'Severe',
                     '<cf.INPUT_FOLDER>/test_IDRiD_047-No_DR-0.jpg': 'No_DR',
                     '<cf.INPUT_FOLDER>/train_IDRiD_236-Moderate-2.jpg': 'Moderate'}


