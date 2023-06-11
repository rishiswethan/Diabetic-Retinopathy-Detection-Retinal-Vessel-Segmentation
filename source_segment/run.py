import os

import source_segment.config as config
import source_segment.segmentation_tools.train as train
import source_segment.segmentation_tools.predict as predict


if __name__ == '__main__':
    print("Initializing segmentation toolkit...\n\n")

    ch = input(
        f"1) Detect all images in '{config.INPUT_FOLDER.split(os.sep)[-2]}' folder\n"
        "2) Training options\n"
        )
    if ch == '1':
        predict.run_images()
    elif ch == '2':
        ch = input("1) New model using random initialisation\n"
                   "2) Continue training from last saved checkpoint\n"
                   "3) Fine tune model from pre-trained weights\n")
        if ch == '1':
            train.train(continue_training=False, load_weights_for_fine_tune=False)
        elif ch == '2':
            train.train(continue_training=True, load_weights_for_fine_tune=False)
        elif ch == '3':
            train.train(continue_training=False, load_weights_for_fine_tune=True)
