import os

import source_segment.config as config
import source_segment.segmentation_tools.train as train
import source_segment.segmentation_tools.predict as predict
import source_segment.segmentation_tools.data_handling as data_handling
import source_segment.segmentation_tools.segmentation_config as seg_config
import source_segment.config as cf


if __name__ == '__main__':
    print("Initializing segmentation toolkit...\n\n")

    ch = input(
        f"1) Visualise model on val set\n"
        f"2) Predict on custom image(s)\n"
        "3) Training options\n"
        )
    if ch == '1':
        data_handling.init()
        train.visualise_generator(data_loader='val')

    elif ch == '2':
        path = input("Enter the full path to a single image, or a folder containing images: ")

        pred_cls = predict.Predict(verbose=True)
        pred_cls.predict(images=path)

    elif ch == '3':
        ch = input("1) New model using random initialisation\n"
                   "2) Continue training from last saved checkpoint\n"
                   "3) Fine tune model from pre-trained weights\n")
        if ch == '1':
            train.train(continue_training=False, load_weights_for_fine_tune=False)
        elif ch == '2':
            train.train(continue_training=True, load_weights_for_fine_tune=False)
        elif ch == '3':
            train.train(continue_training=False, load_weights_for_fine_tune=True)
