import os

if __name__ == '__main__':
    ch = input("1) Image processing\n"
               "2) Point cloud toolkit\n"
               "3) Segmentation toolkit\n"
                        "Enter your choice: ")

    # 360 tools
    if ch == '1':
        print("Initializing image processing toolkit...\n\n")

        import source.config as config
        import source.image as image
        import source.utils as utils
        import source.tools_360.change_tile_colour as change_tile_colour

        ch = input("1) Equirectungular to Cubemap\n"
                   "2) Cubemap to Equirectungular\n"
                   "3) Equirectungular to custom perspective\n"
                   "4) Custom perspective to Equirectungular\n"
                   "5) Detect borders\n"
                   "6) Change tile colour\n"
                   "Enter your choice: ")

        if ch == '1':
            image_name_input = input("Enter the name of the equirectangular image in the input_folder: ")
            image_name = utils.find_filename_match(image_name_input, config.INPUT_FOLDER)
            if image_name != image_name_input:
                print("The image name was changed to: " + image_name.split(config.INPUT_FOLDER)[1])

            if image_name == '':
                image.equi_cubemap()
            else:
                image.equi_cubemap(image_name)

            print(f"Done! Please check the \"{config.E2C_OUTPUT_FOLDER.split(config.PROJECT_NAME)[1][1:]}\" folder for the output image.")

        elif ch == '2':
            prefix = input(f"Enter the prefix of the multiple cubemap images in the input_folder(press enter to use default: {config.E2C_OUTPUT_IMAGE_PREFIX}): ")
            if prefix == '':
                image.cubemap_equi()
            else:
                image.cubemap_equi(prefix)

            print(f"Done! Please check the \"{config.C2E_OUTPUT_FOLDER.split(config.PROJECT_NAME)[1][1:]}\" folder for the output images.")

        elif ch == '3':
            image_name_input = input("Enter the name of the equirectangular image in the input_folder: ")
            image_name = utils.find_filename_match(image_name_input, config.INPUT_FOLDER)
            if image_name != image_name_input:
                print("The image name was changed to: " + image_name.split(config.INPUT_FOLDER)[1])

            image.equi_perspective(image_name)

            print(f"Done! Please check the \"{config.E2P_OUTPUT_FOLDER.split(config.PROJECT_NAME)[1][1:]}\" folder for the output image.")

        elif ch == '4':
            print("Make sure the images of multiple perspectives are in input_folder/stitch_images")
            image.pers_equi()

            print(f"Done! Please check the \"{config.P2E_OUTPUT_FOLDER.split(config.PROJECT_NAME)[1][1:]}\" folder for the output image.")

        elif ch == '5':
            image_name_input = input("Enter the name of the equirectangular image in the input_folder: ")
            image_name = utils.find_filename_match(image_name_input, config.INPUT_FOLDER)
            if image_name != image_name_input:
                print("The image name was changed to: " + image_name.split(config.INPUT_FOLDER)[1])

            image.detect_border(image_name)

            print(f"Done! Please check the \"{config.BORDER_DETECT_OUTPUT_FOLDER.split(config.PROJECT_NAME)[1][1:]}\" folder for the output images.")

        elif ch == '6':
            input("Place the image in the corresponding input folders of \"input/" + config.TILE_COLOUR_CHANGE_TARGET_INPUT_FOLDER.split(os.sep)[-2] + "\" and press enter")
            change_tile_colour.target_tile_to_base_tile()

            print(f"Done! Please check the \"{config.SIMPLE_OUTPUT_FOLDER.split(config.PROJECT_NAME)[1][1:]}\" folder for the output image.")

    # Point cloud tools
    elif ch == '2':
        print("Initializing point cloud toolkit...\n\n")

        import source.point_cloud_toolkit.point_cloud as pt_cloud

        ch = input("1) Point cloud coordinate extract and combine\n")
        if ch == '1':
            ch = input("Do you want to confirm matches when combining point clouds? (y/n) [default - n]: ")
            show_image = True if ch == "y" else False

            pt_cloud.load_screenshots_and_correct()
            pt_cloud.combine_corrected_point_clouds(show_image=show_image, show_final_image=True)

    # Segmentation toolkit
    elif ch == '3':
        print("Initializing segmentation toolkit...\n\n")

        import source.config as config
        import source.image as image
        import source.utils as utils
        import source.segmentation_tools.train as train
        import source.segmentation_tools.data_handling as data_handling
        import source.segmentation_tools.predict as predict

        ch = input(
            f"1) Detect 360 image in '{config.INPUT_FOLDER.split(os.sep)[-2]}' folder and stitch them\n"
            f"2) Detect all 360 images in '{config.INPUT_FOLDER.split(os.sep)[-2]}/{config.FLOOR_DETECT_360_INPUT_FOLDER.split(os.sep)[-2]}' folder\n"
            f"3) Detect all normal images in '{config.FLOOR_DETECT_INPUT_FOLDER.split(os.sep)[-2]}' folder\n"
            "4) Additional options\n"
            )
        if ch == '1':
            image_name_input = input("Enter the name of the equirectangular image in the input_folder: ")
            image_name = utils.find_filename_match(image_name_input, config.INPUT_FOLDER)
            if image_name != image_name_input:
                print("The image name was changed to: " + image_name.split(config.INPUT_FOLDER)[1])

            include_roof = 'y'
            image.extract_and_predict(
                input_path=image_name,
                predict=True,
                skip_roof='y'
            )
        elif ch == "2":
            image.predict_all_360_images()
        elif ch == '3':
            predict.run_images()
        elif ch == '4':
            ch = input("1) New model using random initialisation\n"
                       "2) Continue training from last saved checkpoint\n"
                       "3) Fine tune model from pre-trained weights\n")
            if ch == '1':
                train.train(continue_training=False, load_weights_for_fine_tune=False)
            elif ch == '2':
                train.train(continue_training=True, load_weights_for_fine_tune=False)
            elif ch == '3':
                train.train(continue_training=False, load_weights_for_fine_tune=True)
