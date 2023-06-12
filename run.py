import os

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

if __name__ == "__main__":
    ch = input("1) Classify DR\n"
               "2) Segment Vessel\n")

    if ch == '1':
        import source_classify.train as train
        import source_classify.predict as predict

        print("1) Train the model\n"
              "2) Tune hyperparameters\n"
              "3) Visualise the model on val set\n"
              "4) Predict on custom image(s)\n")
        choice = input("Enter your choice: ")

        if choice == '1':
            # ch = input("Continue training from last best training epoch? (y/n): ")
            ch = input(
                "1) Continue training from last best training epoch\n"
                "2) Fine tune from a pre-trained model\n"
                "3) Start training from scratch\n"
                "Enter your choice: "
            )
            if ch == '1':
                # train the model using the last saved model as pre-trained weights
                train.train_using_best_hp(continue_training=True)
            elif ch == '2':
                # train the model by fine-tuning from a pre-trained model in the given path
                ft_path = input("Enter the full path to the pre-trained model: ")

                train.train_using_best_hp(continue_training=False, fine_tune=ft_path)
            else:
                # train the model using random/default initialisation
                train.train_using_best_hp(continue_training=False)

        elif choice == '2':
            # tune the hyperparameters
            ch = input("1) Continue tuning\n"
                       "2) Start tuning from scratch\n"
                       "Enter your choice: ")
            train.hyper_parameter_optimise(load_if_exists=True if ch == '1' else False)

        elif choice == '3':
            # visualise the model on val set, using the stored model
            train.visualise_generator(data_loader='val')

        elif choice == '4':
            # predict on custom image
            path = input("Enter the full path to a single image, or a folder containing images: ")

            pred_class = predict.Predict(verbose=True)
            print(pred_class.predict(path))

    elif ch == '2':

        import source_segment.segmentation_tools.train as train
        import source_segment.segmentation_tools.predict as predict
        import source_segment.segmentation_tools.data_handling as data_handling

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
