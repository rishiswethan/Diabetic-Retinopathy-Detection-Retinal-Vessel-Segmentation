import source.train as train

print("1) Train the model\n"
      "2) Tune hyperparameters\n"
      "3) Visualise the model on val set\n")
choice = input("Enter your choice: ")

if choice == '1':
    ch = input("Continue training from last best training epoch? (y/n): ")
    if ch.lower() == 'y' or ch == '1':
        # train the model using the last saved model as pre-trained weights
        train.train_using_best_hp(continue_training=True)
    else:
        # train the model using random/default initialisation
        train.train_using_best_hp(continue_training=False)

elif choice == '2':
    # tune the hyperparameters
    ch = input("1) Continue tuning\n"
               "2) Start tuning from scratch\n")
    train.hyper_parameter_optimise(load_if_exists=True if ch == '1' else False)

elif choice == '3':
    # visualise the model on val set, using the stored model
    train.visualise_generator(data_loader='val')
