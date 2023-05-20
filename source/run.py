import train

print("1) Train the model\n"
      "2) Tune hyperparameters\n"
      "3) Visualise the model on val set\n")
choice = input("Enter your choice: ")

if choice == '1':
    # train the model
    train.train_using_best_hp()

elif choice == '2':
    # tune the hyperparameters
    ch = input("1) Continue tuning\n"
               "2) Start tuning from scratch\n")
    train.hyper_parameter_optimise(load_if_exists=True if ch == '1' else False)

elif choice == '3':
    # visualise the model on val set, using the stored model
    train.visualise_generator(data_loader='val')
