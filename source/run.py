import train

print("1) Train the model\n"
      "2) Visualise the model on val set\n")
choice = input("Enter your choice: ")

if choice == '1':
    # train the model
    best_hp_dict = {
        'batch_size': 8,
        'conv_model': 'vit',
    }
    train.train(best_hp_dict)

elif choice == '2':
    best_hp_dict = {
        'batch_size': 8,
        'conv_model': 'vit',
    }
    train.visualise_generator(hp_dict=best_hp_dict, data_loader='val')
