import os

def genScript():
    leaveDatset = [i for i in range(8)] #Replace this with names of the datasets you wish to train, validate, test the model on.
    filePath = "./batchTraining.sh"
    max_epoch = 100
    val_size = 1 #Change this to implement validation too


    with open(filePath, 'w') as f:
        for ld in leaveDatset[:]:#training, testing seq = [0, 1, 2,... 6](train) and [7](test), [0, 1 ..5, 7](train) and [6](test) .. [1, .. 7](train) and [0](test)
            f.write(f"\necho '------------NEW BATCH TRAINING AND TESTING STARTS---------'")
            lis = leaveDatset[:]
            lis.remove(ld)#dataset sequence for training
            for ind in range(len(lis)):
                name = f"_test_dataset_{ld}_train_on_{lis[ind]}_" #Unique names for each training
                if ind == 0:
                    #training code
                    f.write(f"\npython train_individualTF.py --max_epoch {max_epoch} --dataset_name {lis[ind]}_no_val_ --name {name}"
                            f" --val_size {val_size} --verbose")
                else:
                    #create new dir, if doesn't exists
                    #copy final trained file from prev trained model
                    #train on new dataset, resuming from the last epoch
                    newFilePath = os.path.join(f"./models/individual/{name}/", f"{(max_epoch - 1):05d}.pth")
                    f.write(f"\nmkdir {os.path.dirname(newFilePath)}")
                    f.write(f"\ncp ./models/individual/{lastName}/{(max_epoch - 1):05d}.pth {newFilePath}")
                    f.write(f"\npython train_individualTF.py --max_epoch {max_epoch} --dataset_name {lis[ind]}_no_val_ --name {name}"
                        f" --val_size {val_size} --verbose --resume_train --model_pth {(max_epoch - 1):05d}.pth")
                lastName = name

            # take final saved model
            # test the model on testing data('ld' dataset)
            name = f"_testing_on_{ld}_"
            newFilePath = os.path.join(f"./models/individual/{name}/", f"{(max_epoch - 1):05d}.pth")
            f.write(f"\nmkdir {os.path.dirname(newFilePath)}")
            f.write(f"\ncp ./models/individual/{lastName}/{(max_epoch - 1):05d}.pth {newFilePath}")
            f.write(f"\npython train_individualTF.py --max_epoch 1 --dataset_name {ld}_no_val_ --name {name}"
                f" --val_size {val_size} --verbose --resume_train --model_pth {(max_epoch - 1):05d}.pth --evaluate True")

            f.write(f"\n\n#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------\n\n")


if __name__ == '__main__':
    genScript()
