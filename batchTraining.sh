
echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 1_no_val_ --name _test_dataset_0_train_on_1_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_0_train_on_2_
cp ./models/individual/_test_dataset_0_train_on_1_/00099.pth ./models/individual/_test_dataset_0_train_on_2_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 2_no_val_ --name _test_dataset_0_train_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_0_train_on_3_
cp ./models/individual/_test_dataset_0_train_on_2_/00099.pth ./models/individual/_test_dataset_0_train_on_3_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 3_no_val_ --name _test_dataset_0_train_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_0_train_on_4_
cp ./models/individual/_test_dataset_0_train_on_3_/00099.pth ./models/individual/_test_dataset_0_train_on_4_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 4_no_val_ --name _test_dataset_0_train_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_0_train_on_5_
cp ./models/individual/_test_dataset_0_train_on_4_/00099.pth ./models/individual/_test_dataset_0_train_on_5_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 5_no_val_ --name _test_dataset_0_train_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_0_train_on_6_
cp ./models/individual/_test_dataset_0_train_on_5_/00099.pth ./models/individual/_test_dataset_0_train_on_6_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 6_no_val_ --name _test_dataset_0_train_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_0_train_on_7_
cp ./models/individual/_test_dataset_0_train_on_6_/00099.pth ./models/individual/_test_dataset_0_train_on_7_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 7_no_val_ --name _test_dataset_0_train_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_0_
cp ./models/individual/_test_dataset_0_train_on_7_/00099.pth ./models/individual/_testing_on_0_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 0_no_val_ --name _testing_on_0_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------


echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 0_no_val_ --name _test_dataset_1_train_on_0_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_1_train_on_2_
cp ./models/individual/_test_dataset_1_train_on_0_/00099.pth ./models/individual/_test_dataset_1_train_on_2_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 2_no_val_ --name _test_dataset_1_train_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_1_train_on_3_
cp ./models/individual/_test_dataset_1_train_on_2_/00099.pth ./models/individual/_test_dataset_1_train_on_3_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 3_no_val_ --name _test_dataset_1_train_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_1_train_on_4_
cp ./models/individual/_test_dataset_1_train_on_3_/00099.pth ./models/individual/_test_dataset_1_train_on_4_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 4_no_val_ --name _test_dataset_1_train_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_1_train_on_5_
cp ./models/individual/_test_dataset_1_train_on_4_/00099.pth ./models/individual/_test_dataset_1_train_on_5_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 5_no_val_ --name _test_dataset_1_train_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_1_train_on_6_
cp ./models/individual/_test_dataset_1_train_on_5_/00099.pth ./models/individual/_test_dataset_1_train_on_6_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 6_no_val_ --name _test_dataset_1_train_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_1_train_on_7_
cp ./models/individual/_test_dataset_1_train_on_6_/00099.pth ./models/individual/_test_dataset_1_train_on_7_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 7_no_val_ --name _test_dataset_1_train_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_1_
cp ./models/individual/_test_dataset_1_train_on_7_/00099.pth ./models/individual/_testing_on_1_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 1_no_val_ --name _testing_on_1_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------


echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 0_no_val_ --name _test_dataset_2_train_on_0_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_2_train_on_1_
cp ./models/individual/_test_dataset_2_train_on_0_/00099.pth ./models/individual/_test_dataset_2_train_on_1_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 1_no_val_ --name _test_dataset_2_train_on_1_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_2_train_on_3_
cp ./models/individual/_test_dataset_2_train_on_1_/00099.pth ./models/individual/_test_dataset_2_train_on_3_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 3_no_val_ --name _test_dataset_2_train_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_2_train_on_4_
cp ./models/individual/_test_dataset_2_train_on_3_/00099.pth ./models/individual/_test_dataset_2_train_on_4_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 4_no_val_ --name _test_dataset_2_train_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_2_train_on_5_
cp ./models/individual/_test_dataset_2_train_on_4_/00099.pth ./models/individual/_test_dataset_2_train_on_5_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 5_no_val_ --name _test_dataset_2_train_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_2_train_on_6_
cp ./models/individual/_test_dataset_2_train_on_5_/00099.pth ./models/individual/_test_dataset_2_train_on_6_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 6_no_val_ --name _test_dataset_2_train_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_2_train_on_7_
cp ./models/individual/_test_dataset_2_train_on_6_/00099.pth ./models/individual/_test_dataset_2_train_on_7_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 7_no_val_ --name _test_dataset_2_train_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_2_
cp ./models/individual/_test_dataset_2_train_on_7_/00099.pth ./models/individual/_testing_on_2_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 2_no_val_ --name _testing_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------


echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 0_no_val_ --name _test_dataset_3_train_on_0_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_3_train_on_1_
cp ./models/individual/_test_dataset_3_train_on_0_/00099.pth ./models/individual/_test_dataset_3_train_on_1_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 1_no_val_ --name _test_dataset_3_train_on_1_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_3_train_on_2_
cp ./models/individual/_test_dataset_3_train_on_1_/00099.pth ./models/individual/_test_dataset_3_train_on_2_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 2_no_val_ --name _test_dataset_3_train_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_3_train_on_4_
cp ./models/individual/_test_dataset_3_train_on_2_/00099.pth ./models/individual/_test_dataset_3_train_on_4_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 4_no_val_ --name _test_dataset_3_train_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_3_train_on_5_
cp ./models/individual/_test_dataset_3_train_on_4_/00099.pth ./models/individual/_test_dataset_3_train_on_5_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 5_no_val_ --name _test_dataset_3_train_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_3_train_on_6_
cp ./models/individual/_test_dataset_3_train_on_5_/00099.pth ./models/individual/_test_dataset_3_train_on_6_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 6_no_val_ --name _test_dataset_3_train_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_3_train_on_7_
cp ./models/individual/_test_dataset_3_train_on_6_/00099.pth ./models/individual/_test_dataset_3_train_on_7_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 7_no_val_ --name _test_dataset_3_train_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_3_
cp ./models/individual/_test_dataset_3_train_on_7_/00099.pth ./models/individual/_testing_on_3_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 3_no_val_ --name _testing_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------


echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 0_no_val_ --name _test_dataset_4_train_on_0_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_4_train_on_1_
cp ./models/individual/_test_dataset_4_train_on_0_/00099.pth ./models/individual/_test_dataset_4_train_on_1_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 1_no_val_ --name _test_dataset_4_train_on_1_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_4_train_on_2_
cp ./models/individual/_test_dataset_4_train_on_1_/00099.pth ./models/individual/_test_dataset_4_train_on_2_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 2_no_val_ --name _test_dataset_4_train_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_4_train_on_3_
cp ./models/individual/_test_dataset_4_train_on_2_/00099.pth ./models/individual/_test_dataset_4_train_on_3_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 3_no_val_ --name _test_dataset_4_train_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_4_train_on_5_
cp ./models/individual/_test_dataset_4_train_on_3_/00099.pth ./models/individual/_test_dataset_4_train_on_5_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 5_no_val_ --name _test_dataset_4_train_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_4_train_on_6_
cp ./models/individual/_test_dataset_4_train_on_5_/00099.pth ./models/individual/_test_dataset_4_train_on_6_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 6_no_val_ --name _test_dataset_4_train_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_4_train_on_7_
cp ./models/individual/_test_dataset_4_train_on_6_/00099.pth ./models/individual/_test_dataset_4_train_on_7_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 7_no_val_ --name _test_dataset_4_train_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_4_
cp ./models/individual/_test_dataset_4_train_on_7_/00099.pth ./models/individual/_testing_on_4_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 4_no_val_ --name _testing_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------


echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 0_no_val_ --name _test_dataset_5_train_on_0_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_5_train_on_1_
cp ./models/individual/_test_dataset_5_train_on_0_/00099.pth ./models/individual/_test_dataset_5_train_on_1_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 1_no_val_ --name _test_dataset_5_train_on_1_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_5_train_on_2_
cp ./models/individual/_test_dataset_5_train_on_1_/00099.pth ./models/individual/_test_dataset_5_train_on_2_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 2_no_val_ --name _test_dataset_5_train_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_5_train_on_3_
cp ./models/individual/_test_dataset_5_train_on_2_/00099.pth ./models/individual/_test_dataset_5_train_on_3_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 3_no_val_ --name _test_dataset_5_train_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_5_train_on_4_
cp ./models/individual/_test_dataset_5_train_on_3_/00099.pth ./models/individual/_test_dataset_5_train_on_4_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 4_no_val_ --name _test_dataset_5_train_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_5_train_on_6_
cp ./models/individual/_test_dataset_5_train_on_4_/00099.pth ./models/individual/_test_dataset_5_train_on_6_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 6_no_val_ --name _test_dataset_5_train_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_5_train_on_7_
cp ./models/individual/_test_dataset_5_train_on_6_/00099.pth ./models/individual/_test_dataset_5_train_on_7_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 7_no_val_ --name _test_dataset_5_train_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_5_
cp ./models/individual/_test_dataset_5_train_on_7_/00099.pth ./models/individual/_testing_on_5_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 5_no_val_ --name _testing_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------


echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 0_no_val_ --name _test_dataset_6_train_on_0_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_6_train_on_1_
cp ./models/individual/_test_dataset_6_train_on_0_/00099.pth ./models/individual/_test_dataset_6_train_on_1_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 1_no_val_ --name _test_dataset_6_train_on_1_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_6_train_on_2_
cp ./models/individual/_test_dataset_6_train_on_1_/00099.pth ./models/individual/_test_dataset_6_train_on_2_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 2_no_val_ --name _test_dataset_6_train_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_6_train_on_3_
cp ./models/individual/_test_dataset_6_train_on_2_/00099.pth ./models/individual/_test_dataset_6_train_on_3_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 3_no_val_ --name _test_dataset_6_train_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_6_train_on_4_
cp ./models/individual/_test_dataset_6_train_on_3_/00099.pth ./models/individual/_test_dataset_6_train_on_4_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 4_no_val_ --name _test_dataset_6_train_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_6_train_on_5_
cp ./models/individual/_test_dataset_6_train_on_4_/00099.pth ./models/individual/_test_dataset_6_train_on_5_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 5_no_val_ --name _test_dataset_6_train_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_6_train_on_7_
cp ./models/individual/_test_dataset_6_train_on_5_/00099.pth ./models/individual/_test_dataset_6_train_on_7_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 7_no_val_ --name _test_dataset_6_train_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_6_
cp ./models/individual/_test_dataset_6_train_on_7_/00099.pth ./models/individual/_testing_on_6_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 6_no_val_ --name _testing_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------


echo '------------NEW BATCH TRAINING AND TESTING STARTS---------'
python train_individualTF.py --max_epoch 100 --dataset_name 0_no_val_ --name _test_dataset_7_train_on_0_ --val_size 1 --verbose
mkdir ./models/individual/_test_dataset_7_train_on_1_
cp ./models/individual/_test_dataset_7_train_on_0_/00099.pth ./models/individual/_test_dataset_7_train_on_1_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 1_no_val_ --name _test_dataset_7_train_on_1_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_7_train_on_2_
cp ./models/individual/_test_dataset_7_train_on_1_/00099.pth ./models/individual/_test_dataset_7_train_on_2_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 2_no_val_ --name _test_dataset_7_train_on_2_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_7_train_on_3_
cp ./models/individual/_test_dataset_7_train_on_2_/00099.pth ./models/individual/_test_dataset_7_train_on_3_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 3_no_val_ --name _test_dataset_7_train_on_3_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_7_train_on_4_
cp ./models/individual/_test_dataset_7_train_on_3_/00099.pth ./models/individual/_test_dataset_7_train_on_4_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 4_no_val_ --name _test_dataset_7_train_on_4_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_7_train_on_5_
cp ./models/individual/_test_dataset_7_train_on_4_/00099.pth ./models/individual/_test_dataset_7_train_on_5_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 5_no_val_ --name _test_dataset_7_train_on_5_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_test_dataset_7_train_on_6_
cp ./models/individual/_test_dataset_7_train_on_5_/00099.pth ./models/individual/_test_dataset_7_train_on_6_/00099.pth
python train_individualTF.py --max_epoch 100 --dataset_name 6_no_val_ --name _test_dataset_7_train_on_6_ --val_size 1 --verbose --resume_train --model_pth 00099.pth
mkdir ./models/individual/_testing_on_7_
cp ./models/individual/_test_dataset_7_train_on_6_/00099.pth ./models/individual/_testing_on_7_/00099.pth
python train_individualTF.py --max_epoch 1 --dataset_name 7_no_val_ --name _testing_on_7_ --val_size 1 --verbose --resume_train --model_pth 00099.pth --evaluate True

#------------------NEW BATCH TRAINING AND TESTING STARTS-----------------------

