cd backend
kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip
rm -rf intel-image-classification.zip
mkdir datasets
mv seg_pred/seg_pred seg_test/seg_test seg_train/seg_train datasets
rmdir seg_train seg_pred seg_test