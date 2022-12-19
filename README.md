- remove null gender/age labels
- make sure ages in range (0,100), (some were negative and one was 200ish)
- face score of above 0.5
- class weighting calculated by # W_i = n_samples / (n_classes * n_samples_class_i)
- per pixel mean subtracted from each image in dataset
- rescales to 244x244

- multi task learning
- transfer learning
- bucketed age
- multi-label classification probllem for age due to ordinal suff
- resnet param freezing
- say num of trainable params before and after model freezing



- resnet-50
- 2048 -> 1024
- 1 epoch 
- ct <6
- overfit after first epoch but second best accuracy

resnet-18
512 -> 258
ct<7
test overnight, no overfit for first 2 epochs

wide_resnet50_2
1000 -> 500
ct<6
best accuracy, overfit after 1st epoch

halved batch size 