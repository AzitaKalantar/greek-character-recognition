# greek-character-recognition
Ancient Greek characters have more than 200 classes of characters and that makes it harder to achieve good accuracy on its OCR problem. In this project I was dealing with a large set of labelled training data which was highly unbalanced; More than 25% of classes had only 4 or less number of sample characters while there were classes with more than 10,000 samples. Considering the distribution shift problem and reducing training size I had to do sampling from each class very carefully. I managed to do so and achieve 97% accuracy on test data using only 4% of raw data. The models that I used for training were KNN, SVM and CNN. I achieved my best accuracy using fine-tuning a Resnet model in Pytorch. In the end, I built a GUI to show the performance of my work better.

![alt text](https://github.com/AzitaKalantar/greek-character-recognition/blob/master/gui.png)
