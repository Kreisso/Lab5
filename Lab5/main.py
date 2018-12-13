import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
import seaborn as sns;
import pickle

label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}

from keras.preprocessing import image
image_file = 'test-img/test-2.jpg'
img = image.load_img(image_file, target_size=(28, 28), color_mode="grayscale")
x = image.img_to_array(img)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

images_train =  []
for image_train in x_train:
    images_train.append(image_train.flatten())

images_test = []

for image_test in x_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

# fashion_mnist_classifier = OneVsRestClassifier(LogisticRegression(verbose=1, max_iter=100))
#
# fashion_mnist_classifier.fit(images_train, y_train);
#
# pickle.dump(fashion_mnist_classifier, open('fashion_mnist_classifier.model', 'wb'));
#
# multi_class_fashion_mnist_classifier = LogisticRegression(verbose=1, max_iter=100, multi_class="multinomial", solver="sag")
#
# multi_class_fashion_mnist_classifier.fit(images_train, y_train)
#
# conf_matrix = confusion_matrix(y_test, multi_class_fashion_mnist_classifier.predict(images_test))
# print("Confusion_matrix:")
# print(conf_matrix)
# sns.heatmap(conf_matrix)
#
#
# print('------------Zapis modeli')
# pickle.dump(multi_class_fashion_mnist_classifier, open('multi_class_fashion_mnist_classifier.model', 'wb'));

fashion_mnist_classifier = pickle.load(open('fashion_mnist_classifier.model', 'rb'))
multi_class_fashion_mnist_classifier_file = pickle.load(open('multi_class_fashion_mnist_classifier.model', 'rb'))

# multi_class_fashion_mnist_classifier_file.fit(images_train, y_train)

conf_matrix = confusion_matrix(y_test, fashion_mnist_classifier.predict(images_test))
print("--------Confusion_matrix for Binary logistic classifier and OvR strategy: \n")
print(conf_matrix)
print()
sns.heatmap(conf_matrix)

conf_matrix = confusion_matrix(y_test, multi_class_fashion_mnist_classifier_file.predict(images_test))
print("--------Confusion_matrix for Single logistic classifier by learning probability distribution across multiple classes: \n")
print(conf_matrix)
print()
sns.heatmap(conf_matrix)

result = fashion_mnist_classifier.predict(x.flatten().reshape(1,-1))[0]
print('--------Result for Binary logistic classifier and OvR strategy:')
print(label_dict[result]);
print()

result = multi_class_fashion_mnist_classifier_file.predict(x.flatten().reshape(1,-1))[0]
print('--------Result for Single logistic classifier by learning probability distribution across multiple classes:')
print(label_dict[result]);
print()