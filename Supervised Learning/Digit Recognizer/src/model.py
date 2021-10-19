from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

digits = load_digits()

# plt.gray() 
# plt.matshow(digits.images[0]) 
# plt.show()

# Preparing data
img_length = len(digits.images)
data = digits.images.reshape((img_length, -1))

# Splitting Data into Train and Test
x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False)

from sklearn import svm
svm_classifier = svm.SVC(gamma=0.001)
svm_classifier.fit(x_train, y_train)
predicted = svm_classifier.predict(x_test)

_, axes = plt.subplots(1, 4)
images_and_predictions = list(zip(digits.images[img_length // 2:], predicted))
for ax, (image, prediction) in zip(axes, images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

print("\nClassification report for classifier %s:\n%s\n" % (svm_classifier, metrics.classification_report(y_test, predicted)))

disp = metrics.plot_confusion_matrix(svm_classifier, x_test, y_test)
disp.figure_.suptitle("Confusion Matrix")

# print("\nConfusion matrix:\n%s" % disp.confusion_matrix)
print("\nAccuracy of the Algorithm: ", svm_classifier.score(x_test, y_test))
# plt.show()

print(x_test.shape)
print(x_test[1:3])