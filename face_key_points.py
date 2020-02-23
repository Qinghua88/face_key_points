import os
import kmodel
from utils import load_data
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load my_model which is trained already
my_model = kmodel.load_trained_model('my_model')
X_test, y_test = load_data(test=True)
n = 4
image_test = X_test[0nn]
keypoints_test = my_model.predict(image_test)
# convert predicted keypoints from [-1, 1] to [0, 96]
keypoints_test = keypoints_test  48 + 48
fig, a =  plt.subplots(n, n, figsize=(16, 16))
for i in range(n)
    for j in range(n)
        keypoints = keypoints_test[in + j]
        image = image_test[in + j]
        x = []
        y = []
        for m in range(len(keypoints)  2)
            x.append(keypoints[2m])
            y.append((keypoints[2m + 1]))

        image = image.reshape((96, 96,))
        a[i][j].imshow(image)
        a[i][j].scatter(x, y, color='r')

plt.savefig('test_result.png')
plt.show()