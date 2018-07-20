

import cv2
import tensorflow as tf

def kerasTest():
    photoPath = '/Users/kyoka/Documents/open_source_data/imagenet_6607/imagenet/n01694178/n01694178_83.JPEG'
    img = cv2.imread(photoPath)
    img = cv2.resize(img, (224,224))
    print(img.shape)
    pretrained = tf.keras.applications.mobilenet.MobileNet()
    print(pretrained.predict(img))

if __name__ == '__main__':
    kerasTest()
