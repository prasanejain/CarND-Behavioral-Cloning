import cv2
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg



csvPaths = ['./trainingData/driving_log.csv', './trainingData2/driving_log.csv' ,'./trainingData3/driving_log.csv']
extrapath='/home/workspace/CarND-Behavioral-Cloning-P3/trainingData3/'

# Function to extract iamge paths and angles
def dataextractor(csvPaths):
    lines = []
    angles = []
    imagesPath = []
    for ik,csvPath in enumerate(csvPaths):
        lines = []
        if ik==2:
            extrapath='/home/workspace/CarND-Behavioral-Cloning-P3/trainingData3/'
        else:
            extrapath=''
                
        with open(csvPath) as csvFile:
            lines = list(csv.reader(csvFile,skipinitialspace=True,delimiter=',',quoting=csv.QUOTE_NONE))
            for line in lines[1:]:
                
                if float(line[6]) < 0.1:
                    continue
                imagesPath.append(extrapath+line[0])
                angles.append(float(line[3]))
                imagesPath.append(extrapath+line[1])
                angles.append(float(line[3]) + 0.2)
                imagesPath.append(extrapath+line[2])
                angles.append(float(line[3]) - 0.2)

    imagesPath = np.array(imagesPath)
    angles = np.array(angles)

    return (imagesPath, angles)

imagesPath,angles=dataextractor(csvPaths)
print(angles.shape)
print(imagesPath.shape)

# Function to generate and save histogram
def histofigGen(angles,fname):
    num_bins = 25*10
    
    fig, ax0 = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.25)
    #ax0= axes.flatten()

    ax0.hist(angles, bins=num_bins, histtype='bar', color='blue', rwidth=0.6, label='train')
    ax0.set_title('Number of training')
    ax0.set_xlabel('Steering Angle')
    ax0.set_ylabel('Total Image')

    fig.tight_layout()
    plt.savefig(fname)

#print(hist)
#print(bins)
histofigGen(angles,fname='allangles.png')

# Function for resizing the input images
def resize(img):
    #new_img = cv2.resize(img,(120, 60), interpolation = cv2.INTER_AREA)
    #return new_img
    import tensorflow
    return tensorflow.image.resize_images(img, (60, 120))

# Function to apply color space transformation
def change_color_space(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image.shape)
    #image=image[None,:,:,:]
    return image

# Generator Function to output training images and labels of given batch size
def generator(angles, imagesPath, batch_size=32):
    path = imagesPath 
    lines = angles
    sum_lines = len(lines)
    batch_number=1
    path,lines=shuffle(path,lines)
    imagesOut = []
    anglesOut = []
    highImgCounter=0
    while 1: 
        for offset in range(0, sum_lines, batch_size):
            #print(offset)
            batch_samples = path[offset:offset+batch_size]
            angleSamples=lines[offset:offset+batch_size]

            for batch_sample,angleSample in zip(batch_samples,angleSamples):
                name=batch_sample
                #image1 = cv2.imread(name)
                angle1=angleSample
                #image1=change_color_space(image1)
                # Collect only <X>% of samples in range -0.2 <= 0 <= 0.2, the value is set to 30% of batch size
                if ( ( abs(angle1)<=0.2 ) and ( highImgCounter<((batch_size*30)//100) ) ) :
                    image1 = cv2.imread(name)
                    image1=change_color_space(image1)
                    imagesOut.append(image1)
                    anglesOut.append(angle1)
                    highImgCounter+=1
                    #print(highImgCounter)
                # If angle is outside abs(0.2) region use the same image and also the flip the same and append
                if (abs(angle1)>0.2) :
                   image1 = cv2.imread(name)
                   image1=change_color_space(image1)
                   imagesOut.append(image1)
                   anglesOut.append(angle1)
                   image1 = np.fliplr(image1)
                   angle1 *= -1
                   imagesOut.append(image1)
                   anglesOut.append(angle1)
                #print(len(anglesOut))
                if len(anglesOut)==batch_size:
                   X_train = np.array(imagesOut)
                   y_train = np.array(anglesOut)
                   #imagesOut = []
                   #anglesOut = []
                   #highImgCounter=0
                   #print('afs',X_train.shape)
                   yield shuffle(X_train, y_train)
                   imagesOut = []
                   anglesOut = []
                   highImgCounter=0
                   #path,lines=shuffle(path,lines)
                batch_number+=1
            path,lines=shuffle(path,lines)

# split the data in training and validation sets
X_train, X_test, y_train, y_test = train_test_split(imagesPath,angles, shuffle=True, test_size=0.2)
X_train_s, y_train_s = shuffle(X_train, y_train)
print(len(X_train_s))
#print(X_train_s[0:10])
X_valid_s, y_valid_s= shuffle(X_test, y_test)
print(len(X_valid_s))

# Generator initialisation for training data and validation data
train_generator = generator(y_train_s, X_train_s, batch_size=128)
validation_generator = generator(y_valid_s, X_valid_s, batch_size=128)

# generate some sample data from generator functions
sampleX_train,sampley_train= (next(generator(y_train_s, X_train_s, batch_size=256)))
histofigGen(sampley_train,fname='GenBatch1.png')        
print('GenBatch1')
sampleX_train,sampley_train= (next(generator(y_train_s, X_train_s, batch_size=256)))
print('GenBatch2')
histofigGen(sampley_train,fname='GenBatch2.png')    
sampleX_train,sampley_train= (next(generator(y_train_s, X_train_s, batch_size=256)))
histofigGen(sampley_train,fname='GenBatch3.png')  
print('GenBatch3')


# flag to avoid network init and training
nwFlag=True
if nwFlag:
    
    import tensorflow

    from keras.models import Sequential,load_model
    from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D 
    from keras.layers import Lambda, Cropping2D, Dropout, ELU
    from keras.layers import BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    model = Sequential()

    #model.add(Lambda(change_color_space),input_shape=(160, 320, 3))
    # Crop 70 pixels from the top of the image and 25 from the bottom
    model.add(Cropping2D(cropping=((75, 25), (0, 0)),
                     input_shape=(160, 320, 3),
                     data_format="channels_last"))

    # Resize the data
    model.add(Lambda(resize))
    #model.add(Lambda(change_color_space))
    # Normalize the data
    model.add(Lambda(lambda x: (x/127.5) - 0.5))

    model.add(Conv2D(3, (1, 1), padding='same'))
    model.add(ELU())

    model.add(BatchNormalization())
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())

    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())

    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(ELU())

    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(ELU())

    model.add(Flatten())
    model.add(ELU())

    model.add(Dense(512))
    model.add(Dropout(.2))
    model.add(ELU())

    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(ELU())

    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(ELU())

    model.add(Dense(1))

    adam = Adam(lr=1e-5)
    model.compile(optimizer= adam, loss="mse", metrics=['accuracy'])
    model.summary()
    
    
    file = 'model_generatorRGB.h5'
    
    # Flag to specify if new network has to be trained or reuse the old network
    LoadOldModel=False
    if not(LoadOldModel):
        model.fit_generator(train_generator, steps_per_epoch= (len(X_train_s)//128),validation_data=validation_generator,
                            validation_steps=(len(X_valid_s)//128), epochs = 10, verbose=1)
        model.save(file)
    else:
        # load the model
        print('Loading old model', file)
        new_model = load_model(file)
        from keras import backend as K
        # To get learning rate
        print(K.get_value(new_model.optimizer.lr))
        # To set learning rate
        K.set_value(new_model.optimizer.lr, 0.001)
        new_model.fit_generator(train_generator, steps_per_epoch= (len(X_train_s)//128),validation_data=validation_generator,
                            validation_steps=(len(X_valid_s)//128), epochs = 5, verbose=1)
        new_model.save(file)