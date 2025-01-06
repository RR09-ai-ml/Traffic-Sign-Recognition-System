# The code appears to be an implementation of a Traffic Sign Recognition System using PyQt5 for the GUI and Keras for the machine learning part.

from PyQt5 import QtCore, QtGui, QtWidgets
# PyQt5: Used for creating the graphical user interface (GUI) of the application. 
# QtCore and QtGui help with core functionalities and GUI elements, while QtWidgets is for widget-based UI components like buttons, labels, etc.

from keras.models import Sequential, load_model
# Keras: The high-level neural network API that simplifies building deep learning models. 
# The Sequential model is used to stack layers of the network. load_model is used to load pre-trained models.

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
# Conv2D, MaxPool2D, Dense, Flatten, Dropout: Different layers for creating Convolutional Neural Networks (CNNs) in Keras.

from keras.utils import to_categorical
# to_categorical: Converts the target labels into a one-hot encoded format.

from sklearn.model_selection import train_test_split
# train_test_split: From Scikit-learn, used to split data into training and testing sets.

import numpy as np
# NumPy: For numerical operations, such as manipulating image arrays.

import matplotlib.pyplot as plt
# Matplotlib: Used for plotting graphs to visualize model performance (accuracy and loss).

from PIL import Image
# PIL (Pillow): To open and manipulate images.

import os
# os: To interact with the operating system (e.g., fetching files and directories).


data = []
# data: A list to hold the image data.

labels = []
# labels: A list to hold corresponding labels (i.e., traffic sign classes).

classes = 43
# classes: The number of traffic sign classes in the dataset (43 classes).

cur_path = os.getcwd() 
# cur_path: The current working directory from where the script is being run. 


# Mapping of Class Labels to Text Descriptions:
classs = {  1:"Speed limit (20km/h)",
            2:"Speed limit (30km/h)",
            3:"Speed limit (50km/h)",
            4:"Speed limit (60km/h)",
            5:"Speed limit (70km/h)",
            6:"Speed limit (80km/h)",
            7:"End of speed limit (80km/h)",
            8:"Speed limit (100km/h)",
            9:"Speed limit (120km/h)",
            10:"No passing",
            11:"No passing veh over 3.5 tons",
            12:"Right-of-way at intersection",
            13:"Priority road",
            14:"Yield",
            15:"Stop",
            16:"No vehicles",
            17:"Veh > 3.5 tons prohibited",
            18:"No entry",
            19:"General caution",
            20:"Dangerous curve left",
            21:"Dangerous curve right",
            22:"Double curve",
            23:"Bumpy road",
            24:"Slippery road",
            25:"Road narrows on the right",
            26:"Road work",
            27:"Traffic signals",
            28:"Pedestrians",
            29:"Children crossing",
            30:"Bicycles crossing",
            31:"Beware of ice/snow",
            32:"Wild animals crossing",
            33:"End speed + passing limits",
            34:"Turn right ahead",
            35:"Turn left ahead",
            36:"Ahead only",
            37:"Go straight or right",
            38:"Go straight or left",
            39:"Keep right",
            40:"Keep left",
            41:"Roundabout mandatory",
            42:"End of no passing",
            43:"End no passing veh > 3.5 tons" }
# classs: A dictionary that maps class numbers to the respective traffic sign labels.


# Loading Images and Labels:
print("Obtaining Images & its Labels..............")
for i in range(classes):
    path = os.path.join(cur_path,'dataset/train/',str(i))   # Path to each class folder
    images = os.listdir(path)                               # List of images in the class folder    

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)              # Open the image
            image = image.resize((30,30))                   # Resize the image to 30x30 pixels
            image = np.array(image)                         # Convert image to NumPy array
            data.append(image)                              # Append image to data
            labels.append(i)                                # Append class label to labels
            print("{0} Loaded".format(a))                   # Print loaded image name
        except:
            print("Error loading image")
print("Dataset Loaded")
# This part loops through each class folder (from 0 to 42) and loads the images.
# The images are resized to 30x30 pixels (a common size for image classification).
# The image data is stored in the data list and the class labels in the labels list.

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
# After all images and labels are loaded, they are converted to NumPy arrays for further processing.
print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# Splits the data into training and testing sets (80% training, 20% testing).
# random_state=42 ensures reproducibility of the split.
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
# Converts the labels into one-hot encoded format. This is required for multi-class classification.


# GUI (PyQt5) Setup:
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # Ui_MainWindow: The main class that sets up the graphical user interface.
        # setupUi(): Initializes and sets up the UI components like buttons, labels, and text fields.

        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        # BrowseImage: A button to open a file dialog for selecting an image.
        # The geometry sets the button's position and size in the window.

        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        # imageLbl: A label where the selected image will be displayed after being loaded.

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        # label_2: A label displaying the title "ROAD SIGN RECOGNITION" at the top of the window.

        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")
        # Classify: A button that triggers the classification process.

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")

        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setObjectName("Training")
        # Training: A button that triggers the training process.

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        self.textEdit.setObjectName("textEdit")
        # textEdit: A text area where the classification result will be displayed after the image is classified.
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)  # Function call

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # Automatically connects signals to their corresponding slots.

        # Event Handling (Button Clicks):
        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)        
        # Connects button clicks to the corresponding functions (loadImage, classifyFunction, trainingFunction).

    # Retranslate and Connecting Events:
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "           ROAD SIGN RECOGNITION"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))
    # retranslateUi: Sets the text for the GUI components.
    # This method ensures the UI can be translated based on the locale if necessary. 

    # Image Loading:
    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: # If the user gives a file
            print(fileName)
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.imageLbl.setPixmap(pixmap) # Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center
    # loadImage(): Opens a file dialog to allow the user to choose an image. The image is displayed in the imageLbl label after resizing.


    # This function is called when the "Classify" button is clicked, and its purpose is to classify the image selected by the user using a pre-trained neural network model.
    def classifyFunction(self):
        model = load_model('my_model.h5')                       # Load the pre-trained Keras model
        print("Loaded model from disk")

        path2 = self.file                                       # The path of the image to classify (set earlier in loadImage)
        print(path2)                                            # Prints the path of the selected image to the console for debugging.

        test_image = Image.open(path2)                          # This opens the selected image file using the Pillow library.
        test_image = test_image.resize((30, 30))                # The image is resized to 30x30 pixels to match the input size expected by the neural network (since the model was trained on images of this size).
        test_image = np.expand_dims(test_image, axis=0)         # This adds an extra dimension to the image array to represent the batch size. Keras models expect a batch of images, not just a single image. The image must have the shape (1, 30, 30, 3) where 1 is the batch size, 30x30 is the image size, and 3 is the number of color channels (RGB).
        test_image = np.array(test_image)                       # Converts the image into a NumPy array so that it can be passed to the neural network for prediction.

        # result = model.predict_classes(test_image)[0]           # Predict the class label of the image
        # edited 4 lines
        # Assuming 'test_image' is a 2D array (e.g., image data), we use `predict()` to get class probabilities.
        predictions = model.predict(test_image)

        # Then, to get the class with the highest probability, use argmax
        result = np.argmax(predictions, axis=-1)[0]
        
        sign = classs[result + 1]                               # Map the predicted label to its corresponding class name (classs is the dictionary)
        print(sign)                                             # Print the predicted class name
        self.textEdit.setText(sign)                             # Display the predicted class name in the textEdit field


    # The trainingFunction is a method within the Ui_MainWindow class that is executed when the "Training" button is clicked in the GUI. The purpose of this function is to train a Convolutional Neural Network (CNN) on the traffic sign dataset and save the trained model to a file (my_model.h5). The function also plots the accuracy and loss curves during training and saves those plots as images (Accuracy.png and Loss.png).
    def trainingFunction(self):
        self.textEdit.setText("Training under process...") 
        # When the training starts, this line updates the textEdit widget in the GUI to display the message "Training under process..." to inform the user that the training has begun.
        
        model = Sequential()
        # model = Sequential(): This initializes a Sequential model. The Sequential model is a linear stack of layers, meaning each layer has exactly one input and one output.

        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
        # This is a 2D convolutional layer with 32 filters (or kernels), each of size 5x5.
        # input_shape=X_train.shape[1:]: This specifies the input shape of the images (height, width, channels). The input size comes from the training dataset X_train, which has shape (num_samples, height, width, channels). We slice it to get (height, width, channels) as the input shape for the model.
        # activation='relu': The activation function used is ReLU (Rectified Linear Unit), which introduces non-linearity and helps the model learn complex patterns.

        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        # This is another convolutional layer with the same configuration as the first one but no input_shape argument, as the model already knows the input shape from the previous layer.

        model.add(MaxPool2D(pool_size=(2, 2)))
        # MaxPool2D(pool_size=(2, 2)): A max-pooling layer that reduces the spatial dimensions (height and width) of the input. It takes a 2x2 window and outputs the maximum value in each window, reducing the image's size by a factor of 2 in both the height and width. Pooling helps reduce computational cost and prevents overfitting by extracting the most important features.

        model.add(Dropout(rate=0.25))
        # Dropout(rate=0.25): A dropout layer is added to randomly "drop" 25% of the neurons during training. This helps to prevent overfitting, a common problem where the model becomes too specific to the training data and performs poorly on unseen data.

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        # Two more convolutional layers are added with 64 filters of size 3x3. These layers continue to extract features from the image. With each additional layer, the model learns more complex patterns and representations.

        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        # Another max-pooling layer is added, followed by a dropout layer to continue reducing spatial dimensions and combat overfitting.

        model.add(Flatten())
        # Flatten(): This layer flattens the 3D output from the previous layers into a 1D vector. This is necessary because the output from the convolutional and pooling layers needs to be converted into a format that can be fed into the fully connected (dense) layers.

        model.add(Dense(256, activation='relu'))
        # Dense(256, activation='relu'): This is a fully connected layer (also called a dense layer) with 256 neurons. Each neuron is connected to every neuron in the previous layer. The ReLU activation function is used to introduce non-linearity.

        model.add(Dropout(rate=0.5))
        # Another dropout layer is added with a dropout rate of 0.5 (50%). This helps regularize the model further, preventing it from overfitting.

        model.add(Dense(43, activation='softmax'))
        # Dense(43, activation='softmax'): This is the final output layer of the model. It consists of 43 neurons, corresponding to the 43 classes in the traffic sign dataset. Softmax activation is used here because it's a multi-class classification problem, and softmax ensures that the output values are probabilities (i.e., the sum of the output values is 1, each representing the probability of each class).
        print("Initialized model")


        # Compilation of the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # compile(): The model is compiled with the following settings:

        # Loss function: categorical_crossentropy is used because it's a multi-class classification problem where each label is one of 43 classes.
        # Optimizer: adam is a popular optimization algorithm that adapts the learning rate during training to speed up convergence.
        # Metrics: The model's performance will be evaluated using accuracy.


        history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
        # fit(): This method starts the training of the model on the X_train data and y_train labels.

        # batch_size=32: The model will process 32 samples at a time before updating its weights.
        # epochs=5: The model will be trained for 5 epochs (iterations over the entire training dataset).
        # validation_data=(X_test, y_test): The model will evaluate its performance on the X_test and y_test data after each epoch. This helps monitor overfitting.


        model.save("my_model.h5")  # save("my_model.h5"): After training is complete, the trained model is saved to a file named my_model.h5. This file can later be loaded to make predictions without retraining.

        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('Accuracy.png')

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('Loss.png')
        # Accuracy and Loss plots: The training process records accuracy and loss values at each epoch, which are stored in the history object.

        # plt.plot(): Plots training and validation accuracy/loss over epochs.
        # plt.savefig(): Saves the plotted graphs to files Accuracy.png and Loss.png.


        self.textEdit.setText("Saved Model & Graph to disk")
        # Once training is complete, the textEdit widget is updated with the message "Saved Model & Graph to disk", indicating to the user that the model and its training graphs have been saved.
        
        
        
        
if __name__ == "__main__":  # Main Program: This is the standard entry point for a PyQt application.
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # QApplication is created to initialize the application.

    MainWindow = QtWidgets.QMainWindow()
    # The main window is created and set up using Ui_MainWindow.

    ui = Ui_MainWindow()
    # MainWindow.show() displays the GUI window.

    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    # sys.exit(app.exec_()) enters the main event loop to keep the GUI responsive until the user closes it.

    


