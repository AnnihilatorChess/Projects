 #  Image Classification Project
This project is an assignment for Programming in Python II. 
It uses a CNN model to classify 20 different image labels. 
The dataset consists of a collection of student's photos, which are 
converted to grayscale and resized to 100x100.
The target labels are items commonly used in daily life.

### Complete list of target labels:

book, bottle, car, cat, chair, computermouse, cup, dog , flower, 
fork, glass, glasses, headphones, knife, laptop, pen, plate, shoes,
spoon, tree


### Example usage
There are 2 use cases: 
1. Train model based on your own data:
    One just needs to run main.py
    ```
    python main.py
    ```
    In main.py one can also specify the dataset path, if one wishes to
    try one's own dataset.
2. Predict labels of img_dir:
    ```
    python classify.py --img_dir img_dir
    ```
   where img_dir is the directory with images to classify

### structure
```
Image Classification Project
|- architectures.py
|    Classes and functions for network architectures
|- datasets.py
|    Dataset classes and dataset helper functions. 1:1 as provided by the tutor.
|- main.py
|    Main file. Includes training, evaluation routines and dataset setup
|- classify.py
|    classify images and return their labels, create directory with corresponding
|    label as its filename
|- README.md
|    A readme file containing info on project, example usage and dependencies.
|- utils.py
|    Utility functions and classes. Contains plotting, logging and
|    image augmetation function, Gaussian noise class,
|     image transformation function and augmented dataset class
|- testing.py
|     A testing file, in which code construction took place while
|     running the training. Can evaluate best model on test set, 
|     plot some sample images with transformation type, plot misclassified 
|     images
|- corrupted_mislabeled_images: dir containing corrupted or mislabeld
|     images found in the original training data and a corresponding 
|     csv file
|- documentation files
|     when running main.py it creates various files for documentation:
|       * model.pth: model parameters of model state with highest val acc
|       * best_result.txt: best model acc on test set, architecture and hyperparams
|       * results.txt: all models that completed training are logged here
|                      with acc, architecture and hyperparams
|       * epoch_loss.png: lineplot of train and val acc for each epoch
|       * results: contains architecture.py, model.pth and epoch_loss.png
|       *          of the best model
```

### dependencies
This project uses Pytorch and cuda installation is highly recommended.
Other packages used:
cv2, random, shutil, os, torchvision, matplotlib, collections, numpy,
tqdm, warnings, argparse