# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:32:53 2021

@author: Folk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class NearestMeans():
    def __init__(self):
        self.features_raw = None # features grouped by entry
        self.features = None # features grouped by feature number
        self.class_data = None #features divided into classes
        self.class_mean = None # mean of each feature of each class = trained data
        self.classes_raw = None # classes of each entry
        self.classes = None # list of unique classes
        
        # Variables used exclusively for classification/testing
        self.distance = None # matrix of distances from each entry to each mean
        self.class_result = None # Result of classified class list of each entry as the label name
        self.class_result_index = None # Result of classified class of each entry as an index
        self.error_rate = None
        self.actual_labels = None # Labels used to compare the error rate
        
    def import_training_dataset(self,filename : str):
        n = np.genfromtxt(filename, delimiter=",")
        
        # Extract of features grouped as [x1, x2, x3], [x1, x2, x3], [...] 
        self.features_raw = n[:, 0:-1]         
        
        # Extracts the classes of each entry
        self.classes_raw = n[:, -1]
        
        # List of features grouped as [x1, ...], [x2, ...], [x3, ...]
        self.features = self.features_raw.T  
        
        # List all the classes
        self.classes = np.unique(self.classes_raw)
        
        # Groups the features into their respective classes
        self.divide_features_into_class()
        
        # Perform training
        self.train()
        
        
    def train(self):
        # Helper function = Find means
        def mean_class():
            # Creates n-classes bins to store the mean of each class
            self.class_mean = [[] for _ in range(len(self.classes))]
            
            # For each class, insert each feature-mean into the bins
            for i in range(len(self.class_data)):
                self.class_mean[i] =np.mean(self.class_data[i], axis=0)
            
            # Convert back to numpy array
            self.class_mean = np.array(self.class_mean)
            return self.class_mean
        
        # Finds the feature-means of each class
        mean_class()
        return self.class_mean
    
    def load_trained(self, class_mean, classes = None):
        self.class_mean = class_mean
        
        # If class labels are given, load that
        # otherwise, use the default labels [0, 1, ...]
        if classes is not None:
            assert len(classes) == len(class_mean)
            self.classes = classes
        else: self.classes = list(range(len(x.class_mean)))
    
    def load_testing_data(self,features_raw_filename, labels_given=True):
        assert self.class_mean is not None, 'Please load trained class means first'
        
        file_load = np.genfromtxt(filename, delimiter=",")
        # Select the number of features equal to the number of means given. Ignore the rest
        self.features_raw = file_load[:len(self.class_mean)]
        
        if labels_given is True: 
            self.actual_labels = file_load[-1]
        

        return self.features_raw
    

    def test(self, ret = "index"):
        assert self.features_raw is not None and self.class_mean is not None,\
            'Please load trained data and data to be classified first'
        
        # Calculate the distance matrix between each data point 
        # and each of the loaded class means
        self.distance =  cdist(self.features_raw, self.class_mean)
       
        # Determine the index of the nearest class mean
        self.class_result_index = np.array(
            [np.where(arr == np.amin(arr))[0] for arr in x.distance]).T[0]
        
        # Convert the resulting indices into the nearest class labels
        self.class_result = np.array([self.classes[i] for i in self.class_result_index])
        
        # Calculate the total error rate
        
        
        # Return either the nearest index or nearest label
        if ret == "labels" : return self.class_result
        else : return self.class_result_index
        

    def plot_test_results(self):
        assert self.class_result is not None,\
            'Please classify/test the data first.'
        self.plot_decision_boundaries_2 \
            (self.features_raw, self.class_result, self.class_mean)
    
    def norm_l2(self, a, b):
        a = np.array(a)
        b = np.array(b)
        assert a.shape[0] == b.shape[0],\
            "Dimensions of input must be the same. Currently {0} and {1}".format(a.shape[0], b.shape[0])
        return np.linalg.norm(a - b, ord=2)
    
    def divide_features_into_class(self):
        # Creates bins to store data from each class
        self.class_data = [[] for _ in range(len(self.classes))]
        
        # For each entry, place entry into the bins
        for i in range(len(self.features_raw)):
            self.class_data[ (np.where(self.classes == 
                                       self.classes_raw[i])[0][0]) ].append(self.features_raw[i])
        
        # Convert back to numpy array
        self.class_data = np.array(self.class_data)
        return self.class_data
    
    
    def plot_decision_boundaries(self, training, label_train, sample_mean):
        # Code for this method provided by Prof B. Keith Jenkins (USC)
        
        #Plot the decision boundaries and data points for minimum distance to
        #two class mean classifier
        #
        # training: traning data (features_raw)
        # label_train: class labels correspond to training data (classes_raw-untrained or class_result - trained)
        # sample_mean: mean vector for each class (class_mean)
        #
        # Total number of classes
        nclass =  max(np.unique(label_train))
    
        # Set the feature range for ploting
        max_x = np.ceil(max(training[:, 0])) + 1
        min_x = np.floor(min(training[:, 0])) - 1
        max_y = np.ceil(max(training[:, 1])) + 1
        min_y = np.floor(min(training[:, 1])) - 1
    
        xrange = (min_x, max_x)
        yrange = (min_y, max_y)
    
        # step size for how finely you want to visualize the decision boundary.
        inc = 0.005
    
        # generate grid coordinates. this will be the basis of the decision
        # boundary visualization.
        (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))
    
        # size of the (x, y) image, which will also be the size of the
        # decision boundary image that is used as the plot background.
        image_size = x.shape
        xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.
    
        # distance measure evaluations for each (x,y) pair.
        dist_mat = cdist(xy, sample_mean)
        pred_label = np.argmin(dist_mat, axis=1)
    
        # reshape the idx (which contains the class label) into an image.
        decisionmap = pred_label.reshape(image_size, order='F')
    
        #show the image, give each coordinate a color according to its class label
        plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
    
        # plot the class training data.
        plt.plot(training[label_train == 1, 0],training[label_train == 1, 1], 'rx')
        plt.plot(training[label_train == 2, 0],training[label_train == 2, 1], 'go')
        if nclass == 3:
            plt.plot(training[label_train == 3, 0],training[label_train == 3, 1], 'b*')
    
        # include legend for training data
        if nclass == 3:
            l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
        else:
            l = plt.legend(('Class 1', 'Class 2'), loc=2)
        plt.gca().add_artist(l)
    
        # plot the class mean vector.
        m1, = plt.plot(sample_mean[0,0], sample_mean[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
        m2, = plt.plot(sample_mean[1,0], sample_mean[1,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
        if nclass == 3:
            m3, = plt.plot(sample_mean[2,0], sample_mean[2,1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')
    
        # include legend for class mean vector
        if nclass == 3:
            l1 = plt.legend([m1,m2,m3],['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
        else:
            l1 = plt.legend([m1,m2], ['Class 1 Mean', 'Class 2 Mean'], loc=4)
    
        plt.gca().add_artist(l1)
    
        plt.show()
    
if __name__ == '__main__':
    x = NearestMeans()
    x.import_training_dataset("../data/synthetic1_train.csv")
    #print((a[1]))
    #x.plot_decision_boundaries(np.array([[1,2]]), np.array([[2,1]), np.array([1,2]))#test[0], test[1], test[2]))