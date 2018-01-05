# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
from PIL import Image

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()
            
        #train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        #train_data, labels = self._post_process(train_data, labels)

        nz = data.shape[1]
        nx = data.shape[2]
        ny = data.shape[3]

        return data.reshape(1, nz, nx, ny, self.channels), labels.reshape(1, nz, nx, ny, self.n_class),
    
    def _process_labels(self, label):
        if self.n_class == 2:
            nz = label.shape[1]
            nx = label.shape[2]
            ny = label.shape[3]
            labels = np.zeros((nz, nx, ny, self.n_class), dtype=np.float32)
            labels[..., 1] = label[0, ..., 0]
            labels[..., 0] = 1 - label[0, ..., 0]
            return labels
        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nz = train_data.shape[1]
        nx = train_data.shape[2]
        ny = train_data.shape[3]
    
        X = np.zeros((n,nz, nx, ny, self.channels))
        Y = np.zeros((n, nz, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data[0, ...]
            Y[i] = labels[0, ...]
    
        return X, Y

