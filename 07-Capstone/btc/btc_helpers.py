# coding: utf-8
# ### Import Necessary Modules

import os
import warnings

import cv2
import h5py
import keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


# %matplotlib inline


class DataUtil:

    def __init__(self, srcPath, trainDir, testDir):
        self.srcPath = srcPath
        self.trainDir = trainDir
        self.testDir = testDir
        self.__trainArr = []
        self.__trainDf = []
        self.__testArr = []
        self.__testDf = []
        self.__mergeAndSplit = 'Original'
        self.__cached = False
        self.__cachedSize = 0

    def getTrainTestData(self, imgResize, mergeSplit=None, forceReread=False):
        """Get train and test datasets
        Parameters
        ----------
        imgResize : An integer corresponding to resize ex. 256
        mergeSplit : a str value ['None','all']. If 'all' then concatenate
                     train and test image arrays as well as labels and do a
                     80:20 to return train and test datasets.
                     defaults to 'None'
        forceReread : force re-caching the dataset, if method is called with
                      resize parameter different from first call, trigger forceReread

        Returns
        -------
        trainImgArr: numpy.ndarray of training images of shape(nImages,resize,resize)
        testImgArr: numpy.ndarray of testing images of shape(nImages,resize,resize)
        trainDf: A pandas.DataFrame of original and cropped image information for training dataset
        testDf: A pandas.DataFrame of original and cropped image information for testing dataset
        """
        if not self.__cached or forceReread or self.__cachedSize != imgResize:
            print('Updating cache with training and testing datasets')
            self.__updateCache(imgResize)

        if mergeSplit == 'all':
            print('merging cached training and testing datasets')
            mergeArr = np.concatenate((self.__trainArr, self.__testArr), axis=0)
            mergeDf = self.__trainDf.append(self.__testDf, ignore_index=True)
            # split into train and test sets and update
            mergeSplitRatio = 0.2
            print(f'\nSplitting ratio for merged dataset is set to {mergeSplitRatio:.2f}')
            self.__trainArr, self.__testArr, self.__trainDf, self.__testDf = \
                train_test_split(mergeArr, mergeDf, test_size=mergeSplitRatio, shuffle=True)  # auto-stratify
            self.__trainDf.reset_index(inplace=True)
            self.__testDf.reset_index(inplace=True)
            self.__mergeAndSplit = 'Merged&Split'
        # return the cached dataset
        print(f'Returning cached [{self.__mergeAndSplit}] training and testing datasets')
        return self.__trainArr, self.__testArr, self.__trainDf, self.__testDf

    def __updateCache(self, imgResize):
        """Update train and test datasets
        Parameters
        ----------
        imgResize : An integer corresponding to resize ex. 256

        Returns
        -------
        """
        trainFile = self.trainDir + '_' + str(imgResize) + '.h5'
        testFile = self.testDir + '_' + str(imgResize) + '.h5'
        trainHdf5File = os.path.join(self.srcPath, trainFile)
        testHdf5File = os.path.join(self.srcPath, testFile)
        # assert files exist
        # assert os.path.exists(trainHdf5File), os.path.abspath(trainHdf5File)
        # assert os.path.exists(testHdf5File), os.path.abspath(testHdf5File)
        if not os.path.exists(trainHdf5File) or (not os.path.exists(testHdf5File)):
            print('Converting dataset to HDF5 files')
            self.convertToHdf5(self.trainDir, imgResize)
            self.convertToHdf5(self.testDir, imgResize)

        # Read train and test HDF5 files
        print('Caching train and test datasets')
        self.__trainArr, self.__trainDf = self.readHdf5File(trainHdf5File)
        self.__trainDf['setName'] = 'Training'
        self.__trainDf['imageUID'] = self.__trainDf['setName'] \
            + '_' + self.__trainDf['tumorCategory'] + '_' + self.__trainDf['fileId']
        self.__testArr, self.__testDf = self.readHdf5File(testHdf5File)
        self.__testDf['setName'] = 'Testing'
        self.__testDf['imageUID'] = self.__testDf['setName'] + '_' \
            + self.__testDf['tumorCategory'] + '_' + self.__testDf['fileId']
        # set cached flag
        self.__cached = True
        self.__cachedSize = imgResize
        # return nothing

    def convertToHdf5(self, dirName,imgResize):
        dirPath = os.path.join(self.srcPath, dirName)
        hdfFile = os.path.join(self.srcPath, (dirName + '_' + str(imgResize) + '.h5'))
        if not os.path.exists(hdfFile):
            imgArr, imgInfoDf = self.getImageDataset(dirPath, imgResize)
            self.writeHdf5File(hdfFile, imgArr, imgInfoDf)
        imgArr, infoDf = self.readHdf5File(hdfFile)
        return imgArr, infoDf

    @staticmethod
    def cropImg(inImg, thresh=25):
        """Crops an image at a given threshold of grayscale value
        Parameters
        ----------
        inImg : An image - A 2D array of int values (0-255)
        thresh : A int specifying the grayscale value at which to
                 crop the image default=25

        Returns
        -------
        Cropped image
        """
        masked = inImg > thresh
        return inImg[np.ix_(masked.any(1), masked.any(0))]

    # Read original image data directory structure
    def getImageDataset(self, dirPath, imgResize):
        """Read image datafiles from directory path with category
           names as sub-directories and default crops at grayscale
           value of 25, to remove borders
        Parameters
        ----------
        dirPath : Full path to the directory containing category folders
                  example [BrainTumorDataSet/Training] containing list of
                  sub-folder names
                  ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
                  that each have the image files
        imgResize : An int specifying how each image should be resized

        Returns
        -------
        images : An array of images that correspond to the order it was read
                 from the directory structure the information is captured in
                 the ImgInfoDf below
        imageInfoDf : A dataframe of image information for each image in the images array
        """
        categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        _labels = []
        _files = []
        _width = []
        _height = []
        _cropWidth = []
        _cropHeight = []
        images = []
        for label in categories:
            labelPath = os.path.join(dirPath, label)
            for f in os.listdir(labelPath):
                # get image metadata
                inImg = cv2.imread(os.path.join(labelPath, f), cv2.IMREAD_GRAYSCALE)
                _labels.append(label)
                _files.append(f)
                _width.append(inImg.shape[0])
                _height.append(inImg.shape[1])
                img = self.cropImg(inImg)
                _cropWidth.append(img.shape[0])
                _cropHeight.append(img.shape[1])
                images.append(cv2.resize(img, (imgResize, imgResize)))

        imageInfoDf = pd.DataFrame({'tumorCategory': _labels, 'fileId': _files,
                                    'origWidth': _width, 'origHeight': _height,
                                    'cropWidth': _cropWidth,
                                    'cropHeight': _cropHeight})
        return images, imageInfoDf

    # Utilities for reading and writing datafiles
    @staticmethod
    def readHdf5Var(hdfFile, varName):
        """Read specific variable from previously HDF5 format datafile

        Parameters
        ----------
        hdfFile : Name of HDF5 file containing data with full path
        varName : Name of the variable saved to the file. Valid names are
                  ['tumorCategory','origWidth','origHeight',
                   'cropWidth','cropHeight','images']

        Returns
        -------
        An named array stored in the file
        """
        varList = ['tumorCategory', 'origWidth', 'origHeight', 'cropWidth', 'cropHeight', 'images']
        assert varName in varList, f'Variable name not in list {varList}'
        h5f = h5py.File(hdfFile, 'r')
        if varName in ['tumorCategory', 'fileId']:
            return getShortLabels(np.array(h5f['/' + varName]).astype("str"))
        elif varName in ['origWidth', 'origHeight', 'cropWidth', 'cropHeight']:
            return np.array(h5f['/' + varName]).astype("uint16")
        elif varName == 'images':
            return np.array(h5f["/images"]).astype("uint8")
        else:
            print(f'****Var name {varName} not found in file {hdfFile}****')
            return None

    # see: https://realpython.com/storing-images-in-python/
    # we will store all the images into an HDF5 dataset for faster image loading
    # quote: "HDF has its origins in the National Center for Supercomputing Applications,
    #         as a portable, compact scientific data format"
    # write to HDF5 file
    @staticmethod
    def writeHdf5File(hdfFile, imgArr, imgInfoDf):
        """Write HDF5 format datafile for imgArr and imgInfoDf
           To be called after calling getImageDataset method
        Parameters
        ----------
        hdfFile : Name of HDF5 file containing data with full path
                  example: os.path.join(dirName,hdf5File + '_' + str(resize) + '.h5')
        imgArr : An image array to be stored in the file
        imgInfoDf : A dataframe of image information for each image in the imgArr
        """
        _str = h5py.string_dtype(encoding='utf-8', length=None)  # for category labels
        _ui8 = h5py.h5t.STD_U8BE  # for image pixel values 0-255
        _ui16 = h5py.h5t.STD_U16BE  # for img width height
        # data other than image array
        tCat = [lab.encode('ascii', 'ignore') for lab in imgInfoDf['tumorCategory']]
        fileId = [f.encode('ascii', 'ignore') for f in imgInfoDf['fileId']]

        h5f = h5py.File(hdfFile, 'w')
        h5f.create_dataset('images', np.shape(imgArr), _ui8, data=imgArr,
                           compression='gzip', compression_opts=9)
        # column 0
        h5f.create_dataset('tumorCategory', np.shape(tCat), _str, data=tCat,
                           compression="gzip", compression_opts=9)
        # column 1
        h5f.create_dataset('fileId', np.shape(fileId), _str, data=fileId,
                           compression="gzip", compression_opts=9)

        # column 2:end are all unsigned int16s
        for tag in imgInfoDf.columns[2:]:
            vals = imgInfoDf[tag]
            h5f.create_dataset(tag, np.shape(vals), _ui16, data=vals,
                               compression="gzip", compression_opts=9)
        h5f.close()
        print(f'wrote file {os.path.abspath(hdfFile)}')

    @staticmethod
    def readHdf5File(hdfFile):
        """Read a previously written Training or Testing HDF5 format datafile

        Parameters
        ----------
        hdfFile : Name of HDF5 file containing data with full path
                  os.path.join(dirName,hdf5File + '_' + str(resize) + '.h5')

        Returns
        -------
        imgArr : An image array stored in the file
        labels : An array of labels that correspond to axis[0] in the imgArr
        infoDf : A dataframe of image information for each image in the imgArr
        """
        print(f'Reading HDF5 file {hdfFile}')
        assert os.path.exists(hdfFile)
        h5f = h5py.File(hdfFile, 'r')
        imgArr = np.array(h5f["/images"]).astype("uint8")
        infoDf = pd.DataFrame()
        infoDf['tumorCategory'] = np.array(h5f["/tumorCategory"]).astype("str")
        infoDf['fileId'] = np.array(h5f["/fileId"]).astype("str")

        tags = ['origWidth', 'origHeight', 'cropWidth', 'cropHeight']
        for tag in tags:
            infoDf[tag] = np.array(h5f['/' + tag]).astype('uint16')
        # shorten labels
        infoDf['tumorCategory'] = getShortLabels(infoDf['tumorCategory'].values)

        return imgArr, infoDf


def getCPUorGPUorTPUStrategy():
    """Check if have TPU or GPU or just CPU
    Returns
    -------
    A tf.distribute.get_strategy() for implementing data parallelism.
      strategy is used for distributing jobs
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print('Running on TPU')
        return tf.distribute.TPUStrategy(tpu)
    except ValueError:  # if TPU not found
        tpu = None
        # print('No TPU found!')
        if tpu is None:
            # Detect GPU hardware
            gpu = tf.test.gpu_device_name()  # check oif you have a GPU
            if gpu == '/device:GPU:0':
                print('Running on GPU')
                # !/opt/bin/nvidia-smi
                return tf.distribute.get_strategy()
            else:
                print('Running on CPU')
                return tf.distribute.get_strategy()


# Get distribution of labels
def getLabelDistributionDf(labelsDict):
    """Computes the counts and fraction of total counts for set of labels

    Parameters
    ----------
    labelsDict : {key=value key=`str`, value=`array-like`}
                given as train=trainLabels, test=testLabels
                key `str` and value `array-like` of category labels.

    Returns
    -------
    Dataframe of label distribution with index of `category labels` and column
             names as `key, keyFraction`
    """
    labelCounts = {}
    for key, val in labelsDict.items():
        labelCounts[key] = pd.Series(val).value_counts()
        labelCounts[''.join([key, 'Fraction'])] = labelCounts[key] / labelCounts[key].sum()
    labelDist = pd.DataFrame(labelCounts)
    labelDist.sort_index(inplace=True)  # always get labels in same order
    labelDist.loc['Total'] = labelDist.sum(axis=0)
    return labelDist


# Plot an array of images
def plotImageArr(nRows, nCols, figSize, imgArr, rowLabels=None, colLabels=None, figFile=None):
    """Plot a given array of images into a grid with labels, if specified and save plot
    Parameters
    ----------
    nRows : An int specifying number of rows in the plot grid
    nCols : An int specifying number of columns in the plot grid
    figSize : A tuple of 2 numbers specifying the size of figure
    imgArr : The image array that needs to be plotted.
             The image array must be (nRows*nCols, width, height [nChannels])
             Only the 0th channel is plotted in case of a 4D-tensor
    rowLabels : A list for labels for each row of images.
                Labels are used (aa y-label) only for the first column
    colLabels : A list of labels for the column of images.
                Labels are used (as title) only for the first row
    figFile : optional. A full path to file to which the figure will be saved
              defaults to None, whjich does not save the figure file
    """
    assert imgArr.shape[0] == nRows * nCols
    # get handles
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=figSize, sharex=True, sharey=True)
    # remove '_tumor' suffix if labels have it
    rowLabels = getShortLabels(rowLabels)
    colLabels = getShortLabels(colLabels)
    imgIdx = 0
    for r in range(nRows):
        for c in range(nCols):
            if nRows == 1 or nCols == 1:
                pltIx = (r + c)
            else:
                pltIx = (r, c)
            ax = axs[pltIx]
            # check tensor dimensions
            if np.ndim(imgArr) == 3:
                ax.imshow(imgArr[imgIdx, :, :], cmap='gray')
            else:
                ax.imshow(imgArr[imgIdx, :, :, 0], cmap='gray')
            ax.set_axis_off()
            # add row label to the 1st column
            if c == 0 and len(rowLabels) > 0:
                ylab = rowLabels[imgIdx]
                ax.annotate(ylab, xy=(0, 0), xytext=(-.1, .3), textcoords='axes fraction',
                            ha='center', va='bottom', rotation=90, fontweight='bold')
            # add column label to the 1st row
            if r == 0 and len(colLabels) > 0:
                ax.set_title(colLabels[c])
            imgIdx = imgIdx + 1
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=.01, hspace=0.05)
    if figFile is not None:
        plt.savefig(figFile, bbox_inches='tight')
    plt.show()


def getShortLabels(labels):
    """Remove '_tumor' suffix if labels have it except for no_tumor
    Parameters
    ----------
    labels : An array of labels

    Returns
    -------
    Returns an array of labels with _tumor suffix removed except for no_tumor
    """
    if labels is not None:
        if len(labels) > 0:
            return np.array([lab if lab.startswith('no_') else lab.replace('_tumor', '') for lab in labels])
        else:
            return labels
    else:
        return []  # return an empty array


def _pieTexts(percentVal, allCounts):
    catCount = int(percentVal / 100.0 * np.sum(allCounts))
    return f'{percentVal:.1f}%\n({catCount:d})'


def _flushPlot(figFile):
    if figFile is not None:
        plt.savefig(figFile, bbox_inches='tight')
    plt.show()


def plotPieDistribution(trainInfoDf, testInfoDf, figFile=None):
    dat = dict(Training=trainInfoDf, Testing=testInfoDf)
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharex=True)
    pltNo = -1
    for figTit, df in dat.items():
        pltNo = pltNo + 1
        d = getLabelDistributionDf(dict(t=df['tumorCategory']))[:-1]
        explode = [0.02, 0.02, 0.02, 0.02]
        # explode lowers more
        explode[np.where(d['t'] == min(d['t']))[0][0]] = 0.10

        ax = axs[pltNo]
        # w, tx, aTx = \
        ax.pie(x=d['t'], autopct=lambda pcentVal: _pieTexts(pcentVal, d['t']),
               explode=explode, textprops=dict(color='w', weight='bold', size=14),
               labels=d.index)
        if pltNo == 0:
            # ax.legend(loc='top right')
            # ax.legend(bbox_to_anchor=(1.1, 0.8),loc = 'center',frame)
            # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ax.legend(bbox_to_anchor=(0., -0.1, 2., .2), loc='lower left',
                      ncol=4, mode="expand", borderaxespad=0., frameon=False, fontsize='x-large')
        ax.set_title(figTit, fontdict={'fontsize': 14, 'fontweight': 'bold'}, pad=0.1)
    if figFile is not None:
        plt.savefig(figFile, bbox_inches='tight')
    plt.show()


def plotWidthHeightViolin(trainInfoDf, testInfoDf, xyCols, figFile=None):
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharex=True)
    pltNo = -1
    for df in [trainInfoDf, testInfoDf]:
        pltNo = pltNo + 1
        ax = axs[pltNo]
        dfMelt = pd.melt(df, id_vars='tumorCategory', value_vars=xyCols,
                         var_name='imageDim', value_name='pixels')
        g = sns.violinplot(x='tumorCategory', y='pixels', hue='imageDim', split=True,
                           palette='Set2', alpha=0.2, linewidth=0.5,
                           scale='count', inner='quartile', data=dfMelt, ax=ax)
        ax.legend(loc='upper left', frameon=False)
        if pltNo > 0:
            g.legend_.remove()
        # if pltNo > 0:
        #     ax.legend(loc='upper left', frameon=False)
    if figFile is not None:
        plt.savefig(figFile, bbox_inches='tight')
    plt.show()


def plotAspect(trainInfoDf, testInfoDf, xyCols, lims, figFile=None):
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharex=True)
    pltNo = -1
    for df in [trainInfoDf, testInfoDf]:
        pltNo = pltNo + 1
        ax = axs[pltNo]
        g = sns.kdeplot(data=df, x=xyCols[0], y=xyCols[1], hue='tumorCategory',
                        alpha=0.6, fill=True, ax=ax)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        # draw diagonal line for square shape ref.
        g.plot(lims, lims, ':k', linewidth=0.5)
        if pltNo > 0:
            g.legend_.remove()
    if figFile is not None:
        plt.savefig(figFile, bbox_inches='tight')
    plt.show()


def plotPixelValueDist(trainArr, testArr, trainInfoDf, testInfoDf, scale='Count', ylim=None, figFile=None):
    catLabels = sorted(trainInfoDf['tumorCategory'].unique())
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharex=True)
    pltNo = -1
    for imArr, df in zip([trainArr, testArr], [trainInfoDf, testInfoDf]):
        # get pixel value distribution for each category
        # assume raw image so pixel values 0-255
        imDf = pd.DataFrame()
        imDf['Pixel Value'] = np.arange(256)
        for cat in catLabels:
            idx = np.where(df['tumorCategory'] == cat)
            imDf[cat] = np.bincount(imArr[idx].reshape(-1), minlength=256)
        imDf = imDf.melt(id_vars='Pixel Value', var_name='tumorCategory', value_name='Count')
        yvar = 'Count'
        if scale == 'log':
            yvar = 'log. Count'
            imDf[yvar] = np.log10(imDf['Count'] + 1)  # log(0)=-1
        pltNo = pltNo + 1
        ax = axs[pltNo]
        g = sns.histplot(data=imDf, x='Pixel Value', y=yvar, hue='tumorCategory',
                         bins=256, kde=True, alpha=0.5, ax=ax)
        if ylim is not None:
            ax.set_ylim(ylim)
        # ax.legend_(loc='upper left', frameon=False)
        if pltNo > 0:
            g.legend_.remove()
    if figFile is not None:
        plt.savefig(figFile, bbox_inches='tight')
    plt.show()


# ### Image processing: Using lambdas directly is not pythonic so using def...

# Normalize image (subtract mean/range)
def imNorm(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))


# Z-score image
def imZsc(im):
    ret = (im - np.mean(im)) / np.std(im)
    ret[np.where(ret > 3.0)] = 3.0
    ret[np.where(ret < -3.0)] = -3.0
    return ret


# log transform
def imLog(im):
    return imNorm(np.log10(im + 1))


# Laplacian transformation
def imLaplacian(im):
    ret = cv2.Laplacian(im, cv2.CV_64F, ksize=21)
    return imNorm(ret)


# create a histogram equalized image
def imHe(im):
    return imNorm(cv2.equalizeHist(im))


# Contrast Limited Adaptive Histogram normalization
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def imClahe(im):
    return imNorm(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(im))


# Scharr edge filter
def imScharr(im):
    return imNorm(cv2.Scharr(cv2.Scharr(im, cv2.CV_64F, 1, 0), cv2.CV_64F, 0, 1))


# changed Z-Scored to ZScored since the fx name is got by splitting on '-' :-)
__PRE_PROC_FXS = {'Raw': None,
                  'Scaled': lambda xx: xx / 255.0,
                  'Normalize': imNorm,
                  'ZScore': imZsc,
                  'HistEqual': imHe,
                  'CLAHE': imClahe,
                  'Laplacian': imLaplacian,
                  'LogTransform': imLog,
                  'ZScored_HistEqual': lambda xx: imZsc(imHe(xx)),
                  'ZScored_CLAHE': lambda xx: imZsc(imClahe(xx)),
                  'ZScored_Laplacian': lambda xx: imZsc(imLaplacian(xx)),
                  'ZScored_LogTransform': lambda xx: imZsc(imLog(xx))
                  }


def getPreProcNames():
    return [k for k in __PRE_PROC_FXS.keys()]


def getPreProcFx(fxName):
    assert fxName in __PRE_PROC_FXS.keys()
    return __PRE_PROC_FXS[fxName]


# #################### Modelling class ################
class MyModel:

    # build Base model, no hyper params
    # l2reg: https://medium.com/@mjbhobe/classifying-fashion-with-a-keras-cnn-achieving-94-accuracy-part-2-a5bd7a4e7e5a
    # noinspection DuplicatedCode
    @staticmethod
    def getBaseModel(shapeTensor, strategy, numClasses=4, bNorm=False, drop=0, l2Lambda=None):

        kernelL2 = None if l2Lambda is None else keras.regularizers.l2(l2=l2Lambda)

        with strategy.scope():
            # Inputs
            inputs = keras.Input(shape=shapeTensor)
            # build model using functional APIs
            x = inputs
            # conv-1
            x = MyModel.__convBlock(x, 32, kSize=3, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
            # conv-2
            x = MyModel.__convBlock(x, 64, kSize=3, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
            # conv-3
            x = MyModel.__convBlock(x, 32, kSize=3, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
            # conv-4
            x = MyModel.__convBlock(x, 16, kSize=5, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
            # flatten to dense...
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(512, kernel_regularizer=kernelL2)(x)
            x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
            if drop > 0:
                x = keras.layers.Dropout(0.3)(x)

            x = keras.layers.Dense(256, kernel_regularizer=kernelL2)(x)
            x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
            if drop > 0:
                x = keras.layers.Dropout(0.2)(x)

            outputs = keras.layers.Dense(numClasses, activation='softmax')(x)
            model = keras.Model(inputs, outputs)
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6),
                          metrics=['accuracy',
                                   keras.metrics.Precision(name='precision'),
                                   keras.metrics.Recall(name='recall')])
        return model

    # noinspection DuplicatedCode
    @staticmethod
    def getBaseModelNew(shapeTensor, numClasses=4, bNorm=False, drop=0, l2Lambda=None,
                        loss='categorical_crossentropy', optimizer=None, metrics='accuracy'):
        kernelL2 = None if l2Lambda is None else keras.regularizers.l2(l2=l2Lambda)
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)
        # Inputs
        inputs = keras.Input(shape=shapeTensor)
        # build model using functional APIs
        x = inputs
        # conv-1
        x = MyModel.__convBlock(x, 32, kSize=3, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
        # conv-2
        x = MyModel.__convBlock(x, 64, kSize=3, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
        # conv-3
        x = MyModel.__convBlock(x, 32, kSize=3, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
        # conv-4
        x = MyModel.__convBlock(x, 16, kSize=5, bNorm=bNorm, drop=drop, kernelL2=kernelL2)
        # flatten to dense...
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(512, kernel_regularizer=kernelL2)(x)
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
        if drop > 0:
            x = keras.layers.Dropout(0.3)(x)

        x = keras.layers.Dense(256, kernel_regularizer=kernelL2)(x)
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
        if drop > 0:
            x = keras.layers.Dropout(0.2)(x)

        outputs = keras.layers.Dense(numClasses, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model

    @staticmethod
    def getBaseModel5(shapeTensor, strategy, numClasses=4):
        with strategy.scope():
            # Inputs
            inputs = keras.Input(shape=shapeTensor)
            # build model using functional APIs
            x = inputs
            # conv-1
            x = MyModel.__convBlock(x, 32, kSize=3, bNorm=False, drop=0)
            # conv-2
            x = MyModel.__convBlock(x, 64, kSize=3, bNorm=False, drop=0)
            # conv-3
            x = MyModel.__convBlock(x, 64, kSize=3, bNorm=False, drop=0)
            # conv-4
            x = MyModel.__convBlock(x, 32, kSize=5, bNorm=False, drop=0)
            # conv-5
            x = MyModel.__convBlock(x, 16, kSize=5, bNorm=False, drop=0)
            # flatten to dense...
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(512)(x)
            x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
            x = keras.layers.Dense(256)(x)
            x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
            outputs = keras.layers.Dense(numClasses, activation='softmax')(x)
            model = keras.Model(inputs, outputs)
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6),
                          metrics=['accuracy',
                                   keras.metrics.Precision(name='precision'),
                                   keras.metrics.Recall(name='recall')])
        return model

    @staticmethod
    def getModelPaper2020(shapeTensor, strategy, numClasses=4):
        # from paper Classification of... Applied Sci. 2020
        with strategy.scope():
            # Inputs
            inputs = keras.Input(shape=shapeTensor)
            # build model using functional APIs
            x = inputs
            # conv-1
            x = MyModel.__convBlock2020(x, nFilters=16, kSize=5, kStride=2, poolStride=2, drop=0.3)
            # conv-2
            x = MyModel.__convBlock2020(x, nFilters=32, kSize=3, kStride=2, poolStride=2, drop=0.3)
            # conv-3
            x = MyModel.__convBlock2020(x, nFilters=64, kSize=3, kStride=1, poolStride=2, drop=0.3)
            # conv-4
            x = MyModel.__convBlock2020(x, nFilters=128, kSize=3, kStride=1, poolStride=2, drop=0.3)
            # flatten to dense...
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(1024, activation='relu')(x)
            outputs = keras.layers.Dense(numClasses, activation='softmax')(x)
            model = keras.Model(inputs, outputs)
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                          metrics=[keras.metrics.Accuracy(name='accuracy'),
                                   keras.metrics.Precision(name='precision'),
                                   keras.metrics.Recall(name='recall')])
        return model

    @staticmethod
    def __convBlock2020(x, nFilters=16, kSize=5, kStride=2, poolStride=2, drop=0.0):
        x = keras.layers.Conv2D(filters=nFilters, kernel_size=kSize,
                                strides=kStride, activation='relu',
                                padding='same')(x)
        if drop > 0.0:
            x = keras.layers.Dropout(drop)(x)
        x = keras.layers.MaxPooling2D(pool_size=2, strides=poolStride, padding='same')(x)
        return x

    @staticmethod
    def __convBlock(x, filtNo, kSize=3, bNorm=False, drop=0.0, kernelL2=None):
        # conv ->batchNorm->maxPool->dropout
        x = keras.layers.Conv2D(filters=filtNo, kernel_size=kSize,
                                kernel_regularizer=kernelL2, padding='same')(x)
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
        if bNorm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)
        if drop > 0.0:
            x = keras.layers.Dropout(drop)(x)
        return x

    @staticmethod
    def _convBlock(x):
        x = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(x)
        return x


def getTrainTestIndexes(trainLabels, testSplit=0.2):
    # split training data into get train and validation indexes for use across models:
    # No Random-state -> every call will give different indices.
    # here we will call this *once* for running different models so that have same
    # train, val split
    trainIndex, validationIndex = train_test_split(
        trainLabels, test_size=testSplit, shuffle=True)
    trainIndex = trainIndex.index
    validationIndex = validationIndex.index
    # use sorted labels from distribution as categories
    labelDist = getLabelDistributionDf({'Train': trainLabels[trainIndex],
                                        'Val': trainLabels[validationIndex]})
    tumorCategories = labelDist.index[:-1]
    classWeights = compute_class_weight('balanced', tumorCategories, trainLabels[trainIndex])
    classWeightsDict = dict(zip(np.arange(4), classWeights))
    return trainIndex, validationIndex, tumorCategories, classWeightsDict


# Using predefined split indices to run different image processing calls
# tried to do sklearn pipeline... but this was simpler
def getTrainValTestData(trainImgs, trainLabels, trainIndex, validationIndex,
                        testImgs, testLabels, preProcName='Z-Scored'):
    if type(trainIndex) is pd.Series:
        # print('is Series')
        trainIndex = trainIndex.index
    if type(validationIndex) is pd.Series:
        validationIndex = validationIndex.index
    # pre-process images and expand to 4D tensor:
    fx = getPreProcFx(preProcName)
    # Prepare data to be 4D tensors
    if fx is None:
        print(f'No Pre-Processing raw image arrays')
        tr_x = np.expand_dims(trainImgs[trainIndex], axis=3)
        vl_x = np.expand_dims(trainImgs[validationIndex], axis=3)
        ts_x = np.expand_dims(testImgs, axis=3)
    else:
        print(f'Pre-Processing raw image arrays with function to {preProcName}')
        tr_x = np.expand_dims(np.array([fx(x) for x in trainImgs[trainIndex]]), axis=3)  # add 4the dimension
        vl_x = np.expand_dims(np.array([fx(x) for x in trainImgs[validationIndex]]), axis=3)  # add 4the dimension
        ts_x = np.expand_dims(np.array([fx(x) for x in testImgs]), axis=3)  # add 4the dimension

    # OneHotEncode class names
    tumorCategoryOHE = OneHotEncoder()
    tr_y = tumorCategoryOHE.fit_transform(trainLabels[trainIndex].values.reshape(-1, 1)).toarray()
    vl_y = tumorCategoryOHE.transform(trainLabels[validationIndex].values.reshape(-1, 1)).toarray()
    ts_y = tumorCategoryOHE.transform(testLabels.values.reshape(-1, 1)).toarray()

    classLabels = [x.replace('x0_', '') for x in tumorCategoryOHE.get_feature_names()]
    # Print model training data information
    # print('Shapes for model input and evaluation')
    # print(f'  Training: image-array (tr_x) {tr_x.shape}, OHE-labels (tr_y) {tr_y.shape}')
    # print(f'Validation: image-array (vl_x) {vl_x.shape}, OHE-labels (vl_y) {vl_y.shape}')
    # print(f'   Testing: image-array (ts_x) {ts_x.shape}, OHE-labels (vl_y) {ts_y.shape}')
    # display(btc.getLabelDistributionDf({'Train':trainLabels[tr_ix],'Val':trainLabels[vl_ix],'Test':testLabels}))
    # print(f'Class weights for different categories {classWeights}')
    # print(f'Class weights Dict for different categories {classWeightsDict}')
    # print('Class weights * trainFraction:', classWeights*labelDist['TrainFraction'][:-1] )
    return tr_x, tr_y, vl_x, vl_y, ts_x, ts_y, classLabels


# Plot accuracy and loss
def plotAccuracyAndLoss(trainingHistory, saveFn='', varNames=None):
    if varNames is None:
        varNames = ['loss', 'accuracy']
    epochList = [1 + x for x in trainingHistory.epoch]
    history = trainingHistory.history
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochList, history[varNames[0]], ls='-.', label='train_' + varNames[0])
    plt.plot(epochList, history['val_' + varNames[0]], ls='-.', label='validation_' + varNames[0])
    plt.ylabel(varNames[0])
    plt.xlabel('Epochs')
    plt.legend()
    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochList, history[varNames[1]], ls='--', label='train_' + varNames[1])
    plt.plot(epochList, history['val_' + varNames[1]], ls='--', label='validation_' + varNames[1])
    plt.ylabel(varNames[1])
    plt.xlabel('Epochs')
    plt.legend()
    if len(saveFn) > 0:
        plt.savefig(saveFn, bbox_inches='tight')
    plt.show()


# prediction classification report
def getClassificationReport(model, testData, testCats, targetNames, asDataframe=False):
    test_pred = model.predict(testData)
    if asDataframe:
        df = pd.DataFrame(classification_report(
            np.argmax(testCats, axis=1), np.argmax(test_pred, axis=1),
            target_names=targetNames, output_dict=True))
        df.index.name = 'tumorCategory'
        df = df.reset_index()
        return df
    else:  # just text
        return classification_report(np.argmax(testCats, axis=1),
                                     np.argmax(test_pred, axis=1),
                                     target_names=targetNames)


if __name__ == '__main__':
    print('in Main')
    dataPath = '../../DataSetBrainTumor'  # dir or link to dir for running local
    outputPath = 'outputMergedData'  # dir or link to dir (where the source file is)
    # Output path for figures, models, and model-tuning
    modelPath = os.path.join(outputPath, 'models')
    figurePath = os.path.join(outputPath, 'figures')
    modelTunerPath = os.path.join(outputPath, 'model-tuner')
    # ######################################################
    # btc = BTCHelper(dataPath,'Training','Testing')
    # # test train and test datasets
    # trainArr,testArr,trainDf,testDf = btc.getDataSet(256)
    # print(trainArr.shape, testArr.shape)
    # print(trainDf)
    # print(testDf)
    # # test train and test datasets for merge and split
    # trainArr, testArr, trainDf, testDf = btc.getDataSet(256,mergeSplit='all')
    # print(trainArr.shape, testArr.shape)
    # print(trainDf)
    # print(testDf)
    # ######################################################
    # #test train and test create hdf5 file if not exist
    # first delete if exists:
    btc = DataUtil(dataPath, 'Training', 'Testing')
    resize = 32  # so that it is fast
    for prefix in ['Training', 'Testing']:
        fil = os.path.join(dataPath, prefix + '_' + str(resize) + '.h5')
        if os.path.exists(fil):
            os.remove(fil)

    trainArr, testArr, trainDf, testDf = btc.getTrainTestData(resize)
    print(trainArr.shape, testArr.shape)
    print(trainDf)
    print(testDf)
    # check reading from cache
    trainArr1, testArr1, trainDf1, testDf1 = btc.getTrainTestData(resize)
    print(trainArr1.shape, testArr1.shape)
    print(trainDf1)
    print(testDf1)
    # test write reshaped...
    # imgReshape = 150
    # trainImgs, trainLabels, trainDf = DataUtils.convertToHdf5(dataPath, 'Training',imgReshape)
    # testImgs, testLabels, testDf = DataUtils.convertToHdf5(dataPath, 'Testing',imgReshape)
    # HDF5 path and files
    # trainHdf5File = os.path.join(dataPath, 'Training_256.h5')
    # testHdf5File = os.path.join(dataPath, 'Testing_256.h5')s
    # Read image data
    # tumor_categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    # trainImgs, trainLabels, trainImgSummDf = DataUtils.readHdf5File(trainHdf5File)
    # testImgs, testLabels, testImgSummDf = DataUtils.readHdf5File(testHdf5File)
    # ## print shapes
    # print()
    # print('-' * 80)
    # print(f'Training: images shape {trainImgs.shape}, labels shape {trainLabels.shape}')
    # print(f'Testing: images shape {testImgs.shape}, labels shape {testImgs.shape}')
    # print('-' * 80)
    # labelDistDf = getLabelDistributionDf(dict(train=trainLabels, test=testLabels))
    # print(labelDistDf)
    #
    # # plot image array
    # trIdx = trainImgSummDf.groupby('tumorCategory').sample(n=8).index
    # figFile = os.path.join(figurePath, 'final_01_' + 'trainingImageArray.png')
    # plotImageArr(nRows=4, nCols=8, figSize=(16, 8), imgArr=trainImgs[trIdx],
    #              rowLabels=trainLabels[trIdx])
    # ###### Modelling ##########
    # strategy2 = getCPUorGPUorTPUStrategy()
    # baseModel = MyModel.getModelPaper2020((256, 256, 1), strategy2, 4)
    # print(baseModel.summary())
    # print(getPreProcFx('ZScore'))
