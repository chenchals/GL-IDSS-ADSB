import unittest
import os
import numpy as np
from unittest import TestCase
from timeit import default_timer as timer
import btc.btc_helpers as btc
import cv2

srcPath = '../../DataSetBrainTumor'
trainDir = 'Training'
testDir = 'Testing'
resize = 16 # for testing purpose


class TestBTCDataUtil(TestCase):

    def test_convert_to_hdf5(self):
        btcDataUtil = btc.BTCDataUtil(srcPath,trainDir,testDir)
        imgArr,imgDf = btcDataUtil.convertToHdf5(btcDataUtil.trainDir,resize)
        expected = (resize,resize)
        actual = (imgArr.shape[1],imgArr.shape[2])
        self.assertEqual(expected,actual,f'Expected resize {expected} is not same as actual {actual}')

    def test_crop_img(self):
        btcDataUtil = btc.BTCDataUtil(srcPath,trainDir,testDir)
        img = cv2.imread('../../DataSetBrainTumor/Training/glioma_tumor/gg (11).jpg',cv2.IMREAD_GRAYSCALE)
        cropImg = btcDataUtil.cropImg(img,25)
        self.assertGreater(img.size,cropImg.size,f'Expected resize {img.size} is not greater than actual {cropImg.size}')

    def test_get_image_dataset(self):
        btcDataUtil = btc.BTCDataUtil(srcPath, trainDir, testDir)
        imgArr,imgDf = btcDataUtil.getImageDataset(os.path.join(srcPath,'Testing'),resize)
        self.assertEqual(np.array(imgArr).shape[1],resize,
                         f'Expected resize {resize} is not same as actual {np.array(imgArr).shape[1]}')
        self.assertEqual(np.array(imgArr).shape[0], len(imgDf),
                         f'Expected no. of images {np.array(imgArr).shape[0]} is not same as number of labels {len(imgDf)}')

    def test_read_hdf5var(self):
        self.fail()

    def test_write_hdf5file(self):
        self.fail()

    def test_read_hdf5file(self):
        self.fail()

    def test_get_train_test_data(self):
        # getDataSet(self, imgResize, mergeSplit=None, forceReread=False)
        btcDataUtil = btc.BTCDataUtil(srcPath,trainDir,testDir)
        trArr,tsArr,trDf,tsDf = btcDataUtil.getTrainTestData(256)
        self.assertEqual(trArr.shape[0], len(trDf),
                         f'Number of Training-images {trArr.shape[0]} not equal to number of training-labels {len(trDf)}')
        self.assertEqual(tsArr.shape[0], len(tsDf),
                         f'Number of Testing-images {tsArr.shape[0]} not equal to number of testing-labels {len(tsDf)}')
        self.assertGreater(len(trDf),len(tsDf),
                           f'Number of Training-cases {trArr.shape[0]} not Greater than number of testing-cases {len(trDf)}')

    def test_get_data_set_from_cache(self):
        # getDataSet(self, imgResize, mergeSplit=None, forceReread=False)
        btcDataUtil = btc.BTCDataUtil(srcPath,trainDir,testDir)
        st = timer()
        trArr,tsArr,trDf,tsDf = btcDataUtil.getTrainTestData(256)
        firstDuration = timer()-st
        # reget - should get from cache
        st = timer()
        trArr, tsArr, trDf, tsDf = btcDataUtil.getTrainTestData(256)
        secondDuration = timer() - st
        st = timer()
        trArr, tsArr, trDf, tsDf = btcDataUtil.getTrainTestData(256)
        thirdDuration = timer() - st
        self.assertGreater(firstDuration,secondDuration,
                           f'Time to get data FIRST time {firstDuration} is NOT GREATER than SECOND time {secondDuration}')
        # assert read within a millisecond resolution
        self.assertAlmostEqual(secondDuration,thirdDuration,delta=1e-3,
                               msg=f'Getting data from CACHE:  Second time {secondDuration} not EQUAL to Third time {thirdDuration}')


