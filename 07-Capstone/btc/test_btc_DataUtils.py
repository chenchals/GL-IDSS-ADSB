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
resize = 16  # for testing purpose


class TestDataUtil(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dataUtil = btc.DataUtil(srcPath, trainDir, testDir)

    def tearDown(self) -> None:
        super().tearDown()
        # remove all files created by tests, if exists
        trFile = os.path.join(srcPath, 'Training_' + str(resize) + '.h5')
        tsFile = os.path.join(srcPath, 'Testing_' + str(resize) + '.h5')
        for fn in [tsFile, trFile]:
            os.remove(fn) if os.path.exists(fn) else ''

    def test_convert_to_hdf5(self) -> None:
        imgArr, imgDf = self.dataUtil.convertToHdf5(trainDir, resize)
        expected = (resize, resize)
        actual = (imgArr.shape[1], imgArr.shape[2])
        self.assertEqual(expected, actual, f'Expected resize {expected} is not same as actual {actual}')

    def test_crop_img(self) -> None:
        img = cv2.imread('../../DataSetBrainTumor/Training/glioma_tumor/gg (11).jpg', cv2.IMREAD_GRAYSCALE)
        cropImg = self.dataUtil.cropImg(img, 25)
        self.assertGreater(img.size, cropImg.size,
                           f'Expected resize {img.size} is not greater than actual {cropImg.size}')

    def test_get_image_dataset(self) -> None:
        imgArr, imgDf = self.dataUtil.getImageDataset(os.path.join(srcPath, 'Testing'), resize)
        self.assertEqual(np.array(imgArr).shape[1], resize,
                         f'Expected resize {resize} is not same as actual {np.array(imgArr).shape[1]}')
        self.assertEqual(np.array(imgArr).shape[0], len(imgDf),
                         f'Expected no. of images {np.array(imgArr).shape[0]} is not same as number of labels {len(imgDf)}')

    def test_read_hdf5var(self) -> None:
        self.dataUtil.convertToHdf5('Testing', resize)
        fn = os.path.join(srcPath, 'Testing_' + str(resize) + '.h5')
        labels = self.dataUtil.readHdf5Var(fn, 'tumorCategory')
        self.assertTrue(labels[0].dtype.type is np.str_, f'type of variable is {labels[0].dtype.type}')
        cropWidth = self.dataUtil.readHdf5Var(fn, 'cropWidth')
        self.assertTrue(cropWidth[0].dtype.type is np.uint16, f'type of variable is {cropWidth[0].dtype.type}')

    def test_write_hdf5file(self) -> None:
        self.dataUtil.convertToHdf5('Testing', resize)
        fn = os.path.join(srcPath, 'Testing_' + str(resize) + '.h5')
        self.assertTrue(os.path.exists(fn), f'Converted file {fn} does not exist')

    def test_read_hdf5file(self) -> None:
        self.dataUtil.convertToHdf5('Testing', resize)
        fn = os.path.join(srcPath, 'Testing_' + str(resize) + '.h5')
        imgArr, imgDf = self.dataUtil.readHdf5File(fn)
        self.assertEqual(imgArr.shape[0], len(imgDf),
                         f'Number of images {imgArr.shape[0]} not equals number of labels {len(imgDf)}')

    def test_get_train_test_data(self) -> None:
        # getDataSet(self, imgResize, mergeSplit=None, forceReread=False)
        trArr, tsArr, trDf, tsDf = self.dataUtil.getTrainTestData(resize)
        self.assertEqual(trArr.shape[0], len(trDf),
                         f'Number of Training-images {trArr.shape[0]} not equal to number of training-labels {len(trDf)}')
        self.assertEqual(tsArr.shape[0], len(tsDf),
                         f'Number of Testing-images {tsArr.shape[0]} not equal to number of testing-labels {len(tsDf)}')
        self.assertGreater(len(trDf), len(tsDf),
                           f'Number of Training-cases {trArr.shape[0]} not Greater than number of testing-cases {len(trDf)}')

    def test_get_data_set_from_cache(self) -> None:
        # getDataSet(self, imgResize, mergeSplit=None, forceReread=False)
        st = timer()
        trArr1, _, _, _ = self.dataUtil.getTrainTestData(resize)
        et_1 = timer() - st
        # second time should get from cache
        st = timer()
        trArr2, _, _, _ = self.dataUtil.getTrainTestData(resize)
        et_2 = timer() - st
        # third time should get from cache
        st = timer()
        trArr3, _, _, _ = self.dataUtil.getTrainTestData(resize)
        et_3 = timer() - st
        self.assertTrue(np.allclose(trArr1, trArr2), f'Train Image arrays are NOT equal between 1st and 2nd call')
        self.assertTrue(np.allclose(trArr2, trArr3), f'Train Image arrays are NOT equal between 2nd and 3rd call')
        self.assertGreater(et_1, et_2,
                           f'Time to get data FIRST time {et_1} is NOT GREATER than SECOND time {et_2}')
        # assert read within a millisecond resolution
        self.assertAlmostEqual(et_2, et_3, delta=1e-3,
                               msg=f'Getting data from CACHE:  Second time {et_2} not EQUAL to Third time {et_3}')


if __name__ == 'main':
    unittest.main()
    # # initialize suite
    # loader = unittest.TestLoader()
    # suite = unittest.TestSuite()
    #
    # # add tests to suite
    # suite.addTests(loader.loadTestsFromName(TestDataUtil))
    #
    # # initialize a runner, pass it your suite and run it
    # runner = unittest.TextTestRunner(verbosity=3)
    # result = runner.run(suite)
