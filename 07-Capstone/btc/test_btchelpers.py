from unittest import TestCase
from timeit import default_timer as timer
from btc.btc_helpers import BTCDataUtil
srcPath = '../../DataSetBrainTumor'
trainDir = 'Training'
testDir = 'Testing'
resize = 16 # for testing purpose


class TestBTCDataUtil(TestCase):

    def test_convert_to_hdf5(self):
        btcDataUtil = BTCDataUtil(srcPath,trainDir,testDir)
        imgArr,imgDf = btcDataUtil.convertToHdf5(btcDataUtil.trainDir,resize)
        self.assertEqual((resize,resize),(imgArr.shape[1],imgArr.shape[2]))

    def test_crop_img(self):
        self.fail()

    def test_get_image_dataset(self):
        self.fail()

    def test_read_hdf5var(self):
        self.fail()

    def test_write_hdf5file(self):
        self.fail()

    def test_read_hdf5file(self):
        self.fail()

    def test_get_data_set(self):
        # getDataSet(self, imgResize, mergeSplit=None, forceReread=False)
        btc = BTCDataUtil(srcPath,trainDir,testDir)
        trArr,tsArr,trDf,tsDf = btc.getDataSet(256)
        self.assertEqual(trArr.shape[0], len(trDf))
        self.assertEqual(tsArr.shape[0], len(tsDf))
        self.assertGreater(len(trDf),len(tsDf))
        self.assertGreater(trArr.shape[0],tsArr.shape[0])

    def test_get_data_set_from_cache(self):
        # getDataSet(self, imgResize, mergeSplit=None, forceReread=False)

        btc = BTCDataUtil(srcPath,trainDir,testDir)
        st = timer()
        trArr,tsArr,trDf,tsDf = btc.getDataSet(256)
        firstDuration = timer()-st
        # reget - should get from cache
        st = timer()
        trArr, tsArr, trDf, tsDf = btc.getDataSet(256)
        secondDuration = timer() - st
        st = timer()
        trArr, tsArr, trDf, tsDf = btc.getDataSet(256)
        thirdDuration = timer() - st
        self.assertGreater(firstDuration,secondDuration)
        # assert read within a millisecond resolution
        self.assertAlmostEqual(secondDuration,thirdDuration,delta=1e-3)


