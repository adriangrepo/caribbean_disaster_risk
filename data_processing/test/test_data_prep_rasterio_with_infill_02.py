import unittest
import numpy as np
from PIL import Image
import os,sys,inspect

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_prep_rasterio_with_infill_02 import crop_TL, crop_BL, crop_TR, crop_BR, mask_TL, mask_BL, mask_TR, mask_BR

class TestCropPad(unittest.TestCase):
    def test_crop_TL(self):
        print('>>test_crop_TL')
        height=16
        width=12
        img=np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img_TL=crop_TL(img, height, width)
        np_TL=np.array(img_TL)
        #size is width x height or x * y, while matrix ordering is rows x columns
        w, h = img_TL.size
        self.assertEqual(h, height)
        self.assertEqual(w, width)
        self.assertEqual(np_TL.shape, (height, width, 3))

    def test_mask_TL(self):
        print('>>test_mask_TL')
        height=16
        width=12
        shape_mask = np.zeros((height, width), dtype=bool)
        shape_mask[0:12,0:8]=True
        m_TL = mask_TL(height, width, shape_mask)
        #(3, 4, 12, 16)
        np.testing.assert_array_equal(m_TL[0:12-4, 0:8-3], shape_mask[4:12, 3:8])
        print(f'mask_TL: {m_TL}')
        self.assertEqual((12-4)*(8-3), np.count_nonzero(m_TL))


    def test_crop_BL(self):
        print('>>test_crop_BL')
        height = 16
        width = 12
        shape_mask = np.zeros((height, width), dtype=bool)
        shape_mask[0:12, 0:8] = True
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img_BL= crop_BL(img, height, width)
        np_BL = np.array(img_BL)
        # size is width x height or x * y, while matrix ordering is rows x columns
        w, h = img_BL.size
        self.assertEqual(h, height)
        self.assertEqual(w, width)
        self.assertEqual(np_BL.shape, (height, width, 3))

    def test_mask_BL(self):
        print('>>test_mask_BL')
        height=16
        width=12
        shape_mask = np.zeros((height, width), dtype=bool)
        shape_mask[0:12,0:8]=True
        m_BL = mask_BL(height, width, shape_mask)
        print(f'mask_BL: {m_BL}')
        self.assertEqual(60, np.count_nonzero(m_BL))

    def test_crop_TR(self):
        print('>>test_crop_TR')
        height = 16
        width = 12

        img = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img_TR= crop_TR(img, height, width)
        np_TR = np.array(img_TR)
        # size is width x height or x * y, while matrix ordering is rows x columns
        w, h = img_TR.size
        self.assertEqual(h, height)
        self.assertEqual(w, width)
        self.assertEqual(np_TR.shape, (height, width, 3))
        # (3, 4, 12, 16)
        #np.testing.assert_array_equal(mask_TR[0:12 - 4, 0:8 - 3], shape_mask[4:12, 3:8])

    def test_mask_TR(self):
        print('>>test_mask_TR')
        height=16
        width=12
        shape_mask = np.zeros((height, width), dtype=bool)
        shape_mask[0:12, 0:8] = True
        m_TR = mask_TR(height, width, shape_mask)
        print(f'mask_TR: {m_TR}')
        self.assertEqual(64, np.count_nonzero(m_TR))

    def test_crop_BR(self):
        print('>>test_crop_BR')
        height = 16
        width = 12
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img_BR = crop_BR(img, height, width)
        np_BR = np.array(img_BR)
        # size is width x height or x * y, while matrix ordering is rows x columns
        w, h = img_BR.size
        self.assertEqual(h, height)
        self.assertEqual(w, width)
        self.assertEqual(np_BR.shape, (height, width, 3))

    def test_mask_BR(self):
        print('>>test_mask_BR')
        height=16
        width=12
        shape_mask = np.zeros((height, width), dtype=bool)
        shape_mask[0:12, 0:8] = True
        m_BR = mask_BR(height, width, shape_mask)
        print(f'mask_BR: {m_BR}')
        self.assertEqual(96, np.count_nonzero(m_BR))


if __name__ == '__main__':
    unittest.main()
