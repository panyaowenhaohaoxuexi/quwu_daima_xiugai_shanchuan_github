import os
import shutil
import tempfile
import unittest

import torch
from PIL import Image

from data.data_loader import MultiModalHazeDataset, TestDataset


class SkyMaskDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.hazy_dir = os.path.join(self.tmp, "hazy")
        self.ir_dir = os.path.join(self.tmp, "ir")
        self.clear_dir = os.path.join(self.tmp, "clear")
        self.sky_dir = os.path.join(self.tmp, "sky_mask")
        for path in (self.hazy_dir, self.ir_dir, self.clear_dir, self.sky_dir):
            os.makedirs(path, exist_ok=True)

        rgb = Image.new("RGB", (16, 16), (180, 190, 200))
        rgb.save(os.path.join(self.hazy_dir, "0001.jpg"))
        rgb.save(os.path.join(self.ir_dir, "0001.jpg"))
        rgb.save(os.path.join(self.clear_dir, "0001.jpg"))

        mask = Image.new("L", (16, 16), 0)
        for y in range(8):
            for x in range(16):
                mask.putpixel((x, y), 255)
        mask.save(os.path.join(self.sky_dir, "0001_sky.png"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def make_dataset(self, **kwargs):
        return MultiModalHazeDataset(
            hazy_visible_path=self.hazy_dir,
            infrared_path=self.ir_dir,
            clear_visible_path=self.clear_dir,
            train=False,
            size="full",
            format=".jpg",
            **kwargs,
        )

    def test_returns_three_items_when_sky_mask_disabled(self):
        sample = self.make_dataset(use_sky_mask=False)[0]
        self.assertEqual(len(sample), 3)

    def test_returns_binary_sky_mask_when_enabled(self):
        sample = self.make_dataset(
            sky_mask_path=self.sky_dir,
            use_sky_mask=True,
            sky_mask_suffix="_sky",
            sky_mask_ext=".png",
        )[0]
        self.assertEqual(len(sample), 4)
        sky_mask = sample[3]
        self.assertEqual(tuple(sky_mask.shape), (1, 16, 16))
        self.assertTrue(torch.equal(torch.unique(sky_mask), torch.tensor([0.0, 1.0])))

    def test_missing_sky_mask_falls_back_to_zero_when_not_required(self):
        os.remove(os.path.join(self.sky_dir, "0001_sky.png"))
        sample = self.make_dataset(
            sky_mask_path=self.sky_dir,
            use_sky_mask=True,
            require_sky_mask=False,
        )[0]
        self.assertEqual(len(sample), 4)
        self.assertEqual(float(sample[3].sum()), 0.0)

    def test_missing_sky_mask_raises_when_required(self):
        os.remove(os.path.join(self.sky_dir, "0001_sky.png"))
        dataset = self.make_dataset(
            sky_mask_path=self.sky_dir,
            use_sky_mask=True,
            require_sky_mask=True,
        )
        with self.assertRaises(FileNotFoundError):
            _ = dataset[0]


class SkyMaskTestDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.hazy_dir = os.path.join(self.tmp, "hazy")
        self.ir_dir = os.path.join(self.tmp, "ir")
        self.clear_dir = os.path.join(self.tmp, "clear")
        self.sky_dir = os.path.join(self.tmp, "sky_mask")
        for path in (self.hazy_dir, self.ir_dir, self.clear_dir, self.sky_dir):
            os.makedirs(path, exist_ok=True)

        rgb = Image.new("RGB", (16, 16), (180, 190, 200))
        rgb.save(os.path.join(self.hazy_dir, "0001.jpg"))
        rgb.save(os.path.join(self.ir_dir, "0001.jpg"))
        rgb.save(os.path.join(self.clear_dir, "0001.jpg"))

        mask = Image.new("L", (16, 16), 0)
        for y in range(8):
            for x in range(16):
                mask.putpixel((x, y), 255)
        mask.save(os.path.join(self.sky_dir, "0001_sky.png"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def make_dataset(self, **kwargs):
        return TestDataset(
            hazy_visible_path=self.hazy_dir,
            infrared_path=self.ir_dir,
            clear_visible_path=self.clear_dir,
            size="full",
            format=".jpg",
            **kwargs,
        )

    def test_returns_four_items_when_test_sky_mask_disabled(self):
        sample = self.make_dataset(use_sky_mask=False)[0]
        self.assertEqual(len(sample), 4)
        self.assertEqual(sample[3], "0001.jpg")

    def test_returns_binary_sky_mask_before_image_name_when_test_enabled(self):
        sample = self.make_dataset(
            sky_mask_path=self.sky_dir,
            use_sky_mask=True,
            sky_mask_suffix="_sky",
            sky_mask_ext=".png",
        )[0]
        self.assertEqual(len(sample), 5)
        sky_mask = sample[3]
        self.assertEqual(sample[4], "0001.jpg")
        self.assertEqual(tuple(sky_mask.shape), (1, 16, 16))
        self.assertTrue(torch.equal(torch.unique(sky_mask), torch.tensor([0.0, 1.0])))

    def test_missing_test_sky_mask_falls_back_to_zero_when_not_required(self):
        os.remove(os.path.join(self.sky_dir, "0001_sky.png"))
        sample = self.make_dataset(
            sky_mask_path=self.sky_dir,
            use_sky_mask=True,
            require_sky_mask=False,
        )[0]
        self.assertEqual(len(sample), 5)
        self.assertEqual(float(sample[3].sum()), 0.0)
        self.assertEqual(sample[4], "0001.jpg")

    def test_missing_test_sky_mask_raises_when_required(self):
        os.remove(os.path.join(self.sky_dir, "0001_sky.png"))
        dataset = self.make_dataset(
            sky_mask_path=self.sky_dir,
            use_sky_mask=True,
            require_sky_mask=True,
        )
        with self.assertRaises(FileNotFoundError):
            _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
