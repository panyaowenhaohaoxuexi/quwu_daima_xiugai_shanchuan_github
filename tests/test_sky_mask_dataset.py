import os
import shutil
import tempfile
import unittest

import torch
from PIL import Image

from data.data_loader import MultiModalCLIPLoader, MultiModalHazeDataset, TestDataset


def save_rgb(path, size=(16, 16), color=(180, 190, 200)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path)


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


class MultiLevelHazeDatasetCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.hazy_dir = os.path.join(self.tmp, "hazy")
        self.ir_dir = os.path.join(self.tmp, "ir")
        self.clear_dir = os.path.join(self.tmp, "clear")
        for path in (self.hazy_dir, self.ir_dir, self.clear_dir):
            os.makedirs(path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def make_supervised(self, **kwargs):
        return MultiModalHazeDataset(
            hazy_visible_path=self.hazy_dir,
            infrared_path=self.ir_dir,
            clear_visible_path=self.clear_dir,
            train=False,
            size="full",
            **kwargs,
        )

    def make_test_dataset(self, **kwargs):
        return TestDataset(
            hazy_visible_path=self.hazy_dir,
            infrared_path=self.ir_dir,
            clear_visible_path=self.clear_dir,
            size="full",
            **kwargs,
        )

    def make_clip_loader(self, **kwargs):
        return MultiModalCLIPLoader(
            hazy_visible_path=self.hazy_dir,
            infrared_path=self.ir_dir,
            train=False,
            size="full",
            **kwargs,
        )

    def create_png_triplet(self, name="0001.png"):
        save_rgb(os.path.join(self.hazy_dir, name))
        save_rgb(os.path.join(self.ir_dir, name))
        save_rgb(os.path.join(self.clear_dir, name))

    def create_multilevel_png_triplet(self, name="0001.png"):
        for level in ("mist", "middle", "dense"):
            save_rgb(os.path.join(self.hazy_dir, level, name))
        save_rgb(os.path.join(self.ir_dir, name))
        save_rgb(os.path.join(self.clear_dir, name))

    def test_flat_png_dataset_uses_auto_format_by_default(self):
        self.create_png_triplet()
        dataset = self.make_supervised()
        self.assertEqual(dataset.layout, "flat")
        self.assertEqual(dataset.image_list, ["0001.png"])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(dataset[0]), 3)

    def test_flat_png_dataset_returns_sky_mask_with_auto_format(self):
        self.create_png_triplet()
        sky_dir = os.path.join(self.tmp, "sky_mask")
        save_rgb(os.path.join(sky_dir, "0001_sky.png"))
        dataset = self.make_supervised(
            sky_mask_path=sky_dir,
            use_sky_mask=True,
            sky_mask_suffix="_sky",
            sky_mask_ext=".png",
        )
        self.assertEqual(len(dataset[0]), 4)

    def test_multilevel_png_dataset_expands_levels_in_stable_order(self):
        self.create_multilevel_png_triplet()
        dataset = self.make_supervised()
        self.assertEqual(dataset.layout, "multi_level")
        self.assertEqual(dataset.level_counts, {"mist": 1, "middle": 1, "dense": 1})
        self.assertEqual(dataset.image_list, ["mist_0001.png", "middle_0001.png", "dense_0001.png"])
        self.assertEqual([sample["haze_level"] for sample in dataset.samples], ["mist", "middle", "dense"])
        self.assertEqual(len(dataset), 3)

    def test_auto_format_pairs_hazy_png_with_ir_jpg_and_clear_png(self):
        save_rgb(os.path.join(self.hazy_dir, "mist", "0001.png"))
        save_rgb(os.path.join(self.ir_dir, "0001.jpg"))
        save_rgb(os.path.join(self.clear_dir, "0001.png"))
        dataset = self.make_supervised()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(os.path.basename(dataset.samples[0]["ir_path"]), "0001.jpg")
        self.assertEqual(os.path.basename(dataset.samples[0]["clear_path"]), "0001.png")

    def test_explicit_jpg_format_ignores_png_and_raises_when_empty(self):
        self.create_png_triplet()
        with self.assertRaises(RuntimeError):
            self.make_supervised(format=".jpg")

    def test_auto_format_keeps_same_stem_hazy_duplicates(self):
        save_rgb(os.path.join(self.hazy_dir, "0001.jpg"))
        save_rgb(os.path.join(self.hazy_dir, "0001.png"))
        save_rgb(os.path.join(self.ir_dir, "0001.jpg"))
        save_rgb(os.path.join(self.clear_dir, "0001.jpg"))
        dataset = self.make_supervised()
        self.assertEqual(dataset.image_list, ["0001.jpg", "0001.png"])
        self.assertEqual(len(dataset), 2)

    def test_multilevel_test_dataset_returns_unique_output_names(self):
        self.create_multilevel_png_triplet()
        dataset = self.make_test_dataset()
        returned_names = [dataset[i][3] for i in range(len(dataset))]
        self.assertEqual(returned_names, ["mist_0001.png", "middle_0001.png", "dense_0001.png"])

    def test_multilevel_clip_loader_keeps_two_item_return_format(self):
        self.create_multilevel_png_triplet()
        dataset = self.make_clip_loader()
        self.assertEqual(dataset.layout, "multi_level")
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset[0]), 2)

    def test_missing_ir_skips_partial_samples_and_counts_missing(self):
        save_rgb(os.path.join(self.hazy_dir, "0001.png"))
        save_rgb(os.path.join(self.hazy_dir, "0002.png"))
        save_rgb(os.path.join(self.ir_dir, "0001.png"))
        save_rgb(os.path.join(self.clear_dir, "0001.png"))
        save_rgb(os.path.join(self.clear_dir, "0002.png"))
        dataset = self.make_supervised()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.missing_ir, 1)
        self.assertEqual(dataset.image_list, ["0001.png"])

    def test_all_missing_ir_raises_runtime_error(self):
        save_rgb(os.path.join(self.hazy_dir, "0001.png"))
        save_rgb(os.path.join(self.clear_dir, "0001.png"))
        with self.assertRaises(RuntimeError):
            self.make_supervised()

    def test_all_missing_clear_raises_runtime_error_for_supervised_dataset(self):
        save_rgb(os.path.join(self.hazy_dir, "0001.png"))
        save_rgb(os.path.join(self.ir_dir, "0001.png"))
        with self.assertRaises(RuntimeError):
            self.make_supervised()

    def test_missing_clear_skips_partial_samples_and_counts_missing(self):
        save_rgb(os.path.join(self.hazy_dir, "0001.png"))
        save_rgb(os.path.join(self.hazy_dir, "0002.png"))
        save_rgb(os.path.join(self.ir_dir, "0001.png"))
        save_rgb(os.path.join(self.ir_dir, "0002.png"))
        save_rgb(os.path.join(self.clear_dir, "0001.png"))
        dataset = self.make_supervised()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.missing_clear, 1)
        self.assertEqual(dataset.image_list, ["0001.png"])

    def test_clip_loader_ignores_missing_clear(self):
        save_rgb(os.path.join(self.hazy_dir, "0001.png"))
        save_rgb(os.path.join(self.ir_dir, "0001.png"))
        dataset = self.make_clip_loader()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.missing_clear, 0)


if __name__ == "__main__":
    unittest.main()
