# preprocess.py

import os
import numpy as np
from openslide import OpenSlide
from skimage import io, transform, exposure

class WSIPreprocessor:
    def __init__(self, patch_size=512, thumbnail_size=50, sample_ratio=0.1):
        self.patch_size = patch_size
        self.thumbnail_size = thumbnail_size
        self.sample_ratio = sample_ratio

    def process_slide(self, svs_path, output_dir):
        slide = OpenSlide(svs_path)
        wsi_id = os.path.basename(svs_path).split('.')[0]
        os.makedirs(f"{output_dir}/{wsi_id}", exist_ok=True)

        level = 0
        dims = slide.level_dimensions[level]
        if dims[0] < self.patch_size or dims[1] < self.patch_size:
            print(f"WSI {wsi_id} is too small ({dims[0]}x{dims[1]}) for patch size {self.patch_size}. Skipping.")
            return None

        num_patches = int((dims[0]*dims[1]*self.sample_ratio) / (self.patch_size**2))
        coords = np.random.randint(0, [dims[0]-self.patch_size, dims[1]-self.patch_size], size=(num_patches, 2))

        valid_patches = []
        for i, (x, y) in enumerate(coords):
            try:
                patch = slide.read_region((x, y), level, (self.patch_size, self.patch_size))
                patch = np.array(patch.convert('RGB'))

                if exposure.is_low_contrast(patch, fraction_threshold=0.3):
                    continue

                try:
                    io.imsave(f"{output_dir}/{wsi_id}/patch_{i}.png", patch)
                except Exception as e:
                    print(f"Error saving patch {i} for {wsi_id}: {e}")
                    continue

                thumbnail = transform.resize(patch, (self.thumbnail_size,)*2)
                valid_patches.append(thumbnail)
            except Exception as e:
                print(f"Error processing patch {i} for {wsi_id}: {e}")
                continue

        return np.stack(valid_patches) if valid_patches else None
