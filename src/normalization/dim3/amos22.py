import nibabel as nb
import numpy as np
from pathlib import Path
from skimage.restoration import denoise_wavelet
from skimage.transform import resize

class NormalizeDataset:

    def __init__(self, target_size=(256, 256, 128)):
        self.subset = ['Tr', 'Va', 'Ts']
        self.target_size = target_size
        

    def __call__(self, src, dest):
        src_path = Path(src)
        dest_path = Path(dest)

        for s in self.subset:
            img_dir = src_path / f"images{s}"
            lbl_dir = src_path / f"labels{s}"
            
            out_img_dir = dest_path / f"images{s}"
            out_lbl_dir = dest_path / f"labels{s}"
            
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            files = sorted([f for f in img_dir.glob("*.nii.gz") if not f.name.startswith('.')])
            
            for file_path in files:
                print(f"Processing {file_path.name}...")
                
                img_nii = nb.load(file_path)
                img_data = img_nii.get_fdata()

                denoised = self.denoise_image(img_data)

                normalized = self.normalize_intensity(denoised)

                resized_img = self.resize_image(normalized, self.target_size)
                
                new_img = nb.Nifti1Image(resized_img.astype(np.float32), img_nii.affine)
                nb.save(new_img, out_img_dir / file_path.name)

                if lbl_dir.exists():
                    lbl_path = lbl_dir / file_path.name
                    if lbl_path.exists():
                        lbl_nii = nb.load(lbl_path)
                        lbl_data = lbl_nii.get_fdata()
                        resized_lbl = resize(lbl_data, self.target_size, order=0, anti_aliasing=False)
                        new_lbl = nb.Nifti1Image(resized_lbl.astype(np.uint8), lbl_nii.affine)
                        nb.save(new_lbl, out_lbl_dir / file_path.name)

    @staticmethod
    def denoise_image(image):
        return denoise_wavelet(
            image=image,
            mode='soft',
            method='BayesShrink',
            rescale_sigma=True,
            channel_axis=None
        )
    
    @staticmethod
    def resize_image(image, target_size, is_label=False):
        return resize(
            image=image, 
            output_shape=target_size, 
            order=0 if is_label else 3, 
            mode='reflect', 
            anti_aliasing=not is_label    
        )

    @staticmethod
    def normalize_intensity(image):
        p05, p995 = np.percentile(image, [0.5, 99.5])
        return np.clip((image - p05) / (p995 - p05), 0, 1)
    
if __name__ == "__main__":
    SRC = '/home/alfredo-vr/Documentos/grado/tfg/data/amos22'
    DST = '/home/alfredo-vr/Documentos/grado/tfg/data/amos22_preprocessed'

    NormalizeDataset()(src=SRC, dest=DST)

