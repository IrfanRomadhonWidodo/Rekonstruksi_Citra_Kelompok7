import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io, color, img_as_float
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class ImageReconstructor:
    def __init__(self, image_path):
        try:
            # Baca citra & konversi ke grayscale float
            self.original = io.imread(image_path)
            if len(self.original.shape) == 3:
                self.original = color.rgb2gray(self.original)
            self.original = img_as_float(self.original)
        except FileNotFoundError:
            raise ValueError(f"Citra '{image_path}' tidak ditemukan.")
        
        # Precompute SVD sekali saja (hemat waktu)
        self.U, self.S, self.Vt = np.linalg.svd(self.original, full_matrices=False)
        
        # Setup awal
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.svd_components = 50
        self.dft_cutoff = 0.1
        
        self.setup_plots()
        self.setup_sliders()
        self.update_reconstruction(None)
        
    def setup_plots(self):
        """Setup tiga panel plot: asli, SVD, DFT"""
        # Citra asli
        self.axes[0].imshow(self.original, cmap='gray')
        self.axes[0].set_title('Citra Asli')
        self.axes[0].axis('off')
        
        # Placeholder untuk SVD & DFT
        self.svd_plot = self.axes[1].imshow(self.original, cmap='gray')
        self.axes[1].set_title('Rekonstruksi SVD')
        self.axes[1].axis('off')
        
        self.dft_plot = self.axes[2].imshow(self.original, cmap='gray')
        self.axes[2].set_title('Rekonstruksi DFT')
        self.axes[2].axis('off')
        
    def setup_sliders(self):
        """Setup slider untuk parameter rekonstruksi"""
        plt.subplots_adjust(bottom=0.2)  # lebih lega untuk slider
        
        # Slider SVD
        ax_svd = plt.axes([0.15, 0.05, 0.25, 0.03])
        self.svd_slider = Slider(
            ax=ax_svd,
            label='Komponen SVD',
            valmin=1,
            valmax=min(self.original.shape),
            valinit=self.svd_components,
            valstep=1
        )
        self.svd_slider.on_changed(self.update_reconstruction)
        
        # Slider DFT
        ax_dft = plt.axes([0.6, 0.05, 0.25, 0.03])
        self.dft_slider = Slider(
            ax=ax_dft,
            label='Cutoff DFT',
            valmin=0.01,
            valmax=0.5,
            valinit=self.dft_cutoff,
            valstep=0.01
        )
        self.dft_slider.on_changed(self.update_reconstruction)
        
    def svd_reconstruct(self, k):
        """Rekonstruksi citra menggunakan SVD dengan k komponen"""
        S_reduced = np.diag(self.S[:k])
        reconstructed = self.U[:, :k] @ S_reduced @ self.Vt[:k, :]
        return reconstructed
    
    def dft_reconstruct(self, cutoff):
        """Rekonstruksi citra menggunakan DFT dengan low-pass filter"""
        dft = fft2(self.original)
        dft_shifted = fftshift(dft)
        
        rows, cols = self.original.shape
        crow, ccol = rows // 2, cols // 2
        max_radius = np.sqrt(crow**2 + ccol**2)
        cutoff_radius = cutoff * max_radius
        
        y, x = np.ogrid[:rows, :cols]
        mask = ((x - ccol)**2 + (y - crow)**2) <= cutoff_radius**2
        
        dft_filtered = dft_shifted * mask
        idft = ifftshift(dft_filtered)
        reconstructed = np.abs(ifft2(idft))
        return reconstructed
    
    def update_reconstruction(self, val):
        """Update rekonstruksi berdasarkan nilai slider"""
        self.svd_components = int(self.svd_slider.val)
        self.dft_cutoff = self.dft_slider.val
        
        # Rekonstruksi SVD
        svd_result = self.svd_reconstruct(self.svd_components)
        self.svd_plot.set_data(svd_result)
        self.svd_plot.set_clim(vmin=svd_result.min(), vmax=svd_result.max())
        
        # Rekonstruksi DFT
        dft_result = self.dft_reconstruct(self.dft_cutoff)
        self.dft_plot.set_data(dft_result)
        self.dft_plot.set_clim(vmin=dft_result.min(), vmax=dft_result.max())
        
        # Update judul
        self.axes[1].set_title(f'SVD (k={self.svd_components})')
        self.axes[2].set_title(f'DFT (cutoff={self.dft_cutoff:.2f})')
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Tampilkan GUI"""
        plt.show()


# Penggunaan
if __name__ == "__main__":
    image_path = "example.jpg"  # Ganti dengan path citra Anda
    reconstructor = ImageReconstructor(image_path)
    reconstructor.show()
