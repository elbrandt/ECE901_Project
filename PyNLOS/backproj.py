import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import sparse
from scipy.fft import fft, ifft
from tqdm import tqdm
import time
import PIL

class NLOSScene():
    def __init__(self, matfile: str):
        self.matfile = matfile
        self.hf = h5py.File(matfile,'r')
        print(f"Loading scene file {matfile}...")
        self.hf.visit(self.assign)
        print(f"Done loading.")
        
    def assign(self, path: str):
        if path[0] == '#':
            return
        path_split = path.split('/')
        print(f"Reading {path} into {path_split[-1]}...")
        setattr(self, path_split[-1], np.squeeze(np.array(self.hf.get(path))).transpose())

    def phasor_pulse_convolution(self, wavelength: float, cycle_times: int, show_plot: bool = False, use_cache: bool = True):

        if use_cache:
            # check if we have pre-computed this wavelength and times for this request
            tm_start = time.perf_counter()
            fname = os.path.join(os.path.dirname(self.matfile),
                        f"{os.path.basename(self.matfile)[:-4]}_{wavelength:.6f}_{cycle_times}.npy")
            if os.path.exists(fname):
                with open(fname, 'rb') as f:
                    P = np.load(f)
                    tm_stop = time.perf_counter()
                    print(f"loaded from cached file in {tm_stop-tm_start:.6f} seconds")
                    return P

        pulse_size = np.round((cycle_times * wavelength) / self.deltat)
        cycle_size = np.round(wavelength / self.deltat)
        sigma = cycle_times * wavelength / 6
        t = self.deltat * (np.arange(1,pulse_size+1) - pulse_size / 2)

        gaussian_pulse = np.exp(-(t * t) / (2 * sigma * sigma))
        # use the range(1,pulse_size+1) here to match matlab, but (0,pulse_size) is perhaps more correct
        cplx_wave = np.exp((0+1j) * 2*np.pi*(1/cycle_size * np.arange(1,pulse_size+1)))
        cplx_pulse = cplx_wave * gaussian_pulse

        if show_plot:
            fig, ax = plt.subplots()
            ax.plot(np.real(cplx_pulse), 'b', label='cos kernel')
            ax.plot(np.imag(cplx_pulse), 'g', label='sin kernel')
            ax.plot(gaussian_pulse, 'r', label='gauss envelope')
            ax.legend()
            plt.show()

        H = self.data
        P_cplx = np.empty(H.shape, dtype=np.complex)

        method = signal.choose_conv_method(H[0,:], cplx_pulse, mode='same')

        tm_start = time.perf_counter()
        for p in tqdm(range(P_cplx.shape[0])):
            P_cplx[p,:] = signal.convolve(H[p,:], cplx_pulse, 'same', method=method)
        tm_stop = time.perf_counter()
        P = P_cplx
        print(f'complex primal convolution took: {tm_stop-tm_start:.6f}')

        if use_cache:
            assert(fname is not None)
            with open(fname, 'wb') as f:
                np.save(f, P)

        return P

class NLOSReconstruction():
    def __init__(self, scene: NLOSScene, P: np.array, gridsize: float):
        self.scene = scene
        self.P = P
        self.gridsize = gridsize
        self._prep_voxels()
        self.d1 = np.linalg.norm(self.scene.laserPos - self.scene.laserOrigin, axis=1)
        self.d4 = np.linalg.norm(self.scene.cameraPos - self.scene.cameraOrigin)
        self._reconstruct()

    def _prep_voxels(self):
        self.voxel_dims = np.ceil(np.abs(self.scene.minimalpos - self.scene.maximalpos) / self.gridsize).astype(int)
        self.nvoxels = np.prod(self.voxel_dims)
        pts = np.empty((self.nvoxels, 3))
        
        x = np.arange(start=self.scene.minimalpos[0], stop=self.scene.maximalpos[0], step=self.gridsize)
        y = np.arange(start=self.scene.minimalpos[1], stop=self.scene.maximalpos[1], step=self.gridsize)
        z = np.arange(start=self.scene.minimalpos[2], stop=self.scene.maximalpos[2], step=self.gridsize)
        X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

        self.voxel_pts = np.reshape(np.stack([X,Y,Z], axis=-1).flatten(), (-1, 3))
        print(f'Reconstruction will have {self.nvoxels} voxels ({x.shape[0]} x {y.shape[0]} x {z.shape[0]})')

    def _reconstruct(self):
        print('Starting reconstruction...')
        tm_start = time.perf_counter()
        self.voxels = np.zeros(self.nvoxels, dtype=np.complex)
        nlaserPos = self.scene.laserPos.shape[0]
        ntimes = self.scene.data.shape[1]
        linP = np.reshape(self.P, -1)
        offsets = self.P.shape[1] * np.arange(self.P.shape[0])
        for v in tqdm(np.arange(self.nvoxels)):
            intensity = 0
            v_loc = self.voxel_pts[v,:] 
            d2 = np.linalg.norm(v_loc - self.scene.laserPos, axis=1)
            d340 = np.linalg.norm(v_loc - self.scene.cameraPos) + self.d4 - self.scene.t0
            d = self.d1 + d2 + d340
            t = np.round((d / self.scene.deltat) + 1).astype(np.int32)
            t += offsets
            intensity = sum(linP[t])
            self.voxels[v] = intensity
        self.voxels = np.reshape(np.abs(self.voxels), self.voxel_dims)


        tm_end = time.perf_counter()
        print(f'Reconstructed {self.nvoxels} in {tm_end-tm_start:.6f} sec')

def main():
    parser = argparse.ArgumentParser(description='Performs NLOS Backprojection to reconstruct a NLOS scene')
    parser.add_argument('infile', help='the *.mat file that contains the input dataset')
    parser.add_argument('-r', '--resolution', type=float, default=0.16, help='the reconstruction resolution (in meters)')
    parser.add_argument('-o', '--outfile', required=False, default=None, help='filename to save the reconstructed cube (in numpy format)')
    parser.add_argument('-np', '--noplot', action='store_true', help='do not plot output')
    args = parser.parse_args()

    if len(args.infile) < 4 or args.infile[-4:].lower() != '.mat' or not os.path.exists(args.infile):
        raise Exception('Invalid infile argument.')
    if args.resolution < 0.01:
        raise Exception('Invalid resolution argument.')
    

    scene_name = os.path.basename(args.infile)[:-4]
    # scene = NLOSScene(f'../DataCode_Phasor_Field_VWNLOS/Datasets/{scene_name}.mat')
    scene = NLOSScene(args.infile)

    gridsize = args.resolution
    lambda_times = 2 * gridsize / scene.sampling_grid_spacing
    wavelength = lambda_times * 2 * scene.sampling_grid_spacing

    print(f'Generating phasor field for reconstruction resolution {args.resolution}...')
    P = scene.phasor_pulse_convolution(wavelength, cycle_times=4, use_cache=True, show_plot=False)

    tm_start = time.perf_counter()
    R = NLOSReconstruction(scene, P, gridsize)

    tm_end = time.perf_counter()
    print(f'Reconstruction took {tm_end-tm_start:.6f} sec')

    if args.outfile is None:
        args.outfile = f'{scene_name}_{gridsize:.6f}_W.npy'
    np.save(args.outfile, R.voxels)
    print(f"'Saved reconstructed cube to '{args.outfile}'")

    if not args.noplot:
        img = np.max(R.voxels, axis=2)
        img = np.flip(np.flip(img, axis=0), axis=1).transpose()
        
        plt.imshow(img)
        plt.show()
    
if __name__ == '__main__':
    main()