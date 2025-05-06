import numpy as np

from trainer import *
import pandas as pd
from scipy import sparse
import os
import argparse

parser = argparse.ArgumentParser(
    description='De-MSI for data denoising of mass spectrometry imaging')
parser.add_argument('--input_Matrix',required= True,help = 'path to inputting MSI data matrix')
parser.add_argument('--input_PeakList',required= True,help = 'path to inputting MSI peak list')
parser.add_argument('--input_Shape',required= True,type = int, nargs = '+', help='inputting MSI file shape')
parser.add_argument('--input_MASK', help = 'path to inputting mask matrix')
parser.add_argument('--input_Monoiso', help = 'path to inputting the pair of isotopic ion and monoisotopic ions')
parser.add_argument('--output_File', default='output/',help='output file name')

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    data = np.loadtxt(args.input_Matrix,delimiter=',')
    data = nor_std(data)

    input_shape = args.input_Shape

    peak = np.loadtxt(args.input_PeakList)
    peak = np.around(peak,4)

    mask = np.loadtxt(args.input_MASK)

    mono_iso = pd.read_csv(args.input_Monoiso, delimiter=',')
    mono_iso = mono_iso.values

    denoised = DeNoising(data,peak,mono_iso,input_shape,mask,patch_size = 64)

    np.savetxt(args.output_File,denoised,delimiter=',')


