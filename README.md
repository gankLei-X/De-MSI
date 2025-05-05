# De-MSI

De-MSI is a novel deep learning-based method specifically developed for denoising MSI data without ground truth. The core concept of De-MSI involves constructing the reliable training dataset by leveraging prior knowledge of mass spectrometry from the noisy MSI data, followed by training a deep neural network to improve the data quality by removing the noise from the original images. Developer is Lei Guo from Fuzhou University of China.


# Overview of De-MSI
<div align=center>

<img src="https://github.com/user-attachments/assets/3ca0cad9-7ebc-4251-a0f0-f95b59d9ba7f" width="600" height="360" /><br/>
</div>

__Overflow of the proposed De-MSI for MSI data denoising__. Initially, the pairs of isotopic ions and monoisotopic ions are identified using the presented DeepION or other established tools. Subsequently, isotopic ion is processed through the deep denoising network to produce the denoised images. The model optimization involves minimizing the reconstruction loss, calculated as the mean absolute error between the output of deep denoising network and the monoisotopic ions. After training, the original MSI data is input into the trained deep denoising network to interface the final denoised output .


# Requirement

    python == 3.5, 3.6 or 3.7

    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    
    conda install opencv=4.5.3 numpy=1.8.0 matplotlib=2.2.2 scipy=1.12.0 networkx=3.1

    conda install scikit-learn=1.4.1

    
# Quickly start

## Input
The input consists of preprocessed MSI data with a two-dimensional shape of [XY, P], where X and Y represent the pixel numbers of the horizontal and vertical coordinates of the MSI data, respectively, and P represents the number of ions. A masking matrix [XY, 1], where the background is set to 0, and the tissue region is set to 1.

## Run De-MSI model

cd to the De-MSI fold,

if you want to denoise mouse fetus data acquired from MALDI, you can run:

    python RUN.py --input_Matrix .../data/Fetus/BabyMALDI_82_127.csv --input_PeakList .../data/Fetus/BabyMALDI_peaklist.csv --input_Shape 82 127  --input_MASK .../data/Fetus/BabyMALDI_mask.csv --input_Monoisotope .../data/Fetus/BabyMALDI_monoiso.csv --output_File output/

if you want to denoise rat brain data acquired from DESI, you can run:

    python RUN.py --input_Matrix .../data/Fetus/BabyMALDI_82_127.csv --input_PeakList .../data/Fetus/BabyMALDI_peaklist.csv --input_Shape 82 127  --input_MASK .../data/Fetus/BabyMALDI_mask.csv --input_Monoisotope .../data/Fetus/BabyMALDI_monoiso.csv --output_File output/
    
# Contact

Please contact me if you have any help: gl5121405@gmail.com
