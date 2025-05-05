# De-MSI

De-MSI is a novel deep learning-based method specifically developed for denoising MSI data without ground truth. The core concept of De-MSI involves constructing the reliable training dataset by leveraging prior knowledge of mass spectrometry from the noisy MSI data, followed by training a deep neural network to improve the data quality by removing the noise from the original images. Developer is Lei Guo from Fuzhou University of China.


# Overview of De-MSI
<div align=center>
<img src="![1746435987903](https://github.com/user-attachments/assets/91ae0a5e-7ebd-4957-a076-179354eb56fa)
" width="800" height="480" /><br/>
</div>

__Overflow of the proposed De-MSI for MSI data denoising__. Initially, the pairs of isotopic ions and monoisotopic ions are identified using the presented DeepION or other established tools. Subsequently, isotopic ion is processed through the deep denoising network to produce the denoised images. The model optimization involves minimizing the reconstruction loss, calculated as the mean absolute error between the output of deep denoising network and the monoisotopic ions. After training, the original MSI data is input into the trained deep denoising network to interface the final denoised output .


# Requirement

    python == 3.5, 3.6 or 3.7

    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    
    conda install opencv=4.5.3 numpy=1.8.0 matplotlib=2.2.2 scipy=1.12.0 networkx=3.1

    conda install scikit-learn=1.4.1

    conda install umap-learn -c conda-forge
    
# Quickly start

## Input
The input consists of preprocessed MSI data with a two-dimensional shape of [XY, P], where X and Y represent the pixel numbers of the horizontal and vertical coordinates of the MSI data, respectively, and P represents the number of ions. A masking matrix [XY, 1], where the background is set to 0, and the tissue region is set to 1.

## Run De-MSI model

cd to the De-MSI fold, and run:

    python run_ref.py --input_PMatrix Cell_Test.csv --input_Pshape 26 26 --input_Matrix Cell_Train.csv --input_shape 25 26 --n_components 20 --input_PeakList Cell_peak.csv --output_file output
    
# Contact

Please contact me if you have any help: gl5121405@gmail.com
