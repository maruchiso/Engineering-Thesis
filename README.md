# Engineering-Thesis
"Deep-learning-based 3D denoising in holographic tomography"
# Step 1
I used data from https://www.allencell.org/ for: 
  Tagged protein name: Desmoplakin    
  Primary structure labeled: Cell-cell junction
  Allen Institute Cell Line ID: AICS-17

This data comes from confocal microscopy, so I have to rearrange the data so that instead of an intensity distribution I get values close to the value of the refractive index distribution and prepare data to easy machine learning.

# Step 2
On the basis of the received data, prepare the de-noised data using BM3D algorithm and noised data

# Step 3
Writing, training and testing neural networks based on Autoencoder and U-net architectures.
