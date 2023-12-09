# Engineering-Thesis
"Deep-learning-based 3D denoising in holographic tomography"
# Step 1
I used data from https://www.allencell.org/ for: 
  Tagged protein name: Desmoplakin    
  Primary structure labeled: Cell-cell junction
  Allen Institute Cell Line ID: AICS-17

This data comes from confocal microscopy, so I have to rearrange the data so that instead of an intensity distribution I get values close to the value of the refractive index distribution and prepare data to easy machine learning. That was made in file named "prepare_data.py"

# Step 2
On the basis of the received data, prepare the de-noised data using https://github.com/LesikDee/ED_Denoising/tree/master

# Step 3
Writing a network and training it using the data from the steps above
