## PIRATES
Code for PIRATES Image Reconstruction Algorithm

There are two components to the code provided for the PIRATES Algorithm. 

* PART A) Code to build training data for PIRATES
  * make_MCFOST_model.py - code to generate MCFOST models to train PIRATES on.
  * custom_MCFOST_density.py - collection of adapted files from (insert Alison's Github)
  * Alison_functions.py - collection of adapted files from (insert Alison's Github)
  * requirements file is called : requirements_MCFOST.txt
 
To run code in Part A), you will need to install MCFOST (Christophe Pinte) version 4.03 - https://github.com/cpinte/mcfost.
    
* PART B) Code to build and train the PIRATES Algorithm
  * final_runexpt_fitnn_jatis.py - organisational file from which PIRATES is designed and constructed. Handles metadata
  * final_imrfitnn_jatis.py - contains all functions to build, train, iteratively fit and evaluate PIRATES performance
  * requirements file is called : requirements_PIRATES.txt
 
Requirements files are provided for Part A) and Part B), as you may wish to run each component of the package in a unique environment, on different computers. Part A - code to generate training data for PIRATES is best run on a HPC computer with many CPUs. Part B - code to build, train and fit PIRATES is best run on a GPU.
 
We provide the following meta data for the g18 NRM mask:
* interferometric (u,v) coordinates
* the index of the (u,v) coordinates used for closure phase quantities
 


The directory structure for this repository is as follows. To use this code out of the box, the user is required to add some folders where indicated by 'User created' or 'User generated' labels.

<pre>
├── PIRATES/ (clone of this repository)
│   ├── make_MCFOST_model.py               # Script to generate training distributions for PIRATES from radiative transfer code MCFOST
│   ├── custom_MCFOST_density.py           # Script to generate custom density files for input to MCFOST
│   ├── Alison_functions.py                # Miscellaneous function for MCFOST parameter file generation
│   ├── final_runexpt_fitnn_jatis.py       # Script to organise and design PIRATES
│   ├── final_imrfitnn_jatis.py            # Contains all functions to build, train, iteratively fit and evaluate PIRATES
│   ├── u_coords.npy                       # Example u coordinates (u,v) for g18 mask
│   ├── v_coords.npy                       # Example v coordinates (u,v) for g18 mask
│   └── indx_of_cp.npy                     # Example indicies of (u,v) for closure phase sampling
├── image_recon_data/                      # User created - main data folder  
│   ├── model_test/                        # User created - model specific data folder  
│   │   └── savefigs/                      # User created - data folder for saving figures
│   ├── pre-saved_y.npy                    # User generated - training data (y - images)
│   └── pre-saved_x.npy                    # User generated - training data (x - observables)
├── README.md
├── requirements_MCFOST.txt                # Requirements file for Part A) - MCFOST training data generation
└── requirements_PIRATES.txt               # Requirements file for Part B) - PIRATES running code
</pre>

 
