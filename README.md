## PIRATES
Code for PIRATES Image Reconstruction Algorithm

There are two components to the code provided for the PIRATES Algorithm. 
* Code to build training data for PIRATES
  * file file
* Code to build and train the PIRATES Algorithm
  * final_runexpt_fitnn_jatis.py - organisational file from which PIRATES is designed and constructed. Handles metadata
  * final_imrfitnn_jatis.py - contains all functions to build, train, iteratively fit and evaluate PIRATES performance
 
Prior to using this package you will need to have the following meta data:
* interferometric (u,v) coordinates
* the index of the (u,v) coordinates used for closure phase quantities
 

The directory structure for this repository is as follows. To use this code out of the box, the user is required to add some folders where indicated.

<pre>
├── PIRATES/ (clone of this repository)
│   ├── final_runexpt_fitnn_jatis.py       # Script to organise and design PIRATES
│   ├── final_imrfitnn_jatis.py            # Contains all functions to build, train, iteratively fit and evaluate PIRATES
│   ├── u_coords.npy                       # Example u coordinates (u,v) for g18 mask
│   ├── v_coords.npy                       # Example v coordinates (u,v) for g18 mask
│   └── indx_of_cp.npy                     # Exmaple indicies of (u,v) for closure phase sampling
├── image_recon_data/                      # User created - main data folder  
│   ├── model_test/                        # User created - model specific data folder  
│   │   └── savefigs/                      # User created - data folder for saving figures
│   ├── pre-saved_y.npy                    # User generated - training data (y - images)
│   └── pre-saved_x.npy                    # User generated - training data (x - observables)
├── README.md
└── requirements.txt                        # Dependencies
</pre>

 
