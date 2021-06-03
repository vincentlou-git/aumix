# aumix
Audio Unmixing and Music Score Transcription. Developed as my undergraduate final year project.

# Installation & Run
1. Clone this git repository.
2. Optionally, create a new virtual environment. For example, using Anaconda, this can be done by `conda create -n aumix-test-venv python=3.6 anaconda`.
3. Activate a Python virtual environment (either an existing one, or the one created in Step 2).
4. Install `aumix` as a package by `pip install --index-url https://test.pypi.org/simple --no-deps aumix-DoraMemo`.

   Alternatively, append the `aumix` folder in the cloned repository to the `sys.path` variable:
   ```
   import sys
   sys.path.append("path_to_aumix")
   ```
5. Install all dependencies in the repository by `pip install -r requirements.txt`.
6. Change the directory to the script you want to run, then run it by `python3 <script_name>`. For example:

   ```
   repo$   cd examples
   repo$   python3 fourier_transform.py
   ```
   
   This is done to preserve the relative paths used in some scripts, such as `examples/eval/separation/urmp_gen.py`.
   
   
# Data
The URMP dataset has to be downloaded separately due to its size. It is available at http://www2.ece.rochester.edu/projects/air/projects/URMP.html. Although we will only use the audio files, the whole dataset (12.5GB) has to be downloaded by filling out the Google Form at https://goo.gl/forms/xSvMzlwl3IWijvcp2. The video files can be subsequently deleted.

For the evaluation scripts to work automatically, the URMP dataset has to be placed in the `data` folder with this structure:
```
repo/
└── data/
    └── URMP/
        └── Dataset
            └── 01_Jupiter_vn_vc
                └── AuMix_01_Jupiter_vn_vc.wav
                └── AuSep_1_vn_01_Jupiter.wav
                ...
            └── 02_Sonata_vn_vn
            ...
```
The sample piece, 34_Fugue_tpt_tpt_hn_tbn, has been included in the repository as a guide.
   
   
# Issues with Plots
The size of the plots in the visualiser might not be the intended size, since an IDE was used during the development and the plots show up fine in the plots pane. In any case, saving the plots should retain the correct size. A figure can be saved by passing the `savefig_path` parameter in `aumix.plot.plot.single_plot` and `aumix.plot.plot.single_subplots`. For example:

```
import aumix.plot.plot as aplot

# Create FigData

aplot.single_plot(fig_data=fig_data,
                  savefig_path="output.png"   # <<< this
                  )
```
