# aumix
Audio Unmixing and Music Score Transcription. Developed as my undergraduate final year project.

# Installation & Run
1. Clone this git repository.
2. Install `aumix` as a package by `pip install aumix`.

   Alternatively, append the `aumix` folder in the cloned repository to the `sys.path` variable:
   ```
   import sys
   sys.path.append("path_to_aumix")
   ```
3. Change the directory to the script you want to run, then run it by `python3 <script_name>`. For example:

   ```
   repo$   cd examples
   repo$   python3 fourier_transform.py
   ```
   
   This is done to preserve the relative paths used in some scripts, such as `examples/eval/separation/urmp_gen.py`.
