"""
anthem_transcribe.py

Transcribe using AnthemScore's command line interface.

@author: Chan Wai Lou
"""


import os

#
# Paramters
#

# AnthemScore's installation path
anthem_path = "D:\\Softwares\\AnthemScore4\\AnthemScore.exe"

# number of threads in AnthemScore
n_threads = 8

# Path to folder containing only files to be transcribed
input_path = "D:\\Year 3\\COMP3931 Individual Project\\repo\\examples\\eval\\separation\\adress_audio"

# Path to musicXML output
output_xml_path = "D:\\Year 3\\COMP3931 Individual Project\\repo\\data\\URMP\\AnthemScore_xml"

# Path to pdf output
output_pdf_path = "D:\\Year 3\\COMP3931 Individual Project\\repo\\data\\URMP\\AnthemScore_pdf"

# Path to asdt (AnthemScore project file format) output
output_asdt_path = "D:\\Year 3\\COMP3931 Individual Project\\repo\\data\\URMP\\AnthemScore_asdt"


#
# Computation
#

# Get all absolute filenames from the input folder
os.chdir(input_path)

filenames = list()
for dir_name, subdir_list, files in os.walk(input_path):
    filenames = [ (os.path.abspath(f), f.split(".")[0]) for f in files ]

# Go to AnthemScore's folder and start transcribing
for abs_input, name in filenames:
    cmd = f"{anthem_path} \"{abs_input}\" -a --threads {n_threads} " \
          f"-x \"{output_xml_path}\\{name}.xml\" " \
          f"-p \"{output_pdf_path}\\{name}.pdf\""
          # f"-d \"{output_asdt_path}\\{name}.asdt\" "
    os.system(cmd)
