# -*- coding: utf-8 -*-
"""
module_path.py

Adds relative paths for module searching for aumix.

@author: DoraMemo
"""

import sys
if "../" not in sys.path:
    sys.path.append("../")
if "../aumix" not in sys.path:
    sys.path.append("../aumix")