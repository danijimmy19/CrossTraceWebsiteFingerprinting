# HiddenText

⚠️ This code and data is for research purposes only.⚠️

This repository contains source code and data for reproducing our results from HiddenText: Cross-Trace Website Fingerprinting Over Encrypted Traffic. 

#### Code
The <code>src</code> directory comprises sub-directories, each of which is labeled as a performance evaluation part of the paper depending on experiment name. At the beginning of each script the instructions to execute the script are included.

- For experiments A1 and A2, execute following command ```python <name-of-script.py> /path/to/save/model/model-name.h5 /path/to/dataset```
- For experiments A3, A4, and A5 execute the following command ```python <name-of-script.py> /path/to/wt-def-model.h5 /path/to/paired/dataset```

##### Python Libraries Required

#### Dataset
The used data set is given as CSV files for this research. For each experiment six CSV files are required, each of which is described in the following manner:
- x_train: This file contains the traffic traces which are used as an input for the CNN for training
- y_train: This file contains labels corresponding to the traffic traces available in x_train
- x_valid: This file contains the traffic traces used for validation
- y_valid: This file contains labels corresponding to the traffic traces available in x_valid
- x_test: This file constains the traffic traces for testing/evaluating the trained CNN model
- y_test: This file contains labels corresponding the traffic traces in x_test

**Note: -** The data needed for various experiments are organized in subdirectories similar to the code.

#### Contact
- Jimmy Dani (<danijy@mail.uc.edu>), University of Cincinnati
- Boyang Wang (<boyang.wang@uc.edu>), University of Cincinnati
