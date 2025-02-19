# About the Example Notebooks

In this directory, you will find example use cases of TIAToolbox functionalities in the form of Jupyter Notebooks. These notebooks are designed to run on Google Colab (unless otherwise stated), but you can also run them on your local system. Each example notebook includes links to open the notebook in Google Colab and GitHub. Simply click on one of these links to open the notebook on the selected platform. If the left click does not work, you can right-click on "Open in Colab" and select "Open in new tab".

## Running Notebooks Locally

To run the notebooks on your local machine, set up your Python environment as explained in the [installation guide](../docs/installation.rst).

## Running Notebooks on [Google Colab](https://colab.research.google.com/)

Each notebook contains all the information needed to run the example remotely on Colab. All you need is a local computer with a standard browser connected to the Internet. The local computer does not need to have any programming languages installed—all necessary resources are provided at the remote site, free of charge, thanks to Google Colaboratory. Ensure that "colab" appears in the address bar. Familiarize yourself with the drop-down menus near the top of the window. You can edit the notebook during the session, for example, by substituting your own image files for those used in the demo. Experiment by changing the parameters of functions. You cannot permanently change this version of the notebook on GitHub or Colab, so you cannot inadvertently mess it up. Use the notebook's File Menu if you wish to save your own (changed) version.

### GPU or CPU Runtime

Processes in the notebooks can be accelerated by using a GPU. Whether you are running the notebook on your system or Colab, you need to check and specify if you are using GPU or CPU hardware acceleration. In Colab, ensure that the runtime type is set to GPU in the *"Runtime → Change runtime type → Hardware accelerator"*. If you are *not* using GPU, consider changing the `device` variable to `cpu`, otherwise, some errors may occur when running the following cells.

> **IMPORTANT**: If you are using Colab and install TIAToolbox, please note that you need to restart the runtime after installation before proceeding (menu: *"Runtime → Restart runtime"*). This is necessary to load the latest versions of prerequisite packages installed with TIAToolbox. After restarting, you should be able to run all the remaining cells together (*"Runtime → Run after"* from the next cell) or one by one.

## Structure of the Examples Directory

The examples directory includes general notebooks explaining different functionalities/modules incorporated in TIAToolbox. Most of these notebooks are written with less advanced users in mind—some familiarity with Python is assumed—but the capabilities they demonstrate are also useful to more advanced users.

The examples directory contains two subdirectories: `full-pipelines` and `inference-pipelines`, which include examples of using TIAToolbox for training neural networks or inference of WSIs for high-level computational pathology applications, such as patient survival prediction and MSI status prediction from H&E whole slide images.

## A) Examples of TIAToolbox Functionalities

Below is a list of our Jupyter notebooks, with brief descriptions of the TIAToolbox functionalities each notebook provides.

### 1. Reading Whole Slide Images ([01-wsi-reading](./01-wsi-reading.ipynb))

This notebook shows how to use TIAToolbox to read different kinds of WSIs. TIAToolbox provides a uniform interface to various WSI formats. Learn some well-known techniques for WSI masking and patch extraction.

[![image](../docs/images/wsi-reading.png)](./01-wsi-reading.ipynb)

### 2. Stain Normalization of Histology Images ([02-stain-normalization](./02-stain-normalization.ipynb))

Stain normalization is a common pre-processing step in computational pathology to reduce color variation that has no clinical significance. TIAToolbox offers several stain-normalization algorithms, including Reinhard, Ruifork, Macenko, and Vahadane.

[![image](../docs/images/stain-norm-example.png)](./02-stain-normalization.ipynb))

### 3. Extracting Tissue Mask from Whole Slide Images ([03-tissue-masking](./03-tissue-masking.ipynb))

This notebook shows how to extract tissue regions from a WSI using TIAToolbox with a single line of Python code.

[![image](../docs/images/tissue-mask.png)](./03-tissue-masking.ipynb)

### 4. Extracting Patches from Whole Slide Images ([04-patch-extraction](./04-patch-extraction.ipynb))

Learn how to use TIAToolbox to extract patches from large histology images based on point annotations or using a fixed-size sliding window.

[![image](../docs/images/patch-extraction.png)](./04-patch-extraction.ipynb)

### 5. Patch Prediction in Whole Slide Images ([05-patch-prediction](./05-patch-prediction.ipynb))

Use TIAToolbox for patch-level prediction with a range of deep learning models. Predict the type of patches in a WSI with just two lines of Python code.

[![image](../docs/images/patch-prediction.png)](./05-patch-prediction.ipynb)

### 6. Semantic Segmentation of Whole Slide Images ([06-semantic-segmentation](./06-semantic-segmentation.ipynb))

Use pretrained models to automatically segment different tissue region types in WSIs.

[![image](../docs/images/sematic-segment.png)](./06-semantic-segmentation.ipynb)

### 7. Advanced Model Techniques ([07-advanced-modeling](./07-advanced-modeling.ipynb))

This notebook is aimed at advanced users familiar with object-oriented programming concepts in Python and the TIAToolbox models framework.

[![image](../docs/images/advanced-techniques.png)](./07-advanced-modeling.ipynb)

### 8. Nucleus Instance Segmentation in Whole Slide Images Using the HoVer-Net Model ([08-nucleus-instance-segmentation](./08-nucleus-instance-segmentation.ipynb))

Demonstrate the use of the TIAToolbox implementation of the HoVer-Net model for nucleus instance segmentation and classification.

[![image](../docs/images/hovernet.png)](./08-nucleus-instance-segmentation.ipynb)

### 9. Multi-task Segmentation in Whole Slide Images Using the HoVer-Net+ Model ([09-multi-task-segmentation](./09-multi-task-segmentation.ipynb))

Demonstrate the use of the TIAToolbox implementation of the HoVer-Net+ model for nucleus instance segmentation/classification and semantic segmentation of intra-epithelial layers.

[![image](../docs/images/hovernetplus.png)](./09-multi-task-segmentation.ipynb)

### 10. Image Alignment ([10-wsi_registration](./10-wsi-registration.ipynb))

Show how to use TIAToolbox for registration of an image pair using Deep Feature Based Registration (DFBR) followed by non-rigid alignment using SimpleITK.

[![image](../docs/images/wsi-registration.png)](./10-wsi-registration.ipynb)

### 11. Feature Extraction Using Foundation Models ([11-import-foundation-models](./11-import-foundation-models.ipynb))

Explain how to extract features from WSIs using pre-trained models from the `timm` library.

[![image](../docs/images/feature_extraction.png)](./11-import-foundation-models.ipynb)

## B) Examples of High-Level Analysis (Pipelines) Using TIAToolbox

List of Jupyter notebooks demonstrating how to use TIAToolbox for high-level analysis in computational pathology.

### 1. Prediction of Molecular Pathways and Key Mutations in Colorectal Cancer from Whole Slide Images (idars)

Prediction of molecular pathways and key mutations directly from Haematoxylin and Eosin stained histology images can help bypass additional genetic (e.g., polymerase chain reaction or PCR) or immunohistochemistry (IHC) testing, with a view to saving both money and time. In this notebook, we use TIAToolbox's pretrained models to reproduce the inference results obtained by the IDaRS pipeline due to <a href="https://www.thelancet.com/journals/landig/article/PIIS2589-7500(2100180-1/fulltext">Bilal et al</a>. In TIAToolbox, we include models that are capable of predicting the following in whole slide images:

- Microsatellite instability (MSI)
- Hypermutation density
- Chromosomal instability
- CpG island methylator phenotype (CIMP)-high prediction
- BRAF mutation
- TP53 mutation

[![image](../docs/images/idars.png)](./inference-pipelines/idars.ipynb)

### 2. Prediction of HER2 Status in Breast Cancer from H&E Stained Whole Slide Images

Demonstrate how to use TIAToolbox to reproduce the SlideGraph+ method to predict HER2 status of breast cancer samples from H&E stained WSIs.

- Example notebook on training the SlideGraph model: slide-graph for training
- Example notebook on using SlideGraph model for WSI inference: slide-graph for inference

[![image](../docs/images/her2-prediction-example.png)](./inference-pipelines/slide-graph.ipynb)
