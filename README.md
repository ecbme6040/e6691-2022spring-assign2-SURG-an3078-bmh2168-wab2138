# e6691-2022spring-assign2-surg-an3078-bmh2168-wab2138


## Assignment 2 - Surgical Phase Recognition
### E6691 Spring 2022

#### About
Assignment 2 for E6691 Spring 2022. In this assignment we attempt to detect phases of hernia surgeries through videos.

#### Project Structure
* Setup.ipynb
* preprocessing.ipynb
* EfficientNet.ipynb
* Resnet18_training_result.ipynb
* LSTM_sequence_cleaning.ipynb
* Resnet-LSTM.ipynb
* Predictions.ipynb
* video.phase.trainingData.clean.StudentVersion.csv
* all_labels_hernia.csv
* my_kaggle_preds.csv
* kaggle_template.csv
* README.md
* .ipynb_checkpoints/
* __pycache__/
* pickle/
* utils/
  * ImagesDataset.py
  * SequenceDataset.py
  * TestImagesDataset.py
  * VideosDataset.py
  * image_extraction.py
  * workers.py

#### Saved Model Weights
* https://drive.google.com/drive/folders/1kmv8pb2Zxfp-Fe-ZMMG4lW9ZxgYFHMqo?usp=sharing

**References**
* Rémi Cadène, Thomas Robert, Nicolas Thome, Matthieu Cord. “M2CAI Workflow Challenge: Convolutional Neural Networks with Time Smoothing and Hidden Markov Model for Video Frames Classification” Sorbonne Universites, UPMC Univ Paris 06, CNRS, LIP6 UMR 7606 (2016). 
  * https://arxiv.org/abs/1610.05541
* Aksamentov I, Twinanda AP, Mutter D, et al. “Deep Neural Networks Predict Remaining Surgery Duration from cholecystectomy videos.” 2017 International Conference on Medical Image Computing and Computer-Assisted Intervention: Springer; (2017): 586–593. 
  * https://link-springer-com.ezproxy.cul.columbia.edu/chapter/10.1007/978-3-319-66185-8_66 
