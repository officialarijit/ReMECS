# ReMECS: Real-Time Multimodal Emotion Classification System in E-Learning Context

- `ABSTRACT:` Emotions of learners are crucial and important in e-learning as they promote learning. To investigate the effects of emotions on improving and optimizing the outcomes of e-learning, machine learning models have been proposed in the literature. However, proposed models so far are suitable for offline mode, where data for emotion classification is stored and can be accessed boundlessly. In contrast, when data arrives in a stream, the model can see the data once and real-time response is required for real-time emotion classification. Additionally, researchers have identified that single data modality is incapable of capturing the complete insight of the learning experience and emotions. So, multi-modal data streams such as electroencephalogram (EEG), Respiratory Belt (RB), electrodermal activity data (EDA), etc., are utilized to improve the accuracy and provide deeper insights in learners’ emotion and learning experience. In this paper, we propose a Real-time Multimodal Emotion Classification System (ReMECS) based on Feed-Forward Neural Network, trained in an online fashion using the Incremental Stochastic Gradient Descent algorithm. To validate the performance of ReMECS, we have used the popular multimodal benchmark emotion classification dataset called DEAP. The results (accuracy and F1-score) show that the ReMECS can adequately classify emotions in real-time from the multimodal data stream in comparison to the state-of-the-art approaches.

The developed system is called **"Real-time Multi-modal Emotion Classification System (ReMECS)"**. The ReMECS is developed using 3-layer Feed Forward Neural Network optimized with Stochastic Gradient Descent (SGD) in online mode.

## DATASET
- `DEAP dataset` is required. 
- The experiment is conducted using the `EEG, EDA and RB measurements taken from DEAP dataset`. 
- To download `DEAP dataset` click on : https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html

## DATA Rearrangement required
```diff
- CAUTION

+ The DEAP data needs a simple rearrangement to work with the code. 

@@  Check the `data_rearrangements` folder for the  DEAP  data rearrangement from the .dat or .mat file from the DEAP dataset. @@
@@ Then follow the follwoing steps. @@

```


## Installation 
- Programming language
  - `Python 3.6 or above (tested in 3.10)`

- Operating system (tested)
  - `Ubuntu 18.04 (64 bit) - Intel CPU`
  - `MAC mini M1 - ARM based CPU`

- Required packages
  - `Scikit-Learn and River` &#8592; for model's performance matrics.
  - `Numpy` &#8592; for RECS's model development.
  - `River` &#8592; for streaming model development.
  - `Scikit-Learn` &#8592; for offline ML model development.
  
- Installation steps:
  - Step 1: Install `Anaconda`. 
  - Step 2: Create a `virtual environment` in Anaconnda 
  - Install the required packages using `pip` from the `requirements.txt` file.
  - Step 3: Open `terminal`, and `activate environment`.
  - Step 4: Run files :wink:.

## Publication
This work is published in **EANN 2021: Proceedings of the 22nd Engineering Applications of Neural Networks Conference**. The link to the paper "**Real-Time Multimodal Emotion Classification System in E-Learning Context**" is : https://doi.org//10.1007/978-3-030-80568-5_35.

  **Please cite the paper using the following bibtex:**


     @InProceedings{10.1007/978-3-030-80568-5_35,
        author="Nandi, Arijit
        and Xhafa, Fatos
        and Subirats, Laia
        and Fort, Santi",
        editor="Iliadis, Lazaros
        and Macintyre, John
        and Jayne, Chrisina
        and Pimenidis, Elias",
        title="Real-Time Multimodal Emotion Classification System in E-Learning Context",
        booktitle="Proceedings of the 22nd Engineering Applications of Neural Networks Conference",
        year="2021",
        publisher="Springer International Publishing",
        address="Cham",
        pages="423--435",
        isbn="978-3-030-80568-5"
      }

  **AMA style citation:**

     Nandi A., Xhafa F., Subirats L., Fort S. (2021) Real-Time Multimodal Emotion Classification System in E-Learning Context. In: Iliadis L., Macintyre J., Jayne C., Pimenidis E. (eds) Proceedings of the 22nd Engineering Applications of Neural Networks Conference. EANN 2021. Proceedings of the International Neural Networks Society, vol 3. Springer, Cham. https://doi.org//10.1007/978-3-030-80568-5_35



# NOTE*: Please feel free to use the code by giving proper citation and star to this repository.


## 📝 License

Copyright © [Arijit](https://github.com/officialarijit).
This project is MIT licensed.
