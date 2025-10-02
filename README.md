# Brain Data Classification (EEG Neural State Prediction)

This project was originally prepared as part of a GSoC 2025 application test for the ML4-Sci org. It focuses on classifying participant EEG data into two neural/metabolic states.

## Test Prompt

### Brain data classification

**Task**: Build a model for classifying the participant data into neural states using PyTorch or Keras.

Dataset: [https://docs.google.com/spreadsheets/d/e/2PACX-1vSC87hYugbdg0_MbAhqVHaGSTk_-tEb_X_1YeXo6qzuz-bKm3Vo3gQd6m4IlZ5CAQMUUxfZrtCgbWYv/pub?output=csv](https://www.google.com/url?q=https://docs.google.com/spreadsheets/d/e/2PACX-1vSC87hYugbdg0_MbAhqVHaGSTk_-tEb_X_1YeXo6qzuz-bKm3Vo3gQd6m4IlZ5CAQMUUxfZrtCgbWYv/pub?output%3Dcsv&sa=D&source=editors&ust=1759410933776725&usg=AOvVaw0ZzKF1E9OapYgH_SPdUAyp)

**Dataset Description:** The Dataset consists of participant data (rows) of two classes (marked in last column), reflecting different metabolic states. Features (columns) of band power (in [theta](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Theta_wave&sa=D&source=editors&ust=1759410933777068&usg=AOvVaw1PEtn_vt5ULdHomGDPMqto), [delta](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Delta_wave&sa=D&source=editors&ust=1759410933777159&usg=AOvVaw0w9VFc2XF7PpHpEALdZ0-M), [alpha](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Alpha_wave&sa=D&source=editors&ust=1759410933777247&usg=AOvVaw2J9VLtj_h2HKk4VIoMbwyy), [beta](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Beta_wave&sa=D&source=editors&ust=1759410933777320&usg=AOvVaw1pWQZb20ONmNvsVbQD3qjo), and [gamma](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Gamma_wave&sa=D&source=editors&ust=1759410933777405&usg=AOvVaw2NrJaXHWZPmdMHrM0FQxt3) ranges) have been extracted from a 64-channel EEG recording (you are welcome to use topological information for classification, taking into account that channels are not independent, to improve your results  – channel distribution figure attached).  

**Task 1:**  classify the data using linear regression, [SVM](https://www.google.com/url?q=https://scikit-learn.org/stable/modules/svm.html&sa=D&source=editors&ust=1759410933777802&usg=AOvVaw00OGTV7AXyWvs781IW_PP7) (feel free to play with kernel options), and [kNN](https://www.google.com/url?q=https://scikit-learn.org/stable/modules/neighbors.html&sa=D&source=editors&ust=1759410933777898&usg=AOvVaw3rS3iz_uAplKpXlFupJtDi) algorithms; provide accuracy and precision data for each. Suggest reasoning as to why the ones performing better might do so (you do not have to be correct – for this project, this is just something you should enjoy considering).

**Task 2:** identify top 5 features for classification using univariate feature selection ([UFS](https://www.google.com/url?q=https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html&sa=D&source=editors&ust=1759410933778490&usg=AOvVaw1zxRAGGa-XgWEcqDdZdchA)), [RFE](https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html&sa=D&source=editors&ust=1759410933778604&usg=AOvVaw27NIDuk8wouwSbv5g0eXU_), and [PCA](https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html&sa=D&source=editors&ust=1759410933778682&usg=AOvVaw0yz45sncj2G_2ZzKUkp5aQ). Give a quick explanation for why (or why not) they are they same across feature selection algorithms.

## Project Overview

* **Dataset**: 40 participants × 320 EEG features (band powers in theta, delta, alpha, beta, gamma ranges across 64 channels).
* **Tasks**:

  1. Implement classic ML classifiers (**Logistic Regression**, **SVM**, **kNN**) and compare accuracy/precision.
  2. Identify top 5 features using **Univariate Feature Selection (UFS)**, **Recursive Feature Elimination (RFE)**, and **PCA**, and explain differences.
  3. Extend analysis with a **Graph Neural Network (PyTorch Geometric)** leveraging electrode topology.

## Results (Cross-Validation)

* **Baseline (dummy classifier):** 51% acc, ~0.00 precision.
* **Best classic model (SVM poly kernel):** ~61% accuracy · precision ~0.66.
* **GNN ensemble:** 77.5% accuracy · F1-macro 0.77.

## Feature Selection Insights

* **Different methods surfaced different features** due to different criteria:

  * **UFS:** strong univariate correlations.
  * **RFE:** model-dependent importance.
  * **PCA:** variance explanation, not necessarily linked to class separation.

## Repository Contents

* `notebook_1.ipynb` → Classic ML models + feature selection.
* `notebook_2.ipynb` → Graph Neural Network (GCN, ChebConv, GAT) with topology-aware EEG graphs.
* `Clarification_of_Approach.pdf` → How ambiguities in the prompt were handled.
* `Results_and_Explanations.pdf` → Summarized performance and reasoning.
