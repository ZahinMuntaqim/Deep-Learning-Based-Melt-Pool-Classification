# Deep-Learning-Based-Melt-Pool-Classification

# Deep Learning-Based Melt-Pool Classification

A PyTorch-based deep learning framework for multiclass classification of melt-pool morphologies in metal additive manufacturing using process parameters and material properties.

## Project Overview

Melt-pool morphology plays an important role in determining the quality, stability and mechanical properties of additively manufactured components. Different melt-pool conditions, including Lack of Fusion (LOF), Balling, Desirable, Keyhole and Spatter Formation, are associated with different processing conditions.

This project develops a **Multilayer Perceptron (MLP) neural network** to classify melt-pool morphologies using process parameters and material characteristics.

The proposed framework uses data preprocessing, feature selection, class-weighted learning and multiclass classification to predict melt-pool conditions.

## Research Objectives

- Develop a deep learning model for melt-pool morphology classification.
- Investigate the relationship between processing parameters, material properties and melt-pool morphology.
- Classify five different melt-pool conditions.
- Evaluate model performance using multiple classification metrics.
- Develop a reusable prediction framework for new processing conditions.

## Dataset Description

The dataset contains **1,249 observations and 50 original columns**.

**Target variable:** `meltpool shape`

### Classification categories

| Class | Number of observations |
|---|---:|
| LOF (Lack of Fusion) | 305 |
| Balling | 133 |
| Desirable | 364 |
| Keyhole | 439 |
| Spatter Formation | 8 |
| **Total** | **1,249** |

### Input features

**Processing parameters**
- Beam power
- Scanning velocity
- Hatch spacing
- Layer thickness
- Beam diameter

**Material properties**
- Material density
- Specific heat capacity
- Thermal conductivity
- Melting temperature
- Absorption coefficient
- Alloy chemical composition

**Categorical features**
- Material
- Sub-process

The preprocessing pipeline selects available features, converts invalid numeric values into missing values and excludes metadata, post-process quality outcomes and potentially target-derived variables.

## Proposed Neural Network Architecture

The classification model is implemented using PyTorch and follows a fully connected feed-forward neural network architecture.

### Model structure

| Layer | Architecture | Activation | Dropout |
|---|---|---|---|
| Input | Preprocessed feature vector | — | — |
| Hidden Layer 1 | 128 neurons | ReLU | 0.25 |
| Hidden Layer 2 | 64 neurons | ReLU | 0.20 |
| Hidden Layer 3 | 32 neurons | ReLU | 0.10 |
| Output Layer | 5 neurons | Softmax for probabilities | — |

The input dimension depends on the number of features generated after preprocessing.

### Training configuration

| Parameter | Configuration |
|---|---|
| Deep learning framework | PyTorch |
| Model | Multilayer Perceptron (MLP) |
| Optimizer | AdamW |
| Loss function | Class-weighted Cross-Entropy Loss |
| Batch size | 32 |
| Maximum epochs | 250 |
| Early stopping patience | 30 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Random seed | 42 |
| Train/validation/test | Approximately 66% / 12% / 22% |

## Methodology

The complete modelling workflow consists of the following stages:

1. Dataset loading and initial inspection.
2. Missing-value and duplicate-value inspection.
3. Feature selection and data leakage control.
4. Numerical and categorical data preprocessing.
5. Stratified training, validation and testing.
6. Neural network architecture development.
7. Class imbalance handling using weighted loss.
8. Model training with AdamW optimisation.
9. Validation monitoring and early stopping.
10. Final testing and performance evaluation.
11. Confusion matrix and classification report generation.
12. Saving the trained model and preprocessing objects.

## Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

### Preliminary test results

Based on the current test-set confusion matrix:

| Evaluation metric | Result |
|---|---:|
| Accuracy | 80.73% |
| Balanced Accuracy | 83.46% |
| Macro F1-score | 84.39% |
| Weighted F1-score | 80.73% |

### Class-wise observations

- **LOF:** Good classification performance.
- **Balling:** Moderate recall, indicating some misclassification.
- **Desirable:** Some confusion with Keyhole morphology.
- **Keyhole:** Good classification performance.
- **Spatter Formation:** Two test observations were correctly classified; however, this is insufficient to establish reliable generalisation.

The confusion between Desirable and Keyhole is an important area for further investigation.

*Note: These are preliminary holdout results, not independent external validation results.*

## Repository Structure

```text
Melt-Pool-Classification/
│
├── README.md
│
├── Melt_Pool_Deep_Learning_Classification.ipynb
│
├── meltpoolnet_classification(1).csv
│
├── meltpool_mlp_model.pth
│
├── meltpool_preprocessor.joblib
│
└── meltpool_label_encoder.joblib
```

The model and preprocessing files are generated after successfully running the notebook.

## Installation and Requirements

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Melt-Pool-Classification.git
cd Melt-Pool-Classification
```

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn torch joblib jupyter
```

## Running the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Melt_Pool_Deep_Learning_Classification.ipynb
```

Place the dataset CSV file in the same directory as the notebook and execute the cells sequentially.

Alternatively, upload the notebook and CSV dataset to Google Colab and run the notebook there.

## Prediction

The trained model can predict melt-pool morphology for new processing conditions.

The prediction framework returns:

- Predicted melt-pool class.
- Prediction confidence.
- Probability of each of the five melt-pool classes.

The output can support preliminary process-condition assessment and melt-pool morphology analysis.

## Limitations and Future Work

Several opportunities exist to improve the current framework:

1. Collect additional observations for the Spatter Formation class.
2. Investigate the distinction between Desirable and Keyhole morphologies.
3. Perform hyperparameter optimisation.
4. Compare MLP performance with Random Forest, XGBoost and other machine learning algorithms.
5. Investigate alternative loss functions, including Focal Loss.
6. Perform publication-level group validation to assess generalisation to unseen experimental studies.
7. Conduct independent experimental validation.
8. Investigate explainable AI techniques to understand feature contributions.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Reproducibility

A fixed random seed (42) is used for reproducibility. The trained neural network, preprocessing pipeline and label encoder are saved separately for future inference.

Performance may vary with different dataset splits, hardware and software versions.

## License and Dataset Availability

The dataset and associated research materials should only be redistributed if their original source and licensing terms permit public sharing.

A suitable open-source licence can be added after confirming the rights to the code and dataset.

## Acknowledgement

This project explores the application of deep learning to melt-pool morphology classification in metal additive manufacturing.

**Research focus:** Deep Learning | Additive Manufacturing | Melt-Pool Morphology | Neural Networks | Process Classification
