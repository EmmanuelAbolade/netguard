# 🛡️ NetGuard — Cybersecurity Machine Learning Portfolio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://netguard-ml.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)

> **Applying machine learning to real-world cybersecurity problems — spam detection, network intrusion detection, traffic clustering, and cyberattack classification.**

**Student:** Emmanuel Abolade | **Student Number:** C00288657  
**Institution:** South East Technological University, Carlow  
**Module:** Data Science & Machine Learning 2  
**Lecturer:** Ben OShaughnessy  
**Submission:** April 2026



##  Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Portfolio Projects](#-portfolio-projects)
- [Datasets](#-datasets)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Results Summary](#-results-summary)
- [Technologies Used](#-technologies-used)
- [References](#-references)
- [Glossary of Terms & Abbreviations](#-Glossary of Terms & Abbreviations)
- [Previous Portfolio](#-previous-portfolio)



##  Overview

**NetGuard** is a data science and machine learning portfolio built around a single real-world theme: **cybersecurity**. It applies four distinct machine learning algorithms — spanning supervised and unsupervised learning — to genuine cybersecurity problems using full-scale, publicly available datasets.

Each algorithm is implemented **twice**:
1. **From scratch using NumPy only** — demonstrating understanding of the underlying mathematics
2. **Professionally using scikit-learn** — with hyperparameter tuning, cross-validation, and industry-standard evaluation

The portfolio extends the [News→Signal](https://github.com/EmmanuelAbolade/news2signal-portfolio) financial sentiment portfolio from Semester 1, adding four new algorithms across a completely different domain.

### Why Cybersecurity?

- **Real-world relevance** — spam and network attacks cost organisations billions annually and remain the most common security threats
- **Natural fit for ML** — attack patterns are learnable; each algorithm maps naturally to a specific problem
- **Academic differentiation** — a distinct theme demonstrating independent thinking and breadth of application



## 🌐 Live Demo

**[https://netguard-ml.streamlit.app](https://netguard-ml.streamlit.app)**

The interactive dashboard includes:
- **Portfolio Overview** — all four projects with key statistics
- **Live spam classifier** — type any message, SVM classifies it instantly
- **Interactive traffic classifier** — adjust network parameters, KNN classifies the traffic type
- **K-Means visualiser** — watch centroids converge step by step
- **ANN architecture explorer** — adjust layers, see parameter counts in real time
- **Work log** — all documented decisions per algorithm
- **GitHub repository** — structure, references, and submission links



##  Projects & Methodology

### 01 — Email Spam Detection (SVM)

| Attribute | Detail |
|---|---|
| **Algorithm** | Support Vector Machine |
| **Learning type** | Supervised |
| **Dataset** | SMS Spam Collection — 5,572 messages |
| **Task** | Binary classification: spam or ham |

**Problem:** Can a machine learn to automatically distinguish spam from legitimate SMS messages?

**Algorithm Justification:** SVM was chosen for this problem because it is specifically well-suited to high-dimensional feature spaces — exactly what TF-IDF text vectorisation produces. As noted in the course notes: *"SVMs are currently among the best performers for a number of classification tasks ranging from text to genomic data."* The maximum margin decision boundary also provides strong generalisation, which is critical when spam patterns vary widely. Logistic Regression was considered but SVM's margin-based objective is more robust to the noisy, overlapping feature distributions in real SMS data.

**Approach:** TF-IDF vectorisation with bigrams (10,000 features), linear SVM kernel, GridSearchCV over C parameter, F1 score as primary metric to handle 87%/13% class imbalance.

**Key finding:** Linear kernel outperformed RBF — TF-IDF already creates a linearly separable high-dimensional space. Bigrams like "free prize" and "call now" were the strongest spam indicators. URL and phone number tokenisation preserved critical spam signals.

**From scratch:** Sub-gradient descent on soft-margin hinge loss: `min (1/2)||w||² + C·Σmax(0, 1 - yᵢ(wᵀxᵢ + b))`



### 02 — Network Intrusion Detection (KNN)

| Attribute | Detail |
|---|---|
| **Algorithm** | K-Nearest Neighbours |
| **Learning type** | Supervised |
| **Dataset** | NSL-KDD — 125,973 network connections |
| **Task** | 5-class classification: Normal, DoS, Probe, R2L, U2R |

**Problem:** Can KNN identify attack patterns by comparing new traffic connections to the most similar known examples?

**Algorithm Justification:** KNN was chosen because network attacks of the same type produce similar traffic signatures — they naturally cluster together in feature space. KNN exploits this directly by classifying new connections based on the most similar known examples. No assumptions are made about the data distribution (non-parametric), which is appropriate for network traffic that does not follow any standard statistical distribution. KNN also handles multi-class classification natively without modification, unlike SVM which requires one-vs-one or one-vs-rest strategies for more than two classes.

**Approach:** Min-Max normalisation (critical — src_bytes ranges 0–1M), GridSearchCV over k, metric, and weighting, stratified 5-fold cross-validation, distance-weighted voting.

**Key finding:** Normalisation was the single most impactful step — accuracy dropped dramatically without it. Distance-weighted voting consistently outperformed uniform. Optimal k balanced the bias-variance tradeoff visible in the k-value analysis plot.

**From scratch:** Euclidean distance computation to all training points, k-nearest selection, majority vote classification.



### 03 — Network Traffic Clustering (K-Means)

| Attribute | Detail |
|---|---|
| **Algorithm** | K-Means Clustering |
| **Learning type** | Unsupervised |
| **Dataset** | NSL-KDD — 125,973 network connections |
| **Task** | Discover natural attack groupings without labels |

**Problem:** Can an unsupervised algorithm discover attack patterns in network traffic without being told what classes exist?

**Algorithm Justification:** K-Means was chosen to model the real-world scenario where a security analyst encounters unknown traffic with no pre-existing labels. This is the only unsupervised algorithm in the portfolio — intentionally chosen to contrast with the three supervised algorithms and demonstrate understanding of the distinction between learning paradigms. K-Means was selected over hierarchical clustering because its O(nkt) complexity scales to the full 125,973-sample dataset, whereas hierarchical clustering's O(n²) memory requirement makes it impractical at this scale. Using the same NSL-KDD dataset as the KNN project enables a direct comparison between supervised and unsupervised performance.

**Approach:** StandardScaler (robust to outliers), K-Means++ initialisation with 20 restarts, Elbow method + Silhouette score for optimal k, Adjusted Rand Index to evaluate cluster quality against ground truth.

**Key finding:** DoS attacks cluster cleanly due to distinctive traffic patterns. R2L and U2R are harder — they resemble normal traffic in feature space. This directly demonstrates the challenge of unsupervised intrusion detection in practice.

**From scratch:** Full centroid initialisation, assignment, update loop with WCSS convergence tracking.



### 04 — Cyberattack Classification (ANN)

| Attribute | Detail |
|---|---|
| **Algorithm** | Feedforward Neural Network (MLP) |
| **Learning type** | Supervised |
| **Dataset** | NSL-KDD — 125,973 network connections |
| **Task** | 5-class cyberattack classification |

**Problem:** Network attacks have complex non-linear feature relationships. Can a neural network learn hierarchical representations and outperform simpler models?

**Algorithm Justification:** ANN was chosen as the final algorithm because it addresses the fundamental limitation of the previous three models — their inability to learn complex non-linear feature interactions automatically. SVM with a linear kernel assumes linear separability. KNN relies on raw distance in the original feature space. K-Means assumes spherical clusters. An ANN with hidden layers can learn hierarchical representations: lower layers detect simple patterns (individual feature combinations), higher layers detect complex attack signatures (combinations of those patterns). This is particularly important for minority classes like R2L and U2R where the distinguishing features are subtle combinations rather than individual signals. The same NSL-KDD dataset is used for the third consecutive project, enabling a definitive four-way algorithm comparison.

**Approach:** One-hot encoding for categorical features, architecture Input→128 (ReLU)→64 (ReLU)→5 (Softmax), Adam optimiser with early stopping, architecture comparison across depths and activation functions.

**Key finding:** ANN outperformed all other algorithms on complex minority classes (R2L, U2R). Two hidden layers confirmed optimal — directly aligned with course notes: *"Two hidden layers are sufficient to solve many problems."* ReLU outperformed Tanh for NSL-KDD's sparse zero-heavy features.

**From scratch:** Complete backpropagation — forward pass, cross-entropy loss, chain rule gradient computation, gradient descent weight update. Weights initialised between -0.5 and +0.5 as per course notes.



##  Datasets

| Dataset | Source | Samples | Features | Used in |
|---|---|---|---|---|
| [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) | UCI ML Repository | 5,572 | Text | SVM |
| [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | Canadian Institute for Cybersecurity | 125,973 | 41 | KNN, K-Means, ANN |

**Why NSL-KDD over KDD Cup 1999?** The original KDD99 dataset contains duplicate records that inflate accuracy artificially. NSL-KDD removes duplicates and rebalances classes — the standard academic benchmark for network intrusion detection research.

**Datasets are downloaded automatically** by the notebooks on first run — no manual download required.



##  Repository Structure

```
netguard/
│
├── 01_SVM_Spam_Detection.ipynb              # SVM — SMS Spam Collection
├── 02_KNN_Network_Intrusion.ipynb           # KNN — NSL-KDD
├── 03_KMeans_Network_Clustering.ipynb       # K-Means — NSL-KDD
├── 04_ANN_Cyberattack_Classification.ipynb  # ANN — NSL-KDD
│
├── app.py                                   # Streamlit portfolio dashboard
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
│
├── data/                                    # Datasets (auto-downloaded)
│   ├── SMSSpamCollection
│   └── KDDTrain+.txt
│
└── images/                                  # Saved visualisation outputs
```

### Notebook Structure (consistent across all four)

Each notebook follows the same structure:

1. **Theory** — Algorithm explained from course notes with formulas
2. **Dataset** — Loading, exploration, and visualisation
3. **From-Scratch Implementation** — NumPy only, no ML libraries
4. **Scikit-learn Implementation** — Professional pipeline with GridSearchCV
5. **Results & Evaluation** — Metrics, confusion matrix, visualisations
6. **Work Log** — Documented decisions, adjustments, and analysis
7. **References** — All sources correctly cited



##  Getting Started

### Prerequisites

- Python 3.11
- Anaconda (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/EmmanuelAbolade/netguard.git
cd netguard

# Create and activate environment
conda create -n ml_portfolio python=3.11 -y
conda activate ml_portfolio

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Open any notebook → **Kernel → Restart & Run All**. Datasets download automatically.

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`



##  Results Summary

| Algorithm | Dataset | Test Accuracy | Key Strength |
|---|---|---|---|
| SVM (Linear kernel) | SMS Spam | ~88% | High-dimensional text classification |
| KNN (k=5, distance-weighted) | NSL-KDD | ~91% | Instance-based pattern matching |
| K-Means (unsupervised) | NSL-KDD | ARI scored | Discovers unknown attack groups |
| ANN (MLP 128→64, ReLU) | NSL-KDD | ~95% | Complex non-linear attack signatures |



##  Technologies Used

| Category | Technologies |
|---|---|
| Language | Python 3.11 |
| ML Library | scikit-learn 1.7 |
| Numerical Computing | NumPy 2.4 |
| Data Manipulation | Pandas 2.3 |
| Visualisation | Matplotlib 3.10, Seaborn 0.13 |
| Dashboard | Streamlit 1.55 |
| Environment | Anaconda, Jupyter Notebook |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |



##  References

- Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering*. ACM DOCENG'11.
- Tavallaee, M. et al. (2009). *A Detailed Analysis of the KDD CUP 99 Data Set*. IEEE CISDA.
- Boser, B.E., Guyon, I., Vapnik, V.N. (1992). *A Training Algorithm for Optimal Margin Classifiers*. COLT '92.
- Cover, T., Hart, P. (1967). *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory, 13(1), 21–27.
- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations*. Berkeley Symposium.
- Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). *Learning Representations by Back-propagating Errors*. Nature, 323, 533–536.
- Arthur, D., Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*. SODA.
- Doyle, G. (2025). *Data Science & Machine Learning Lecture Notes (Chapters 2–7)*. South East Technological University.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
- Canadian Institute for Cybersecurity. *NSL-KDD Dataset*. https://www.unb.ca/cic/datasets/nsl.html






##  Glossary of Terms & Abbreviations

| Abbreviation | Full Name | Description |
|---|---|---|
| **SVM** | Support Vector Machine | Supervised learning algorithm that finds the optimal hyperplane separating two classes with the maximum possible margin. Only the support vectors — training examples closest to the boundary — define the decision boundary. |
| **KNN** | K-Nearest Neighbours | Supervised, non-parametric, lazy learning algorithm. Classifies a new point by majority vote among its k closest training examples, measured by Euclidean distance. No model is built at training time. |
| **K-Means** | K-Means Clustering | Unsupervised algorithm that partitions data into k clusters by iteratively assigning points to their nearest centroid and recomputing centroids as the cluster mean. Minimises WCSS. |
| **ANN** | Artificial Neural Network | Computational model inspired by the brain, consisting of interconnected layers of neurons. Learns by adjusting connection weights through backpropagation to minimise a loss function. |
| **MLP** | Multi-Layer Perceptron | A specific type of ANN with one or more hidden layers between input and output. The architecture used in scikit-learn's MLPClassifier and in this portfolio's ANN notebook. |
| **TF-IDF** | Term Frequency-Inverse Document Frequency | Text vectorisation method that weights words by how often they appear in a document (TF) penalised by how commonly they appear across all documents (IDF). Distinctive words score highest. |
| **ReLU** | Rectified Linear Unit | Activation function defined as max(0, z). Returns positive values unchanged and sets negative values to zero. Default choice for hidden layers in modern ANNs — avoids vanishing gradients and produces sparse activations. |
| **NSL-KDD** | Network Security Lab — Knowledge Discovery in Databases | Network intrusion detection benchmark dataset from the University of New Brunswick. An improved version of KDD Cup 1999, with duplicate records removed to prevent artificially inflated accuracy. |
| **UCI** | University of California Irvine | UCI Machine Learning Repository — one of the oldest and most widely used public collections of datasets for machine learning research. Source of the SMS Spam Collection used in this portfolio. |
| **DoS** | Denial of Service | Cyberattack that floods a network or server with traffic, preventing legitimate users from accessing it. The largest attack class in NSL-KDD (45,927 samples). Examples: neptune, smurf, teardrop. |
| **R2L** | Remote to Local | Attack where an unauthorised remote user gains local access by exploiting network vulnerabilities. Rare in NSL-KDD (995 samples). Examples: guess_passwd, ftp_write, imap. |
| **U2R** | User to Root | Attack where a user with normal access escalates privileges to administrator (root) level. Rarest class in NSL-KDD (52 samples). Examples: buffer_overflow, rootkit, loadmodule. |
| **WCSS** | Within-Cluster Sum of Squares | The objective function K-Means minimises. Measures total squared distance from each point to its assigned centroid. Also called inertia. Used in the Elbow Method to select optimal k. |
| **ARI** | Adjusted Rand Index | Measures similarity between discovered clusters and true labels, adjusted for chance. Ranges from -1 to +1. Used to evaluate how well K-Means recovered ground truth structure without using labels during training. |
| **F1 Score** | F1 Score | Harmonic mean of precision and recall. Used as the primary metric for the SVM spam classifier to handle the 87%/13% class imbalance — accuracy alone would be misleading. |
| **ROC-AUC** | Receiver Operating Characteristic — Area Under the Curve | ROC plots True Positive Rate vs False Positive Rate across all thresholds. AUC summarises the curve as a single number (0–1). AUC = 1.0 is perfect; AUC = 0.5 is random. |
| **PCA** | Principal Component Analysis | Dimensionality reduction technique projecting high-dimensional data onto a lower-dimensional space preserving maximum variance. Used to visualise 41-dimensional NSL-KDD data in 2D for decision boundary plots. |
| **CV** | Cross-Validation | Model evaluation technique that splits data into k folds, training on k-1 and testing on 1, repeated k times. Stratified CV preserves class proportions in each fold — essential for imbalanced datasets like NSL-KDD. |
| **IBL** | Instance-Based Learning | Family of algorithms that store training instances directly and use them at prediction time. KNN is the most well-known example. Also called lazy learning, memory-based reasoning, or case-based reasoning. |
| **CA** | Continuous Assessment | The assessment format for this module — worth 60% of the final grade, submitted as a portfolio rather than a single exam. |

##  Previous Portfolio

**[News→Signal](https://github.com/EmmanuelAbolade/news2signal-portfolio)** — Semester 1

Financial news sentiment analysis using NLP and Logistic Regression to predict stock market direction from daily business headlines.

[![News2Signal](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://news2signal-portfolio-dl3byf65m8qhncpqcfpdny.streamlit.app/)

---

*South East Technological University, Carlow — Data Science & Machine Learning 2 — 2026*
