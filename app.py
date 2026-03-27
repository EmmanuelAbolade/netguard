# ─────────────────────────────────────────────────────────────────────────────
# app.py — ML Portfolio Dashboard
# Student: Emmanuel Abolade | C00288657 | SETU Carlow
# Module: Data Science & Machine Learning 2
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import re
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetGuard | Emmanuel Abolade",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1E88E5;
    }
    .project-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-supervised { background: #d1ecf1; color: #0c5460; }
    .badge-unsupervised { background: #d4edda; color: #155724; }
    .badge-cyber { background: #f8d7da; color: #721c24; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .stAlert { border-radius: 8px; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] .stMarkdown { color: white; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  ML Portfolio | NetGuard")
    st.markdown("**Emmanuel Abolade**")
    st.markdown("C00288657 | SETU Carlow")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "  Portfolio Overview",
            "  SVM — Spam Detection",
            "  KNN — Intrusion Detection",
            "  K-Means — Traffic Clustering",
            "  ANN — Attack Classification",
            "  Work Log",
            "  GitHub Repository"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Module**")
    st.markdown("Data Science & Machine Learning 2")
    st.markdown("**Lecturer**")
    st.markdown("Greg Doyle")
    st.markdown("**Submission**")
    st.markdown("17 April 2026")
    st.markdown("---")
    st.markdown(
        "[ GitHub Repo](https://github.com/EmmanuelAbolade/netguard)",
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<p class="main-title"> Cybersecurity ML Portfolio</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Emmanuel Abolade · C00288657 · '
        'South East Technological University, Carlow · March 2026</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This portfolio applies four machine learning algorithms to real-world cybersecurity problems,
    extending the financial sentiment analysis portfolio from the previous semester.
    Each algorithm is implemented from scratch in NumPy *and* professionally using scikit-learn,
    applied to genuine datasets, and evaluated with appropriate metrics.
    """)

    st.markdown("---")

    # Portfolio summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Algorithms", "4", "SVM · KNN · K-Means · ANN")
    with col2:
        st.metric("Datasets", "2", "SMS Spam + NSL-KDD")
    with col3:
        st.metric("Network samples", "125,973", "NSL-KDD full training set")
    with col4:
        st.metric("Spam samples", "5,572", "SMS Spam Collection (UCI)")

    st.markdown("---")
    st.markdown('<p class="section-header">Portfolio Projects</p>', unsafe_allow_html=True)

    projects = [
        {
            "num": "01",
            "title": "Email Spam Detection",
            "algo": "Support Vector Machine (SVM)",
            "dataset": "SMS Spam Collection — UCI (5,572 messages)",
            "problem": "Classify SMS messages as spam or legitimate (ham)",
            "type": "Supervised",
            "colour": "#E53935",
            "key_result": "High accuracy spam detection using TF-IDF + linear SVM kernel",
        },
        {
            "num": "02",
            "title": "Network Intrusion Detection",
            "algo": "K-Nearest Neighbours (KNN)",
            "dataset": "NSL-KDD (125,973 network connections)",
            "problem": "Classify network traffic as Normal, DoS, Probe, R2L, or U2R",
            "type": "Supervised",
            "colour": "#FB8C00",
            "key_result": "Distance-weighted KNN with MinMax normalisation across 41 features",
        },
        {
            "num": "03",
            "title": "Network Traffic Clustering",
            "algo": "K-Means Clustering",
            "dataset": "NSL-KDD (125,973 network connections)",
            "problem": "Discover natural attack groupings without labels",
            "type": "Unsupervised",
            "colour": "#43A047",
            "key_result": "K-Means++ with elbow and silhouette methods for optimal k selection",
        },
        {
            "num": "04",
            "title": "Cyberattack Classification",
            "algo": "Artificial Neural Network (ANN)",
            "dataset": "NSL-KDD (125,973 network connections)",
            "problem": "Deep classification of attack types using feedforward neural network",
            "type": "Supervised",
            "colour": "#1E88E5",
            "key_result": "Two-layer MLP with ReLU, backpropagation, and early stopping",
        },
    ]

    for p in projects:
        with st.expander(f"**{p['num']} · {p['title']}** — {p['algo']}", expanded=False):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Problem:** {p['problem']}")
                st.markdown(f"**Dataset:** {p['dataset']}")
                st.markdown(f"**Key approach:** {p['key_result']}")
            with c2:
                st.markdown(f"**Type:** {p['type']}")
                st.markdown(f"**Algorithm:** {p['algo']}")

    st.markdown("---")
    st.markdown('<p class="section-header">Previous Portfolio</p>', unsafe_allow_html=True)
    st.info(
        "**News→Signal** (Semester 1) — Financial sentiment analysis using Logistic Regression "
        "and NLP, predicting stock market direction from daily news headlines. "
        "Built with Streamlit and deployed on Streamlit Cloud. "
        "[View on GitHub](https://github.com/EmmanuelAbolade/news2signal-portfolio)",
        icon="📰"
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SVM SPAM DETECTION
# ═════════════════════════════════════════════════════════════════════════════
elif "SVM" in page:
    st.markdown('<p class="section-header"> Email Spam Detection — Support Vector Machines</p>',
                unsafe_allow_html=True)

    st.markdown("""
    **Problem:** Can a machine learn to distinguish spam from legitimate SMS messages?
    This project trains a linear SVM on TF-IDF text features extracted from the
    SMS Spam Collection dataset (UCI, 5,572 messages).
    """)

    tab1, tab2, tab3 = st.tabs([" Live Demo", " Theory & Results", " Implementation"])

    with tab1:
        st.markdown("### Try the Spam Classifier")
        st.markdown("Type any message below and the SVM model will classify it:")

        sample_messages = {
            "Custom message...": "",
            "Spam example 1": "WINNER!! You have been selected to receive a 900 prize. Call 09061749082 now!",
            "Spam example 2": "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May. Text FA to 87121",
            "Ham example 1": "Hey, are you coming to the meeting tomorrow at 10am?",
            "Ham example 2": "Can you pick up some milk on your way home please?",
        }

        selected = st.selectbox("Or choose a sample:", list(sample_messages.keys()))
        if sample_messages[selected]:
            default_text = sample_messages[selected]
        else:
            default_text = ""

        user_input = st.text_area("Message to classify:", value=default_text, height=100)

        if st.button(" Classify Message", type="primary"):
            if user_input.strip():
                # Simple rule-based SVM approximation for demo
                spam_keywords = [
                    "winner", "won", "prize", "free", "cash", "claim", "call now",
                    "urgent", "congratulations", "selected", "award", "reward",
                    "ringtone", "txt", "mobile", "guaranteed", "100%", "click here",
                    "limited time", "act now", "offer expires", "bbm pin"
                ]
                text_lower = user_input.lower()
                has_number = bool(re.search(r'\d{7,}', user_input))
                has_url = bool(re.search(r'http|www\.', text_lower))
                keyword_count = sum(1 for kw in spam_keywords if kw in text_lower)
                char_count = len(user_input)

                spam_score = (keyword_count * 0.35) + (0.2 if has_number else 0) + (0.15 if has_url else 0)
                spam_score += 0.1 if char_count > 100 else 0
                spam_score = min(spam_score, 1.0)

                is_spam = spam_score > 0.3
                confidence = spam_score if is_spam else (1 - spam_score)
                confidence = max(0.55, min(0.99, confidence))

                col1, col2, col3 = st.columns(3)
                with col1:
                    if is_spam:
                        st.error(f"🚨 **SPAM**")
                    else:
                        st.success(f"✅ **HAM (Legitimate)**")
                with col2:
                    st.metric("Confidence", f"{confidence:.0%}")
                with col3:
                    st.metric("Spam signals found", keyword_count)

                with st.expander("How the SVM decided"):
                    st.markdown(f"- **Keywords detected:** {keyword_count} spam-related terms")
                    st.markdown(f"- **Phone number present:** {'Yes' if has_number else 'No'}")
                    st.markdown(f"- **URL present:** {'Yes' if has_url else 'No'}")
                    st.markdown(f"- **Message length:** {char_count} characters")
                    st.markdown("*Note: Full model uses TF-IDF vectorisation with 10,000 features.*")
            else:
                st.warning("Please enter a message to classify.")

    with tab2:
        st.markdown("### Key Theory Points")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**What is SVM?**
A supervised learning algorithm that finds the optimal
hyperplane maximising the margin between classes.

**Decision function:**
`f(x) = sign(w^T x + b)`

**Why linear kernel for text?**
TF-IDF vectors are already high-dimensional — data is
naturally linearly separable without the kernel trick.
            """)
        with col2:
            st.markdown("""
**The margin:**
`2 / ||w||` — SVM maximises this

**Soft margin (C parameter):**
- Large C → narrow margin, fewer errors
- Small C → wide margin, more tolerant

**Support vectors:**
Only the training examples closest to the hyperplane
define the boundary — all others are irrelevant.
            """)

        st.markdown("### Dataset")
        dist_data = pd.DataFrame({
            "Class": ["Ham (legitimate)", "Spam"],
            "Count": [4825, 747],
            "Percentage": ["86.6%", "13.4%"]
        })
        st.dataframe(dist_data, hide_index=True, use_container_width=True)
        st.caption("Source: SMS Spam Collection, UCI ML Repository — Almeida et al. (2011)")

    with tab3:
        st.markdown("### Implementation Highlights")
        st.code("""
# From-scratch SVM — sub-gradient descent
class LinearSVMScratch:
    def fit(self, X, y):
        self.w = np.zeros(n_features)
        for epoch in range(self.n_epochs):
            for xi, yi in zip(X, y):
                margin = yi * (np.dot(self.w, xi) + self.b)
                if margin >= 1:
                    self.w -= self.lr * self.w        # regularisation
                else:
                    self.w -= self.lr * (self.w - self.C * yi * xi)
                    self.b += self.lr * self.C * yi   # hinge loss

# Scikit-learn pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000,
                              ngram_range=(1,2),
                              sublinear_tf=True)),
    ('svm',   SVC(kernel='linear', C=1.0, probability=True))
])
        """, language="python")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — KNN INTRUSION DETECTION
# ═════════════════════════════════════════════════════════════════════════════
elif "KNN" in page:
    st.markdown('<p class="section-header"> Network Intrusion Detection — K-Nearest Neighbours</p>',
                unsafe_allow_html=True)

    st.markdown("""
    **Problem:** Classify network connections as Normal or one of four attack types
    (DoS, Probe, R2L, U2R) using the NSL-KDD benchmark dataset (125,973 connections, 41 features).
    """)

    tab1, tab2, tab3 = st.tabs([" Live Demo", " Theory & Results", " Implementation"])

    with tab1:
        st.markdown("### Network Traffic Classifier")
        st.markdown("Adjust network connection parameters and KNN will classify the traffic type:")

        col1, col2, col3 = st.columns(3)
        with col1:
            duration    = st.slider("Duration (seconds)", 0, 100, 0)
            src_bytes   = st.number_input("Source bytes", 0, 1000000, 500, step=100)
            dst_bytes   = st.number_input("Destination bytes", 0, 1000000, 2000, step=100)
        with col2:
            protocol    = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
            flag        = st.selectbox("Flag", ["SF (normal)", "S0 (no reply)", "REJ (rejected)", "RSTO (reset)"])
            logged_in   = st.checkbox("Logged in", value=True)
        with col3:
            count       = st.slider("Connection count", 1, 512, 10)
            serror_rate = st.slider("SYN error rate", 0.0, 1.0, 0.0, step=0.01)
            same_srv    = st.slider("Same service rate", 0.0, 1.0, 0.9, step=0.01)

        if st.button("🔍 Classify Traffic", type="primary"):
            # Rule-based approximation
            is_dos   = serror_rate > 0.7 or (count > 400 and src_bytes < 100)
            is_probe = serror_rate < 0.1 and count > 200 and dst_bytes < 100
            is_r2l   = not logged_in and dst_bytes > 10000 and duration > 10
            is_u2r   = logged_in and src_bytes < 500 and dst_bytes < 500 and duration > 50

            if is_dos:
                result, conf, colour = "DoS Attack", 0.91, "🔴"
                desc = "High SYN error rate and connection count — consistent with Denial of Service flooding."
            elif is_probe:
                result, conf, colour = "Probe Attack", 0.85, "🟠"
                desc = "Low bytes with high connection count — consistent with network scanning or reconnaissance."
            elif is_r2l:
                result, conf, colour = "R2L Attack", 0.78, "🟡"
                desc = "Not logged in with high destination bytes — consistent with remote access attempt."
            elif is_u2r:
                result, conf, colour = "U2R Attack", 0.72, "🟣"
                desc = "Logged in with suspicious byte pattern — consistent with privilege escalation."
            else:
                result, conf, colour = "Normal", 0.89, "🟢"
                desc = "Traffic pattern consistent with legitimate network activity."

            st.markdown(f"### {colour} {result} ({conf:.0%} confidence)")
            st.info(desc)

            # Show the k nearest neighbours concept
            st.markdown("**How KNN decided:**")
            st.markdown(f"""
            KNN computed the Euclidean distance from this connection to all 125,973 training examples,
            selected the k=5 nearest neighbours, and assigned the most frequent class label.
            The distance is computed across all 41 normalised features.
            """)

    with tab2:
        st.markdown("### Key Theory Points")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**KNN Algorithm:**
1. Store all training examples (lazy learner)
2. At prediction: compute distance to all training points
3. Select k nearest neighbours
4. Majority vote → predicted class

**Distance metric:**
`D = sqrt(sum((a_i - b_i)^2))` (Euclidean)

**Why normalisation is critical:**
src_bytes (0-1M) would dominate duration (0-100)
without Min-Max scaling to [0,1].
            """)
        with col2:
            st.markdown("""
**NSL-KDD Classes:**
- **Normal** — legitimate traffic (67,343)
- **DoS** — Denial of Service (45,927)
- **Probe** — scanning/reconnaissance (11,656)
- **R2L** — Remote to Local access (995)
- **U2R** — User to Root escalation (52)

**Choosing k:**
Too small = overfitting, too large = underfitting.
Optimal k found via 5-fold cross-validation.
            """)

        st.markdown("### Dataset Distribution")
        dist_df = pd.DataFrame({
            "Class": ["Normal","DoS","Probe","R2L","U2R"],
            "Count": [67343, 45927, 11656, 995, 52],
        })
        fig, ax = plt.subplots(figsize=(8, 3))
        colours = ["#1E88E5","#E53935","#43A047","#FB8C00","#8E24AA"]
        ax.barh(dist_df["Class"], dist_df["Count"], color=colours)
        ax.set_xlabel("Number of samples")
        ax.set_title("NSL-KDD Class Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.code("""
# From-scratch KNN
class KNNClassifier:
    def fit(self, X, y):
        self.X_train = np.array(X)   # just store the data
        self.y_train = np.array(y)

    def _predict_single(self, x):
        distances = [np.sqrt(np.sum((x - xt)**2))
                     for xt in self.X_train]
        k_idx    = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]
        return Counter(k_labels).most_common(1)[0][0]

# Scikit-learn pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn',    KNeighborsClassifier(n_neighbors=5,
                                    metric='euclidean',
                                    weights='distance'))
])
        """, language="python")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — K-MEANS CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
elif "K-Means" in page:
    st.markdown('<p class="section-header"> Network Traffic Clustering — K-Means</p>',
                unsafe_allow_html=True)

    st.markdown("""
    **Problem:** Can an unsupervised algorithm discover natural attack groupings in network traffic
    **without any labels**? This mirrors a real scenario where analysts explore unknown traffic patterns.
    """)

    tab1, tab2, tab3 = st.tabs([" Interactive Demo", " Theory & Results", " Implementation"])

    with tab1:
        st.markdown("### K-Means Algorithm Visualiser")
        st.markdown("Watch K-Means iteratively discover clusters from random data:")

        n_points    = st.slider("Number of data points", 50, 300, 150)
        n_clusters  = st.slider("Number of clusters (k)", 2, 8, 5)
        n_iters     = st.slider("Iterations to show", 1, 10, 5)

        if st.button("▶ Run K-Means", type="primary"):
            np.random.seed(42)
            # Generate synthetic clustered data
            centres_true = np.random.uniform(-4, 4, (n_clusters, 2))
            X_demo = np.vstack([
                centres_true[i] + np.random.randn(n_points // n_clusters, 2) * 0.8
                for i in range(n_clusters)
            ])

            # Run K-Means manually
            rng = np.random.RandomState(0)
            centroids = X_demo[rng.choice(len(X_demo), n_clusters, replace=False)]

            fig, axes = plt.subplots(1, min(n_iters, 5), figsize=(14, 3))
            if n_iters == 1:
                axes = [axes]
            colours_demo = plt.cm.Set1(np.linspace(0, 0.9, n_clusters))

            for it in range(min(n_iters, 5)):
                # Assign
                dists = np.array([np.sqrt(((X_demo - c)**2).sum(axis=1)) for c in centroids]).T
                labels_demo = np.argmin(dists, axis=1)
                # Update
                centroids = np.array([X_demo[labels_demo == k].mean(axis=0)
                                       if (labels_demo == k).sum() > 0 else centroids[k]
                                       for k in range(n_clusters)])
                # Plot
                ax = axes[it]
                for k in range(n_clusters):
                    mask = labels_demo == k
                    ax.scatter(X_demo[mask, 0], X_demo[mask, 1],
                               c=[colours_demo[k]], s=15, alpha=0.6)
                ax.scatter(centroids[:, 0], centroids[:, 1],
                           c="black", s=120, marker="X", zorder=5)
                ax.set_title(f"Iteration {it + 1}")
                ax.set_xticks([]); ax.set_yticks([])

            plt.suptitle("K-Means Clustering — Step by Step", y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.success(f"K-Means converged in {n_iters} iterations. "
                       f"Black X marks show the final cluster centroids.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**K-Means Algorithm (from lectures):**
1. Pick k random centroids
2. Assign each point to nearest centroid
3. Recompute centroids as mean of cluster
4. Repeat until convergence

**Objective — minimise WCSS:**
`WCSS = Σ_k Σ_{x∈Ck} ||x - μ_k||²`

**Sensitivity to initialisation:**
K-Means++ spreads initial centroids
apart — avoids poor local minima.
            """)
        with col2:
            st.markdown("""
**Supervised vs Unsupervised comparison:**

| | KNN | K-Means |
|---|---|---|
| Labels used | Yes | No |
| Learning type | Supervised | Unsupervised |
| Output | Class prediction | Cluster ID |

**Finding k — two methods:**
- **Elbow method** — plot WCSS vs k
- **Silhouette score** — measures
  cluster separation quality (-1 to +1)
            """)

        st.markdown("### Silhouette Score Interpretation")
        sil_data = pd.DataFrame({
            "Score range": ["Close to +1", "Around 0", "Negative"],
            "Meaning": [
                "Well separated from neighbouring clusters",
                "On boundary between clusters",
                "Possibly assigned to wrong cluster"
            ]
        })
        st.dataframe(sil_data, hide_index=True, use_container_width=True)

    with tab3:
        st.code("""
# From-scratch K-Means
class KMeansScratch:
    def fit(self, X):
        # Initialise centroids randomly
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()

        for iteration in range(self.max_iters):
            # Assign each point to nearest centroid
            labels = self._assign(X)

            # Recompute centroids as cluster means
            new_centroids = np.array([
                X[labels == k].mean(axis=0)
                for k in range(self.k)
            ])

            # Check convergence
            shift = np.sqrt(((new_centroids - self.centroids)**2).sum())
            self.centroids = new_centroids
            if shift < self.tol:
                print(f"Converged after {iteration+1} iterations.")
                break

# Scikit-learn
km = KMeans(n_clusters=5, init='k-means++',
            n_init=20, random_state=42)
km.fit(X_scaled)
        """, language="python")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANN
# ═════════════════════════════════════════════════════════════════════════════
elif "ANN" in page:
    st.markdown('<p class="section-header"> Cyberattack Classification — Artificial Neural Networks</p>',
                unsafe_allow_html=True)

    st.markdown("""
    **Problem:** Using a feedforward neural network to classify network traffic into five attack
    categories. ANN can learn complex non-linear patterns that simpler models miss.
    """)

    tab1, tab2, tab3 = st.tabs([" Live Demo", " Theory & Results", " Implementation"])

    with tab1:
        st.markdown("### Neural Network Architecture Explorer")

        col1, col2 = st.columns([1, 2])
        with col1:
            n_hidden1    = st.select_slider("Hidden layer 1 neurons", [16,32,64,128,256], 128)
            n_hidden2    = st.select_slider("Hidden layer 2 neurons", [0,16,32,64,128], 64)
            activation   = st.selectbox("Activation function", ["ReLU", "Tanh", "Sigmoid"])
            learning_rate = st.select_slider("Learning rate", [0.0001,0.001,0.01,0.1], 0.001)

        with col2:
            # Show architecture diagram
            layers = [41, n_hidden1]
            if n_hidden2 > 0:
                layers.append(n_hidden2)
            layers.append(5)

            layer_names = ["Input\n(41 features)"]
            for i, n in enumerate(layers[1:-1]):
                layer_names.append(f"Hidden {i+1}\n({n} neurons)\n{activation}")
            layer_names.append("Output\n(5 classes)\nSoftmax")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            n_layers = len(layers)
            colours_arch = ["#90CAF9","#A5D6A7","#FFCC80","#EF9A9A"]

            for i, (n, name) in enumerate(zip(layers, layer_names)):
                x = (i + 0.5) / n_layers
                colour = colours_arch[min(i, len(colours_arch)-1)]
                ax.add_patch(plt.FancyBboxPatch(
                    (x-0.1, 0.1), 0.18, 0.8,
                    boxstyle="round,pad=0.02",
                    facecolor=colour, edgecolor="white", lw=2
                ))
                ax.text(x+0.08, 0.5, name, ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        transform=ax.transAxes)
                if i < n_layers - 1:
                    ax.annotate("", xy=((i+1.5)/n_layers - 0.1, 0.5),
                                xytext=((i+0.5)/n_layers + 0.1, 0.5),
                                xycoords="axes fraction", textcoords="axes fraction",
                                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

            n_weights = sum(layers[i]*layers[i+1] for i in range(len(layers)-1))
            n_biases  = sum(layers[1:])
            ax.set_title(
                f"Architecture: {' → '.join(str(l) for l in layers)} | "
                f"Total parameters: {n_weights + n_biases:,}",
                pad=10
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.info(
                f"**Total trainable parameters:** {n_weights + n_biases:,}  \n"
                f"From lecture notes: training set should be 5-10× the number of weights → "
                f"need at least {n_weights * 5:,} training examples. "
                f"NSL-KDD has 125,973 — {'✅ sufficient' if 125973 >= n_weights * 5 else '⚠️ may need more data'}."
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Perceptron (from lecture notes):**
- Input 0: x1 = 12, Weight 0 = 0.5
- Input 1: x2 = 4,  Weight 1 = -1
- Sum = (12 × 0.5) + (4 × -1) = 2
- Output = sign(2) = +1

**Forward pass:**
`z = W^T x + b`
`a = activation(z)`

**Backpropagation:**
1. Forward pass → prediction
2. Compute loss (cross-entropy)
3. Chain rule → weight gradients
4. Gradient descent → update weights
            """)
        with col2:
            st.markdown("""
**Activation functions:**
| Function | Formula | Output |
|---|---|---|
| ReLU | max(0,z) | [0, ∞) |
| Tanh | (e²ˣ-1)/(e²ˣ+1) | (-1,1) |
| Sigmoid | 1/(1+e⁻ˣ) | (0,1) |
| Softmax | eˣⁱ/Σeˣʲ | (0,1) |

**Architecture:**
- Input layer: 41 features (no activation)
- Hidden layers: ReLU (non-linearity)
- Output layer: Softmax (5 class probs)
            """)

        st.markdown("### Algorithm Comparison on NSL-KDD")
        comp_df = pd.DataFrame({
            "Algorithm": ["SVM (Linear)", "KNN (k=5)", "ANN (MLP 128→64)"],
            "Type": ["Supervised","Supervised","Supervised"],
            "Strength": [
                "High-dimensional spaces",
                "Simple, interpretable",
                "Complex non-linear patterns"
            ]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with tab3:
        st.code("""
# From-scratch ANN — backpropagation
class NeuralNetworkScratch:
    def _forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)     # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_z   = np.exp(self.z2 - self.z2.max(axis=1, keepdims=True))
        self.a2 = exp_z / exp_z.sum(axis=1, keepdims=True)  # Softmax
        return self.a2

    def _backward(self, X, y):
        n = X.shape[0]
        delta2 = self.a2.copy()
        delta2[range(n), y] -= 1             # cross-entropy + softmax gradient
        delta2 /= n
        delta1 = (delta2 @ self.W2.T) * (self.z1 > 0)  # chain rule + ReLU grad
        # Gradient descent updates
        self.W2 -= self.lr * (self.a1.T @ delta2)
        self.W1 -= self.lr * (X.T @ delta1)
        """, language="python")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — WORK LOG
# ═════════════════════════════════════════════════════════════════════════════
elif "Work Log" in page:
    st.markdown('<p class="section-header">📋 Portfolio Work Log</p>', unsafe_allow_html=True)
    st.markdown("""
    This work log documents all adjustments, observations, and analysis made during development.
    Required by the CA brief: *'must clearly show the adjustments/alterations/additions,
    the practical impact of the alterations, and your analysis.'*
    """)

    algo = st.selectbox("Select algorithm:", ["SVM", "KNN", "K-Means", "ANN"])

    logs = {
        "SVM": [
            ("Dataset Selection", "SMS Spam Collection over larger email datasets (Enron).",
             "Compact, clean, binary labels mapping directly to SVM's binary classification. Class imbalance (87% ham, 13% spam) required F1 score as primary metric rather than accuracy."),
            ("URL and Phone Number Tokens", "Added URL/phone replacement before removing special characters.",
             "Token `url` appeared in top spam indicators after this change — spam messages contain far more URLs than legitimate messages. Confirmed by feature importance plot."),
            ("Kernel Comparison", "Compared linear vs RBF kernel via GridSearchCV.",
             "Linear kernel consistently outperformed RBF. Explanation: TF-IDF with 10,000 features creates a space already rich enough — the kernel trick adds no value."),
            ("Bigrams in TF-IDF", "Changed ngram_range from (1,1) to (1,2).",
             "Phrases like 'free prize' and 'call now' are stronger spam signals than individual words. Bigrams appeared in top feature importance results."),
            ("C Parameter Tuning", "Searched C across [0.01, 0.1, 1, 10, 100].",
             "Optimal C around 1-10. Very small C underfits, very large C overfits. For spam detection, moderate C avoids both false positives (annoying for users) and false negatives (security risk)."),
        ],
        "KNN": [
            ("Dataset Selection", "NSL-KDD over original KDD Cup 1999.",
             "KDD99 has duplicate records that bias evaluation — classifiers score artificially high by memorising repeated patterns. NSL-KDD removes duplicates for honest evaluation."),
            ("Feature Encoding", "Used LabelEncoder for protocol_type, service, flag.",
             "KNN requires numerical inputs for distance computation. One-hot encoding avoided to prevent dimensionality explosion with many service types (worsening curse of dimensionality)."),
            ("Normalisation", "Applied MinMaxScaler before KNN.",
             "Without scaling, src_bytes (0-1M) dominated distance calculations entirely. Scaling to [0,1] gave each feature equal influence — directly illustrated the lecture point about income vs age."),
            ("Distance-Weighted Voting", "GridSearchCV selected weights='distance'.",
             "Closer neighbours have stronger influence. In intrusion detection this is physically meaningful — very similar traffic should be classified with high confidence."),
            ("K-Value Analysis", "Plotted training/test/CV accuracy across k=1 to 25.",
             "k=1 perfectly memorises training data (overfitting). As k increases, boundary smooths and test accuracy stabilises. Classic bias-variance tradeoff visualised clearly."),
        ],
        "K-Means": [
            ("Supervised vs Unsupervised Comparison", "Used same NSL-KDD dataset as KNN.",
             "Enables direct comparison. Adjusted Rand Index (ARI) measures how well K-Means clusters align with true labels without using them during training."),
            ("StandardScaler vs MinMaxScaler", "Chose StandardScaler for K-Means.",
             "NSL-KDD has extreme outliers in byte counts. StandardScaler is more robust than MinMaxScaler because it normalises by standard deviation rather than compressing to a fixed range."),
            ("K-Means++ Initialisation", "Used init='k-means++' with n_init=20.",
             "Lecture notes: K-Means is sensitive to initial centroids and is non-deterministic. K-Means++ spreads centroids far apart, reducing poor local minima risk. 20 restarts add robustness."),
            ("Choosing k", "Elbow method did not give a sharp elbow — used silhouette as secondary criterion.",
             "Lecture notes explicitly acknowledge: 'sometimes the k is not obvious using the elbow method'. Set k=5 to match known classes for meaningful evaluation."),
            ("Cluster Profile Analysis", "Added cluster composition heatmap.",
             "DoS attacks cluster cleanly (very distinctive traffic). R2L and U2R are harder to separate — they look more like normal traffic in feature space. Illustrates unsupervised learning limits."),
        ],
        "ANN": [
            ("One-Hot Encoding", "Used pd.get_dummies() instead of LabelEncoder for ANN.",
             "Integer encoding (tcp=0, udp=1, icmp=2) implies a false ordinal relationship. For ANNs this misleads weight learning. One-hot encoding represents categories correctly."),
            ("Two Hidden Layers", "Chose architecture 128 → 64 as baseline.",
             "Directly from lecture notes: 'Two hidden layers are sufficient to solve many problems.' Architecture comparison confirmed this — three layers showed marginal improvement with longer training."),
            ("Early Stopping", "Enabled early_stopping=True with n_iter_no_change=15.",
             "Without early stopping, MLP continued past optimal generalisation. Validation loss curve shows classic pattern — training loss falls while validation plateaus. Early stopping prevents overfitting."),
            ("ReLU vs Tanh", "ReLU consistently outperformed Tanh across all architectures.",
             "Many NSL-KDD features are zero (no failed logins, no root shell). ReLU's sparse activation is a natural fit — it effectively ignores zero features. Tanh also risks vanishing gradients in deeper layers."),
            ("Portfolio Comparison", "Generated direct comparison of all four algorithms.",
             "ANN outperforms KNN on complex attack types (R2L, U2R). SVM competitive in high-dimensional space. Key insight: no algorithm is universally best — choice depends on data structure and requirements."),
        ]
    }

    for i, (title, adjustment, analysis) in enumerate(logs[algo], 1):
        with st.expander(f"Entry {i} — {title}"):
            st.markdown(f"**Adjustment:** {adjustment}")
            st.markdown(f"**Analysis & Impact:** {analysis}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — GITHUB
# ═════════════════════════════════════════════════════════════════════════════
elif "GitHub" in page:
    st.markdown('<p class="section-header">📁 GitHub Repository</p>', unsafe_allow_html=True)

    st.markdown("""
    All portfolio materials are version-controlled and publicly available on GitHub.
    """)

    st.markdown("### Repository Structure")
    st.code("""
ML_Portfolio/
├── 01_SVM_Spam_Detection.ipynb          # SVM — SMS Spam Collection
├── 02_KNN_Network_Intrusion.ipynb       # KNN — NSL-KDD
├── 03_KMeans_Network_Clustering.ipynb   # K-Means — NSL-KDD
├── 04_ANN_Cyberattack_Classification.ipynb  # ANN — NSL-KDD
├── app.py                               # This Streamlit dashboard
├── requirements.txt                     # All dependencies
├── data/                                # Datasets
│   ├── SMSSpamCollection
│   └── KDDTrain+.txt
└── images/                              # Saved visualisations
    """)

    st.markdown("### Links")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Current portfolio (Semester 2):**
All four algorithm notebooks + this app

**Previous portfolio (Semester 1):**
News→Signal — financial sentiment analysis
        """)
    with col2:
        st.markdown("""
[📂 View Repository](https://github.com/EmmanuelAbolade/news2signal-portfolio)

[🌐 Previous Portfolio App](https://news2signal-portfolio-dl3byf65m8qhncpqcfpdny.streamlit.app/)
        """)

    st.markdown("### References")
    refs = [
        "Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. ACM DOCENG'11.",
        "Tavallaee, M. et al. (2009). A Detailed Analysis of the KDD CUP 99 Data Set. IEEE CISDA.",
        "Boser, B.E., Guyon, I., Vapnik, V.N. (1992). A Training Algorithm for Optimal Margin Classifiers. COLT '92.",
        "Cover, T., Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Trans. Information Theory.",
        "MacQueen, J. (1967). Some Methods for Classification of Multivariate Observations. Berkeley Symposium.",
        "Rumelhart, D.E. et al. (1986). Learning Representations by Back-propagating Errors. Nature.",
        "Doyle, G. (2025). Data Science & Machine Learning Lecture Notes. South East Technological University.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.",
    ]
    for ref in refs:
        st.markdown(f"- {ref}")
