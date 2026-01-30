# ML Foundations

Personal machine learning practice repository following a structured learning roadmap from mathematical foundations through deep learning, NLP, computer vision, and reinforcement learning.

## Repository Structure

```
ml-foundations/
├── 01_linear_algebra/          # Phase 1: Vectors, matrices, SVD, PCA
├── 02_calculus_optimization/   # Phase 1: Gradients, chain rule, gradient descent
├── 03_probability_statistics/  # Phase 1: Distributions, Bayes, MLE
├── 04_supervised_learning/     # Phase 2: Linear/logistic regression, SVM
├── 05_unsupervised_learning/   # Phase 2: K-means, GMMs, PCA applications
├── 06_neural_networks/         # Phase 2: MLPs, backpropagation, PyTorch basics
├── 07_transformers/            # Phase 3: Self-attention, encoder/decoder, pretraining
├── 08_nlp/                     # Phase 4: Tokenization, embeddings, BERT fine-tuning
├── 09_computer_vision_3d/      # Phase 5: Camera models, epipolar geometry, MVG
├── 10_reinforcement_learning/  # Phase 6: MDPs, Q-learning, policy gradients
├── references/                 # Reference materials and notes
├── venv/                       # Python 3.11 virtual environment
└── requirements.txt            # Python dependencies
```

## Learning Phases

| Phase | Topic | Duration | Focus |
|-------|-------|----------|-------|
| 1 | Mathematical Foundations | 8-12 weeks | Linear algebra, calculus, probability, optimization |
| 2 | Classical ML & Basic DL | 6-8 weeks | Supervised/unsupervised learning, MLPs |
| 3 | Transformers & Modern DL | 8-10 weeks | Self-attention, encoder/decoder architectures |
| 4 | NLP Foundations | 5-6 weeks | Tokenization, embeddings, BERT/GPT |
| 5 | 3D Computer Vision | 8-10 weeks | Camera models, multi-view geometry, MVG |
| 6 | Reinforcement Learning | 6-8 weeks | MDPs, Q-learning, policy gradients |

## Setup

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter lab
```

## Key Math References

These references from CS3264 provide solid mathematical background:

- **Bishop's PRML** - Pattern Recognition and Machine Learning (comprehensive ML theory)
- **Matrix Cookbook** - Quick reference for matrix calculus and linear algebra identities
- **MML Book** - Mathematics for Machine Learning (foundational math for ML)

### Core Mathematical Concepts

#### Linear Algebra
- Vector spaces, linear transformations
- Eigenvalues/eigenvectors, SVD
- Matrix decompositions (LU, QR, Cholesky)
- Normal equations: `w* = (X'X)^{-1}X'y`

#### Calculus & Optimization
- Gradients, Jacobians, Hessians
- Chain rule in vector form
- Gradient descent: `w_{t+1} = w_t - α∇L(w_t)`
- Convexity and convergence guarantees

#### Probability & Statistics
- Random variables, expectations, variance
- Common distributions (Gaussian, Bernoulli, Categorical)
- Bayes' rule: `P(θ|D) ∝ P(D|θ)P(θ)`
- Maximum Likelihood Estimation (MLE)

## Implementation Guidelines

1. **Always tie code to equations** - For every implementation, know which objective it optimizes
2. **Re-derive key results** - Don't just accept formulas; derive gradients and losses
3. **Build simplified versions first** - Implement toy problems before scaling up
4. **Explain concepts** - If you can't explain it without notes, revisit fundamentals

## Resources

### Free Online Resources
- [Bloomberg FOML](https://bloomberg.github.io/foml/) - Foundations of ML
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) - Transformers & NLP
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Deep learning framework
- [Sutton & Barto RL Book](http://incompleteideas.net/book/the-book-2nd.html) - RL fundamentals

### Books
- Hartley & Zisserman - Multiple View Geometry in Computer Vision
- Bishop - Pattern Recognition and Machine Learning
- Goodfellow et al. - Deep Learning

## License

Personal learning repository.
