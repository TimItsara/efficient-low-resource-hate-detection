# Efficient Low-Resource Hate Detection

This repository contains code for training and evaluating hate speech detection models in multiple languages using active learning and fine-tuning approaches with transformer-based models.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage Guide](#usage-guide)
  - [0. New Data Collection (Optional)](#0-new-data-collection-optional)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Active Learning Selection](#2-active-learning-selection)
  - [3. Model Fine-tuning](#3-model-fine-tuning)
  - [4. Evaluation](#4-evaluation)
- [Supported Languages](#supported-languages)
- [Configuration](#configuration)
- [Results](#results)
- [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
- [Quick Reference](#quick-reference)
- [FAQ](#frequently-asked-questions-faq)
- [Data Sources & Attribution](#data-sources--attribution)

## Project Overview

This project implements efficient hate speech detection for low-resource scenarios across multiple languages including English, Spanish, Portuguese, Italian, French, Arabic, Hindi, Chinese, and Thai. The approach uses:

- **Active Learning**: Intelligently selecting training samples to maximize model performance with minimal labeled data
- **Multilingual Transformers**: Fine-tuning XLM-RoBERTa and similar models
- **HateCheck Evaluation**: Testing model robustness using the HateCheck benchmark

### Original Repository

This project is based on the work by Paul Röttger et al.:
- **Original Repository**: [efficient-low-resource-hate-detection](https://github.com/paul-rottger/efficient-low-resource-hate-detection/tree/master)
- **Paper**: [Data-Efficient Strategies for Expanding Hate Speech Detection into Under-Resourced Languages](https://aclanthology.org/2022.emnlp-main.383/) (EMNLP 2022)

This repository extends the original work with additional languages (Chinese, French, Thai), enhanced data collection tools, and improved documentation.

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) GPU with CUDA support or Apple Silicon with MPS
  - **Minimum GPU Memory**: 8GB for `xlm-roberta-base` with batch size 16
  - **Recommended**: 16GB+ for larger models or batch sizes
  - **CPU Training**: Possible but significantly slower (10-20x)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/TimItsara/efficient-low-resource-hate-detection.git
cd efficient-low-resource-hate-detection
```

2. **Create a virtual environment**
```bash
python -m venv hate_detection_env
source hate_detection_env/bin/activate  # On macOS/Linux
# OR
hate_detection_env\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Repository Structure

```
efficient-low-resource-hate-detection/
├── 0_data/                          # Data directory
│   ├── hatecheck/                   # HateCheck benchmark datasets (multiple languages)
│   └── main/
│       ├── 0_raw/                   # Raw datasets
│       ├── 1_clean/                 # Cleaned and processed datasets
│       └── 2_active_learning/       # Active learning selected samples
│
├── 1_dataloading/                   # Data preprocessing scripts
│   ├── 1_wrangling_data.ipynb      # Data cleaning and formatting
│   └── 2_active_learning_selection.ipynb  # Active learning sample selection
│
├── 2_finetuning/                    # Model training
│   ├── finetune_and_test.py        # Main training script
│   ├── config_english.json         # Example configuration file
│   ├── config/                      # Multiple configuration files
│   └── launch_*.sh                  # Batch training scripts
│
├── 3_evaluation/                    # Results and evaluation
│   ├── evaluate_results.ipynb      # Analysis notebooks
│   └── results/                     # Model outputs
│
├── new-data-collection/             # Tools for collecting new datasets
│   ├── chinese/
│   │   ├── smart_chinese_collector.py      # Reddit scraper for Chinese
│   │   ├── create_dataset_splits.py        # Create train/dev/test splits
│   │   ├── convert_jsonl_to_csv.py         # Format converter
│   │   ├── chinese_8000.csv                # Pre-collected dataset
│   │   └── FREE_RESOURCES_GUIDE.md         # Collection guide
│   ├── french/
│   │   ├── fr_hf.csv                       # French HuggingFace dataset
│   │   ├── script.py                       # Processing script
│   │   └── README.md                       # Dataset documentation
│   └── thai/
│       ├── HateThaiSent.csv                # Thai hate speech dataset
│       ├── script.py                       # Processing script
│       ├── readme.md                       # Dataset documentation
│       ├── ThaiCaps/                       # Multi-task framework
│       └── Baselines/                      # Baseline models
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Usage Guide

### Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                             │
└─────────────────────────────────────────────────────────────────┘

Phase 0: Data Collection (Optional - for new languages)
├── Chinese: smart_chinese_collector.py → collect Reddit data
├── French: Download from HuggingFace → script.py
└── Thai: Download from GitHub → script.py
    │
    ↓
Phase 1: Data Preparation
├── 1_wrangling_data.ipynb → Clean & format datasets
└── 2_active_learning_selection.ipynb → Select training samples
    │
    ↓
Phase 2: Model Training
├── launch_ft1.sh → Train English models (all sizes, seeds)
├── launch_ft2.sh → Train multilingual models
    │
    ↓
Phase 3: Evaluation
├── launch_test.sh → Test on held-out test sets
├── launch_test_0shot.sh → Zero-shot cross-lingual evaluation
├── launch_test_hatecheck.sh → Robustness testing
└── evaluate_results.ipynb → Analyze and visualize results
```

### 0. New Data Collection (Optional)

If you want to collect or prepare new datasets for additional languages, the `new-data-collection/` directory contains tools and datasets for Chinese, French, and Thai.

#### Chinese Data Collection

The Chinese data collection toolkit includes a smart Reddit scraper that respects rate limits and automatically handles API restrictions.

**Files:**
- `smart_chinese_collector.py` - Intelligent Reddit scraper for Chinese content
- `create_dataset_splits.py` - Splits collected data into train/dev/test sets
- `convert_jsonl_to_csv.py` - Converts JSONL to CSV format
- `chinese_8000.csv` - Pre-collected dataset (8000 samples)
- `FREE_RESOURCES_GUIDE.md` - Guide for collecting data using free resources

**Step 1: Collect Chinese data from Reddit**

```bash
cd new-data-collection/chinese

# Collect 8000 Chinese language items (takes 20-30 hours)
python smart_chinese_collector.py \
  --subs China Sino ChineseLanguage sino_zh \
  --mode hot \
  --target 8000 \
  --chinese-min-chars 15 \
  --chinese-min-ratio 0.6 \
  --sleep 3.5 \
  --wait-on-429 600 \
  --out chinese_8000.jsonl
```

**Parameters explained:**
- `--subs`: Subreddits to scrape
- `--mode`: Collection mode (`hot`, `new`, `top`)
- `--target`: Target number of items
- `--chinese-min-chars`: Minimum Chinese characters required
- `--chinese-min-ratio`: Minimum ratio of Chinese to total characters
- `--sleep`: Base sleep time between requests (seconds)
- `--wait-on-429`: Wait time when rate limited (seconds, default 600 = 10 minutes)

**Step 2: Convert JSONL to CSV**

```bash
python convert_jsonl_to_csv.py
```

**Step 3: Create dataset splits**

```bash
python create_dataset_splits.py
```

This creates:
- `chinese_hatesent/dev_500.csv` - Development set (500 samples)
- `chinese_hatesent/test_2000.csv` - Test set (2000 samples)
- `chinese_hatesent/train/train_{size}_rs{seed}.csv` - Multiple training sets

Training sizes: 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000  
Random seeds: 1-10 (for robust evaluation)

**⚠️ Important: Data Annotation**

The collected Reddit data is **unlabeled**. Before using it for training, you need to annotate it:

1. **Manual Annotation**: 
   - Review each text and assign label (0 = non-hateful, 1 = hateful)
   - Use annotation tools like [Label Studio](https://labelstud.io/) or [Prodigy](https://prodi.gy/)
   
2. **Semi-Automated Annotation**:
   - Use a pre-trained multilingual model to get initial labels
   - Manually review and correct predictions
   - Example:
   ```bash
   # Use existing model to pre-label
   python finetune_and_test.py \
     --model_name_or_path <your-pretrained-model> \
     --test_file chinese_8000.csv \
     --do_predict \
     --output_dir ./prelabeled
   ```

3. **Crowdsourcing**: Use platforms like Amazon MTurk or similar for annotation

**Annotation Guidelines:**
Label as hateful (1) if the text contains:
- Attacks on individuals/groups based on protected characteristics
- Dehumanizing language
- Calls for violence or exclusion
- Toxic stereotyping

Label as non-hateful (0) for:
- Neutral discussions
- Criticism without hate speech
- Regular conversations

#### French Data Collection

The French dataset is sourced from HuggingFace and includes multiple merged datasets.

**Dataset Source:** [French Hate Speech Superset on HuggingFace](https://huggingface.co/datasets/manueltonneau/french-hate-speech-superset)

**Dataset Information:**
- **Size**: 18,071 annotated posts
- **Sources**: CONAN, MLMA, sexism detection corpus
- **Format**: Text classification with binary labels (hateful/non-hateful)
- **Language**: French
- **Author**: Manuel Tonneau

**Files:**
- `fr_hf.csv` - French hate speech dataset from HuggingFace
- `script.py` - Preprocessing and dataset splitting script
- `README.md` - Detailed dataset documentation

**Usage:**

```bash
cd new-data-collection/french

# Process and split the French dataset
python script.py \
  --input ./fr_hf.csv \
  --out_dir ./datasets/french_hatesent \
  --dev 500 \
  --test 2000 \
  --Ns 10 20 30 40 50 100 200 300 400 500 1000 2000 \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

**Parameters:**
- `--input`: Path to input CSV file
- `--out_dir`: Output directory for processed datasets
- `--dev`: Development set size (default: 500)
- `--test`: Test set size (default: 2000)
- `--Ns`: Training set sizes to generate
- `--seeds`: Random seeds for multiple splits (ensures robust evaluation)

The script will:
1. Normalize column names (text, label)
2. Convert labels to binary format (0 = non-hateful, 1 = hateful)
3. Clean and deduplicate data
4. Create stratified train/dev/test splits
5. Generate multiple training subsets with different sizes and random seeds

**Dataset composition:**
- CONAN - Counter Narratives dataset
- MLMA - Multilingual and Multi-Aspect Hate Speech
- Sexism Detection corpus

#### Thai Data Collection

The Thai dataset uses HateThaiSent with sentiment analysis annotations.

**Dataset Source:** [HateThaiSent on GitHub](https://github.com/dsmlr/HateThaiSent)

**Dataset Information:**
- **Size**: 7,597 messages
- **Labels**: Hate speech + Sentiment (Positive/Neutral/Negative)
- **Language**: Thai
- **Source**: HateThaiSent dataset with sentiment annotations
- **Authors**: Krishanu Maity, A.S. Poornash, Shaubhik Bhattacharya, et al.

**Files:**
- `HateThaiSent.csv` - Main dataset file
- `script.py` - Preprocessing script
- `readme.md` - Dataset documentation
- `ThaiCaps/` - Multi-tasking framework implementation
- `Baselines/` - Baseline model implementations

**Usage:**

```bash
cd new-data-collection/thai

# Process and split the Thai dataset
python script.py \
  --input ./HateThaiSent.csv \
  --out_dir ./datasets/thai_hatesent \
  --dev 500 \
  --test 2000 \
  --Ns 10 20 30 40 50 100 200 300 400 500 1000 2000 \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

**Parameters:**
- `--input`: Path to HateThaiSent CSV file
- `--out_dir`: Output directory for processed datasets
- `--dev`: Development set size (default: 500)
- `--test`: Test set size (default: 2000)
- `--Ns`: Training set sizes to generate
- `--seeds`: Random seeds for multiple splits (10 seeds = 10 versions of each training size)

**Output structure:**
```
datasets/thai_hatesent/
├── dev_500.csv
├── test_2000.csv
├── train_full.csv
└── train/
    ├── train_10_rs1.csv to train_10_rs10.csv
    ├── train_20_rs1.csv to train_20_rs10.csv
    ├── train_50_rs1.csv to train_50_rs10.csv
    ...
    └── train_2000_rs1.csv to train_2000_rs10.csv
```

**Multi-task Learning:**

The Thai dataset supports both hate speech detection and sentiment analysis:

```bash
# Run the ThaiCaps framework
cd ThaiCaps
CUDA_VISIBLE_DEVICES=0 python run.py

# Run baseline models
cd ../Baselines
CUDA_VISIBLE_DEVICES=0 python run.py
```

**Citation:**
```bibtex
@article{HateThaiSent,
    author = {Krishanu Maity et al.},
    title = {{HateThaiSent: Sentiment-Aided Hate Speech Detection in Thai Language}},
    journal = {IEEE Transactions on Computational Social Systems},
    year = {2024},
}
```

#### Moving New Data to Main Pipeline

After collecting and processing new data, integrate it into the main pipeline:

```bash
# Example: Move Chinese data to main data directory
cp -r new-data-collection/chinese/chinese_hatesent ../0_data/main/1_clean/

# Example: Move French data
cp -r new-data-collection/french/french_hatesent ../0_data/main/1_clean/

# Example: Move Thai data
cp -r new-data-collection/thai/thai_hatesent ../0_data/main/1_clean/
```

Then update your configuration files to use the new datasets in training.

### 1. Data Preparation

#### Step 1.1: Data Wrangling

The first step is to clean and format raw datasets into a standardized format.

**Required CSV Format:**
All datasets must have these columns:
- `text`: The text content to classify
- `label`: Binary label (1 = hateful, 0 = non-hateful)

Example:
```csv
text,label
"This is an example of hate speech",1
"This is normal content",0
```

```bash
# Open the data wrangling notebook
jupyter notebook 1_dataloading/1_wrangling_data.ipynb
```

**What this does:**
- Loads raw datasets from `0_data/main/0_raw/`
- Standardizes column names (text, label)
- Converts labels to binary format (1 = hateful, 0 = non-hateful)
- Splits data into train/dev/test sets
- Saves cleaned data to `0_data/main/1_clean/`

**Supported datasets:**
- `dyn21_en` - Dynabench 2021 (English)
- `fou18_en` - Founta 2018 (English)
- `ken20_en` - Kennedy 2020 (English)
- `bas19_es` - Basile 2019 (Spanish)
- `for19_pt` - Fortuna 2019 (Portuguese)
- `san20_it` - Sanguinetti 2020 (Italian)
- `ous19_ar` - Ousidhoum 2019 (Arabic)
- `ous19_fr` - Ousidhoum 2019 (French)
- `has19_hi`, `has20_hi`, `has21_hi` - Various Hindi datasets

#### Step 1.2: Active Learning Selection

Select optimal training samples using active learning strategies to maximize performance with minimal labeled data.

**What is Active Learning?**
Active learning intelligently selects the most informative samples for training, rather than random sampling. This is especially valuable in low-resource scenarios where labeling data is expensive.

**How it works:**
1. Trains an initial model on a small seed dataset
2. Uses the model to predict on unlabeled data
3. Selects samples where the model is most uncertain (high entropy, low confidence)
4. These "informative" samples are added to the training set
5. Process repeats until desired dataset size is reached

```bash
# Open the active learning notebook
jupyter notebook 1_dataloading/2_active_learning_selection.ipynb
```

This creates training sets of various sizes (10, 20, 50, 100, 200, 500, 1000, 2000, etc.) with different random seeds for robust evaluation.

**Note:** If you don't want to use active learning, you can create random splits instead by sampling randomly from your data.

### 2. Model Fine-tuning

**Expected Training Time:**
- **Small dataset (100 samples)**: ~2-5 minutes
- **Medium dataset (1000 samples)**: ~10-20 minutes  
- **Large dataset (10000+ samples)**: ~1-3 hours
- Times based on GPU (V100/A100). CPU training will be 10-20x slower.

#### Option A: Single Model Training

Use a configuration file to train a single model:

```bash
cd 2_finetuning

python finetune_and_test.py \
    --model_name_or_path xlm-roberta-base \
    --train_file ../0_data/main/1_clean/dyn21_en/train/train_2000_rs1.csv \
    --validation_file ../0_data/main/1_clean/dyn21_en/dev_500.csv \
    --test_file ../0_data/main/1_clean/dyn21_en/test_2000.csv \
    --output_dir ./results/english_finetuning \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_seq_length 128
```

Or use a pre-configured JSON file:

```bash
python finetune_and_test.py config_english.json
```

#### Option B: Batch Training with Shell Scripts

For systematic experiments across multiple configurations, use the launch scripts to run comprehensive training pipelines.

**📋 Recommended Workflow:**

```
1. launch_ft1.sh      → Train models on English datasets (all sizes, all seeds)
2. launch_ft2.sh      → Train models on additional languages (Spanish, Portuguese, etc.)
3. launch_test.sh     → Evaluate trained models on test sets
4. launch_test_0shot.sh → Zero-shot cross-lingual evaluation (English models on other languages)
5. launch_test_hatecheck.sh → Evaluate on HateCheck benchmark for robustness
```

**Step 1: Configure the Scripts**

Before running, edit each script to set your paths:

```bash
# Edit the launch scripts
nano launch_ft1.sh
```

Update these variables in **ALL** scripts:
```bash
# Set your virtual environment path
source /path/to/your/hate_detection_env/bin/activate

# Set your data path (absolute path required)
DATA=/path/to/efficient-low-resource-hate-detection
```

**Step 2: Make Scripts Executable**

```bash
cd 2_finetuning
chmod +x launch_ft1.sh launch_ft2.sh launch_test.sh launch_test_hatecheck.sh
```

**Step 3: Run Training Pipeline**

**Phase 1: Train English Models**

```bash
# Train on English datasets (dyn21_en, fou18_en, ken20_en)
# Tests multiple training sizes: 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, etc.
# With 10 random seeds each (rs1-rs10)
./launch_ft1.sh
```

**What this does:**
- Trains models on 3 English datasets
- For each dataset: multiple training sizes (10 to 20000 samples)
- For each size: 10 different random seeds
- Total: Could be 100+ model training runs
- **Expected time**: Several hours to days depending on GPU

**Phase 2: Train Additional Language Models**

```bash
# Train on your languages of interest
# Edit launch_ft2.sh first to select specific languages
./launch_ft2.sh
```

**Typical languages in ft2:**
- Spanish (`bas19_es`)
- Portuguese (`for19_pt`)
- French (`french_fr`)
- Chinese (`chinese_cn`)
- Thai (`thai_th`)
- Arabic, Hindi, Italian, etc.

**Phase 3: Evaluate on Test Sets**

```bash
# Test all trained models on their respective test sets
# Generates predictions and performance metrics
./launch_test.sh
```

**What this does:**
- Loads each trained model
- Runs predictions on test set (usually 2000 samples)
- Saves results to `3_evaluation/results/`
- No training - only inference

**Phase 4: Zero-Shot Cross-Lingual Evaluation**

```bash
# Test English-trained models on other languages WITHOUT fine-tuning
# Evaluates multilingual transfer capabilities
./launch_test_0shot.sh
```

**What this does:**
- Takes English-trained models (from Phase 1)
- Tests them directly on Spanish and Portuguese datasets
- **No fine-tuning** on target languages (true zero-shot)
- Evaluates multilingual transformer's cross-lingual transfer ability
- Useful for comparing with fine-tuned models (Phase 2)

**Example:**
```bash
# English model trained on dyn21_en → tested on bas19_es (Spanish)
# Shows how well XLM-RoBERTa transfers knowledge across languages
```

**Phase 5: HateCheck Robustness Testing**

```bash
# Evaluate models on HateCheck benchmark
# Tests edge cases and robustness
./launch_test_hatecheck.sh
```

**What this does:**
- Tests models on HateCheck functional tests
- Available for: Arabic, Chinese, German, Spanish, French, Hindi, Italian, Dutch, Polish, Portuguese, Thai
- Identifies systematic failures (e.g., negation handling, identity mentions)
- Results saved to `3_evaluation/results/hatecheck_*/`

**Available Launch Scripts:**

| Script | Purpose | When to Use | Approx. Time |
|--------|---------|-------------|--------------|
| `launch_ft1.sh` | Train English models | First run, English experiments | Hours to days |
| `launch_ft2.sh` | Train multilingual models | After ft1, for other languages | Hours to days |
| `launch_test.sh` | Test trained models | After training completes | Minutes to hours |
| `launch_test_0shot.sh` | Zero-shot cross-lingual test | Compare transfer vs fine-tuning | Minutes |
| `launch_test_hatecheck.sh` | HateCheck evaluation | Final evaluation step | Minutes |

**💡 Tips for Running Batch Scripts:**

1. **Start Small**: Test with a single model configuration first to verify your setup
2. **Monitor Progress**: Use `tail -f` to watch output:
   ```bash
   ./launch_ft1.sh > training.log 2>&1 &
   tail -f training.log
   ```
3. **Resume Interrupted Training**: Scripts check for existing models and can skip completed runs
4. **Customize Languages**: Edit `launch_ft2.sh` to train only languages you need:
   ```bash
   # In launch_ft2.sh, comment out languages you don't need
   # for model in bas19_es for19_pt french_fr thai_th; do  # Only Spanish, Portuguese, French, Thai
   ```
5. **Parallel Training**: If you have multiple GPUs, run different scripts in parallel:
   ```bash
   # Terminal 1
   CUDA_VISIBLE_DEVICES=0 ./launch_ft1.sh
   
   # Terminal 2
   CUDA_VISIBLE_DEVICES=1 ./launch_ft2.sh
   ```

**Example: Complete Workflow**

```bash
cd 2_finetuning

# 1. Configure paths in all scripts
nano launch_ft1.sh  # Update DATA and virtual env path
nano launch_ft2.sh  # Update DATA and virtual env path
nano launch_test.sh # Update DATA and virtual env path

# 2. Make executable
chmod +x launch_*.sh

# 3. Train English models (long running)
nohup ./launch_ft1.sh > ft1.log 2>&1 &

# 4. Monitor progress
tail -f ft1.log

# 5. Once ft1 completes, train other languages
nohup ./launch_ft2.sh > ft2.log 2>&1 &

# 6. After all training, run tests
./launch_test.sh > test.log 2>&1 &

# 7. Finally, run HateCheck evaluation
./launch_test_hatecheck.sh

# 8. Analyze results
cd ../3_evaluation
jupyter notebook evaluate_results.ipynb
```

#### Key Training Arguments

**Model Arguments:**
- `--model_name_or_path`: HuggingFace model ID (e.g., `xlm-roberta-base`, `cardiffnlp/twitter-xlm-roberta-base`)
- `--tokenizer_name`: Custom tokenizer (optional)
- `--cache_dir`: Model cache directory

**Data Arguments:**
- `--train_file`: Path to training CSV
- `--validation_file`: Path to validation CSV
- `--test_file`: Path to test CSV
- `--max_seq_length`: Maximum sequence length (default: 128)
- `--use_class_weights`: Enable class weighting for imbalanced datasets
- `--store_prediction_logits`: Save prediction probabilities

**Training Arguments:**
- `--output_dir`: Directory for model outputs
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Accumulation steps for larger effective batch size
- `--do_train`: Enable training
- `--do_eval`: Enable evaluation on validation set
- `--do_predict`: Enable prediction on test set

### 3. Evaluation

#### Option A: Evaluate Training Results

```bash
# Open evaluation notebook
jupyter notebook 3_evaluation/evaluate_results.ipynb
```

This notebook:
- Loads prediction results from `3_evaluation/results/`
- Calculates metrics (F1, precision, recall)
- Generates comparison plots
- Analyzes performance across different training set sizes

#### Option B: HateCheck Evaluation

Test model robustness using the HateCheck benchmark:

```bash
cd 2_finetuning

./launch_test_hatecheck.sh
```

This evaluates models on:
- `hatecheck_cn` - Chinese
- `hatecheck_es` - Spanish
- `hatecheck_fr` - French
- `hatecheck_pt` - Portuguese
- `hatecheck_th` - Thai

### 4. Results Analysis

Results are saved in CSV format with the following structure:

```csv
text,label,prediction,logits_0,logits_1
"This is hate speech",1,1,-3.2,2.8
"This is normal content",0,0,2.1,-1.5
"Borderline case",1,0,0.3,-0.2
```

**Column explanation:**
- `text`: Input text
- `label`: True label (ground truth)
- `prediction`: Model prediction (0 or 1)
- `logits_0`: Raw score for class 0 (non-hateful) - higher = more confident
- `logits_1`: Raw score for class 1 (hateful) - higher = more confident

**Interpreting results:**
- When prediction == label: ✓ Correct prediction
- When prediction != label: ✗ Misclassification
- Logits close to 0: Model is uncertain
- Large logit difference: Model is confident

**Evaluation Metrics:**

The evaluation notebook calculates:
- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: Of predicted hate speech, how much is actually hateful
- **Recall**: Of actual hate speech, how much did we catch
- **F1 Score**: Harmonic mean of precision and recall (main metric)
- **Macro F1**: Average F1 across both classes (handles class imbalance)

**Example interpretation:**
```
F1 = 0.85, Precision = 0.80, Recall = 0.90
→ Model catches 90% of hate speech (recall)
→ But 20% of flagged content is false positive (precision)
→ Good for safety-critical applications (high recall)
```

## Supported Languages

| Language | Code | Datasets | HateCheck Support |
|----------|------|----------|-------------------|
| English | en | dyn21, fou18, ken20 | ✗ |
| Spanish | es | bas19 | ✓ |
| Portuguese | pt | for19 | ✓ |
| Italian | it | san20 | ✗ |
| French | fr | ous19 | ✓ |
| Arabic | ar | ous19 | ✓ |
| Hindi | hi | has19, has20, has21 | ✓ |
| Chinese | zh/cn | Custom collection | ✓ |
| Thai | th | HateThaiSent | ✓ |
| Dutch | nl | - | ✓ |
| German | de | - | ✓ |
| Polish | pl | - | ✓ |

## Configuration

### Creating Custom Configurations

Create a JSON configuration file in `2_finetuning/config/`:

```json
{
  "train_file": "0_data/main/1_clean/dyn21_en/train/train_20000_rs1.csv",
  "validation_file": "0_data/main/1_clean/dyn21_en/dev_500.csv",
  "test_file": "0_data/main/1_clean/dyn21_en/test_2000.csv",
  "model_name_or_path": "xlm-roberta-base",
  "output_dir": "./results/my_experiment",
  "do_train": true,
  "do_eval": true,
  "do_predict": true,
  "num_train_epochs": 3,
  "learning_rate": 5e-5,
  "per_device_train_batch_size": 16,
  "max_seq_length": 128,
  "use_class_weights": false,
  "store_prediction_logits": true
}
```

### Recommended Models

- **Multilingual**: `xlm-roberta-base`, `xlm-roberta-large`
- **English (Twitter)**: `cardiffnlp/twitter-roberta-base`
- **Multilingual (Twitter)**: `cardiffnlp/twitter-xlm-roberta-base`

## Results

Results are organized by dataset and training size:

```
3_evaluation/results/
├── bas19_es_test_2000/       # Spanish test results
├── for19_pt_test_2000/       # Portuguese test results
├── french_fr_test_2000/      # French test results
├── chinese_cn_test_2000/     # Chinese test results
├── thai_th_test_2000/        # Thai test results
└── hatecheck_*/              # HateCheck benchmark results
```

## Common Issues and Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:

```bash
# Reduce batch size and use gradient accumulation
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing True
```

### MPS (Apple Silicon) Support

The code automatically detects and uses MPS when available. To force CPU:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Path Issues

Always use absolute paths in configuration files or set the `DATA` environment variable:

```bash
export DATA=/full/path/to/efficient-low-resource-hate-detection
```

### Rate Limiting (Data Collection)

When collecting Chinese data from Reddit:
- The scraper automatically handles 429 (rate limit) errors
- Default wait time is 10 minutes (600 seconds)
- Adjust `--wait-on-429` parameter if needed
- Use `--sleep` to control base delay between requests (recommended: 3-4 seconds)

**Example with custom rate limit handling:**
```bash
python smart_chinese_collector.py \
  --subs China \
  --target 1000 \
  --sleep 4.0 \
  --wait-on-429 900  # Wait 15 minutes on rate limit
```

## Quick Reference

### New Data Collection Commands

```bash
# Chinese - Collect from Reddit
cd new-data-collection/chinese
python smart_chinese_collector.py \
  --subs China Sino ChineseLanguage sino_zh \
  --mode hot \
  --target 8000 \
  --chinese-min-chars 15 \
  --chinese-min-ratio 0.6 \
  --sleep 3.5 \
  --wait-on-429 600 \
  --out chinese_8000.jsonl
python convert_jsonl_to_csv.py
python create_dataset_splits.py

# French - Process HuggingFace dataset
cd new-data-collection/french
python script.py \
  --input ./fr_hf.csv \
  --out_dir ./datasets/french_hatesent \
  --dev 500 --test 2000 \
  --Ns 10 20 30 40 50 100 200 300 400 500 1000 2000 \
  --seeds 1 2 3 4 5 6 7 8 9 10

# Thai - Process HateThaiSent
cd new-data-collection/thai
python script.py \
  --input ./HateThaiSent.csv \
  --out_dir ./datasets/thai_hatesent \
  --dev 500 --test 2000 \
  --Ns 10 20 30 40 50 100 200 300 400 500 1000 2000 \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

### Training Quick Start

```bash
# Single model training
cd 2_finetuning
python finetune_and_test.py config_english.json

# Batch training workflow (recommended for comprehensive experiments)
cd 2_finetuning

# Step 1: Edit scripts to set your paths
nano launch_ft1.sh  # Update DATA and venv path

# Step 2: Make executable
chmod +x launch_*.sh

# Step 3: Run training pipeline in order
./launch_ft1.sh              # Train English models (Phase 1)
./launch_ft2.sh              # Train other languages (Phase 2)
./launch_test.sh             # Test all models (Phase 3)
./launch_test_0shot.sh       # Zero-shot evaluation (Phase 4)
./launch_test_hatecheck.sh   # HateCheck evaluation (Phase 5)

# For long-running jobs, use nohup:
nohup ./launch_ft1.sh > ft1.log 2>&1 &
tail -f ft1.log  # Monitor progress
```

### Evaluation Quick Start

```bash
# Open evaluation notebook
cd 3_evaluation
jupyter notebook evaluate_results.ipynb
```

## Frequently Asked Questions (FAQ)

### General Questions

**Q: Do I need labeled data to start?**  
A: Yes, you need labeled data (text + binary label) for training. For the main datasets, labels are already provided. For newly collected Chinese data, you'll need to annotate it manually or semi-automatically.

**Q: How much data do I need?**  
A: This depends on your goals:
- Minimum viable: ~100 samples (proof of concept)
- Good performance: ~1000 samples
- Strong performance: ~5000+ samples
- The project tests various sizes (10, 20, 50, 100, 200, 500, 1000, 2000+) to find the sweet spot.

**Q: Can I use this for languages not listed?**  
A: Yes! The multilingual models (XLM-RoBERTa) support 100+ languages. You'll need to:
1. Collect/prepare data for your language
2. Format it as `text,label` CSV
3. Use the same training pipeline

**Q: What's the difference between active learning and random sampling?**  
A: Active learning selects the most informative samples (where model is uncertain), while random sampling picks samples randomly. Active learning typically achieves better performance with less labeled data.

**Q: What is zero-shot cross-lingual evaluation?**  
A: Zero-shot means testing a model on a language it was never trained on. For example:
- Train model on English hate speech data
- Test directly on Spanish data WITHOUT any Spanish training
- This evaluates the multilingual model's ability to transfer knowledge across languages
- Compare results with models fine-tuned on Spanish to see the benefit of target-language training

### Technical Questions

**Q: Which model should I use?**  
A: Recommendations:
- Multilingual/cross-lingual: `xlm-roberta-base`
- English Twitter data: `cardiffnlp/twitter-roberta-base`
- Multilingual Twitter: `cardiffnlp/twitter-xlm-roberta-base`
- Need better performance with more resources: `xlm-roberta-large`

**Q: My training is very slow. What can I do?**  
A: Try these optimizations:
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \  # Maintains effective batch size
--gradient_checkpointing True \     # Saves memory
--fp16 True                         # Use mixed precision (if GPU supports it)
```

**Q: I'm getting CUDA out of memory errors. Help!**  
A: Reduce memory usage:
1. Decrease `--per_device_train_batch_size` to 1 or 2
2. Enable `--gradient_checkpointing True`
3. Reduce `--max_seq_length` from 128 to 64 or 96
4. Use a smaller model (base instead of large)
5. Enable `--fp16 True` for mixed precision training

**Q: How do I know if my model is overfitting?**  
A: Check the evaluation notebook. Signs of overfitting:
- High training accuracy but low validation/test accuracy
- Large gap between train and validation F1 scores
- Performance degrades on HateCheck benchmark

Solutions:
- Reduce training epochs
- Add more training data
- Use data augmentation
- Increase dropout (modify model config)

**Q: Can I train on multiple GPUs?**  
A: Yes! Use distributed training:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 \  # Number of GPUs
  finetune_and_test.py config.json
```

**Q: Where are the trained models saved?**  
A: Models are saved in the `--output_dir` specified in your config. By default:
- `./results/<experiment_name>/`
- Contains: model weights, tokenizer, training logs, predictions

### Data Questions

**Q: Can I combine multiple datasets?**  
A: Yes! Simply concatenate the CSV files:
```bash
cat dataset1.csv dataset2.csv > combined.csv
# Remove duplicate header
sed -i '2d' combined.csv  # Linux/Mac
```

**Q: My dataset is imbalanced (90% non-hate, 10% hate). What should I do?**  
A: Use class weighting:
```bash
--use_class_weights True
```
This automatically adjusts loss to account for class imbalance.

**Q: How do I validate my model on a new language?**  
A: Use HateCheck if available for your language, or:
1. Create a high-quality test set (500+ samples)
2. Include diverse examples (edge cases, borderline cases)
3. Calculate metrics with the evaluation notebook
4. Test on out-of-domain data if possible

**Q: What if my text is longer than 128 tokens?**  
A: You can increase max length, but be aware of memory:
```bash
--max_seq_length 256  # or 512
```
Or truncate text during preprocessing to fit within limits.

### Results Questions

**Q: What's a good F1 score?**  
A: It depends on context:
- F1 > 0.80: Excellent
- F1 0.70-0.80: Good
- F1 0.60-0.70: Moderate (low-resource acceptable)
- F1 < 0.60: Needs improvement

**Q: How do I interpret HateCheck results?**  
A: HateCheck tests specific functionalities:
- Check which test cases your model fails
- Look for systematic failures (e.g., fails on negations)
- Use this to guide model improvements

**Q: Can I deploy this model in production?**  
A: Consider these factors:
1. **Latency**: XLM-RoBERTa base ~50-100ms per prediction
2. **Ethics**: Hate speech detection has false positives/negatives
3. **Human in the loop**: Recommend human review for edge cases
4. **Bias**: Test thoroughly for demographic bias
5. **Updates**: Regularly retrain with new data

For deployment:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./results/my_model")
tokenizer = AutoTokenizer.from_pretrained("./results/my_model")

text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
```

## Citation

If you use this code, please cite the relevant papers and datasets used in your research.

### Original Work

This repository is based on the work by Paul Röttger et al. Please cite:

```bibtex
@inproceedings{rottger2022two,
    title = "Data-Efficient Strategies for Expanding Hate Speech Detection into Under-Resourced Languages",
    author = "R{\"o}ttger, Paul  and
      Vidgen, Bertie  and
      Nguyen, Dong  and
      Waseem, Zeerak  and
      Margetts, Helen  and
      Pierrehumbert, Janet",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.383",
    pages = "5673--5691"
}
```

Original repository: https://github.com/paul-rottger/efficient-low-resource-hate-detection

## Data Sources & Attribution

This project uses datasets from multiple sources. Please cite the original authors when using these datasets:

### Main Datasets

- **Dynabench 2021 (English)**: [Dynabench: Rethinking Benchmarking in NLP](https://arxiv.org/abs/2104.14337)
- **Founta 2018 (English)**: [Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior](https://arxiv.org/abs/1802.00393)
- **Kennedy 2020 (English)**: [Constructing interval variables via faceted Rasch measurement](https://doi.org/10.1145/3377325.3377506)
- **Basile 2019 (Spanish)**: [SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter](https://aclanthology.org/S19-2007/)
- **Fortuna 2019 (Portuguese)**: [A Hierarchically-Labeled Portuguese Hate Speech Dataset](https://aclanthology.org/W19-3510/)
- **Sanguinetti 2020 (Italian)**: [HaSpeeDe 2: Hate Speech Detection in Italian](http://ceur-ws.org/Vol-2765/paper160.pdf)
- **Ousidhoum 2019 (Arabic, French)**: [Multilingual and Multi-Aspect Hate Speech Analysis](https://aclanthology.org/D19-1474/)

### New Data Collections

- **French Hate Speech Superset**: [HuggingFace Dataset](https://huggingface.co/datasets/manueltonneau/french-hate-speech-superset)
  ```bibtex
  @dataset{tonneau2024french,
    author = {Manuel Tonneau},
    title = {French Hate Speech Superset},
    year = {2024},
    publisher = {HuggingFace},
    url = {https://huggingface.co/datasets/manueltonneau/french-hate-speech-superset}
  }
  ```

- **HateThaiSent (Thai)**: [GitHub Repository](https://github.com/dsmlr/HateThaiSent)
  ```bibtex
  @article{HateThaiSent,
    author = {Krishanu Maity and A.S. Poornash and Shaubhik Bhattacharya and Salisa Phosit and Sawarod Kongsamlit and Sriparna Saha and Kitsuchart Pasupa},
    title = {HateThaiSent: Sentiment-Aided Hate Speech Detection in Thai Language},
    journal = {IEEE Transactions on Computational Social Systems},
    year = {2024}
  }
  ```

### HateCheck Benchmarks

- **HateCheck**: [Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection](https://aclanthology.org/2021.acl-long.132/)
  - Multilingual versions available for: Arabic, Chinese, German, Spanish, French, Hindi, Italian, Dutch, Polish, Portuguese, Thai

## License

### Academic Use - Macquarie University

This repository was developed as part of **COMP8240 Applications of Data Science** at Macquarie University (2025).

### Code License
The code and scripts in this repository are licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 COMP8240 Project - Macquarie University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Dataset Licenses
Individual datasets retain their original licenses:
- **French Hate Speech Superset**: As specified on [HuggingFace](https://huggingface.co/datasets/manueltonneau/french-hate-speech-superset)
- **HateThaiSent**: MIT License - see [original repository](https://github.com/dsmlr/HateThaiSent)
- **Chinese collected data**: Subject to Reddit's terms of service and API usage policies
- **Other datasets**: See respective papers in [Data Sources & Attribution](#data-sources--attribution) section

### Attribution Requirements
When using this repository, please:
1. Cite the original work by Röttger et al. (see [Citation](#citation) section)
2. Acknowledge the specific datasets used (see [Data Sources & Attribution](#data-sources--attribution))
3. Credit this repository if you build upon it

### Disclaimer
This project is provided for educational and research purposes. The hate speech detection models may contain biases and should not be deployed in production without thorough testing and ethical review.

## Contact

For questions about this repository:
- **Maintainer**: [TimItsara](https://github.com/TimItsara)
- **Course**: COMP8240 Applications of Data Science, Macquarie University (2025)

For questions about the original methodology, contact: paul.rottger@oii.ox.ac.uk
