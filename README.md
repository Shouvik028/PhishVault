# PhishVault

🚨 **Combat Phishing with AI**

PhishVault is an intelligent, machine learning-powered solution to detect and block phishing attacks. As cyber threats become more sophisticated, PhishVault helps protect your sensitive information from malicious websites and emails.

---

## ⚡ Key Features

- 🔍 **Comprehensive Analysis:** Scans website URLs, email text, embedded links, attachments, and sender information for suspicious indicators.
- 🤖 **Cutting-Edge ML Algorithms:** Utilizes advanced models including Support Vector Machine (SVM), Decision Tree, Random Forest, Gaussian Naive Bayes, AdaBoost, and more.
- 🚀 **Fast & Accurate:** Delivers rapid, reliable detection with resource-efficient processing.
- 🛡️ **Scalable Protection:** Suitable for individuals and organizations, adaptable to a range of security needs.

---

## 📂 Inputs

- **CSV files of phishing and legitimate URLs:**  
  - `verified_online.csv` &rarr; phishing website URLs from [phishtank.org](https://phishtank.org)  
  - `tranco_list.csv` &rarr; legitimate website URLs from [tranco-list.eu](https://tranco-list.eu)

---

## 🛠️ General Workflow

1. Use CSV files to load URLs.
2. Send a request to each URL using Python’s `requests` library.
3. Parse the response content with BeautifulSoup.
4. Extract features and create a numerical vector for each website.
5. Repeat extraction for all URLs and build a structured DataFrame.
6. Add a label: `1` for phishing, `0` for legitimate.
7. Save as CSV: see `structured_data_legitimate.csv` and `structured_data_phishing.csv`.
8. Combine and split data for training/testing (see `machine_learning.py` for examples).
9. Implemented ML models:
   - Support Vector Machine
   - Gaussian Naive Bayes
   - Decision Tree
   - Random Forest
   - AdaBoost
10. Evaluate with confusion matrix and performance metrics (accuracy, precision, recall).
11. Visualize all results—Naive Bayes performed best in testing.

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shouvik028/PhishVault.git
   cd PhishVault
