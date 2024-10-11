# Fake News Detector

Welcome to the Fake News Detector project! This repository contains a machine learning-based solution for detecting fake news articles. The aim of this project is to help identify misleading or false information by analyzing the text content of news articles. This README will provide a detailed overview of the project, including its purpose, features, technology stack, and how it all works.

## Project Overview

The Fake News Detector is a classification system that uses natural language processing (NLP) and machine learning techniques to classify news articles as either "fake" or "real." The model was trained on a dataset of labeled articles and uses various NLP features to determine the likelihood of an article being misleading. This project serves as a demonstration of how machine learning can be applied to real-world issues like misinformation.

## How It Works

1. **Data Collection and Preprocessing**:
   - The dataset used for training is loaded from a CSV file (`data/news.csv`). It contains labeled examples of real and fake news.
   - Preprocessing steps include tokenization, removing stopwords, and transforming text data into numerical representations using TF-IDF vectorization.

2. **Feature Extraction**:
   - The project utilizes `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to convert raw text into a feature matrix, capturing important features from the news articles.

3. **Model Training**:
   - The project uses a `PassiveAggressiveClassifier` as the primary model for detecting fake news. This model is well-suited for text classification tasks where the data is continuously evolving.
   - The dataset is split into training and testing sets using `train_test_split` for model evaluation.

4. **Evaluation**:
   - The model is evaluated using accuracy and confusion matrix metrics.
   - Results indicate the model's ability to distinguish between fake and real articles effectively.

5. **Deployment**:
   - The trained model is saved and can be used to classify new articles in real-time.
   - A command-line interface (CLI) is provided for users to input a news article and receive a prediction on its authenticity.

## Technology Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - **Scikit-Learn**: Used for training and evaluating machine learning models, including `TfidfVectorizer` and `PassiveAggressiveClassifier`.
  - **Pandas and NumPy**: Used for data manipulation and analysis.
- **IDE**: Jupyter Notebook or VS Code (recommended for viewing and running the code).

## Installation and Setup

1. Clone this repository:
   ```sh
   git clone https://github.com/rishi-desai/fake-news-detector.git
   ```
2. Navigate to the project directory:
   ```sh
   cd fake-news-detector
   ```
3. Install the necessary Python packages:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Code

### Prerequisites
- Make sure you have Python installed (preferably version 3.6 or above).

### Running the Code

1. **Navigate to the directory** where your script (`detector.py`) is saved:
   ```sh
   cd path/to/your/script/fake-news-detector
   ```

2. **To Train the Model**:
   Use the following command to train the model. Make sure the dataset (`news.csv`) is available in the `./data/` directory or any similar Fake News Classification dataset:
   ```sh
   python detector.py --train
   ```
   This command will train the model, calculate accuracy, and save both the model and vectorizer for future predictions.

3. **To Classify a News Article**:
   Use the following command, providing the text of the article to be classified:
   ```sh
   python detector.py --predict "Your news article text here"
   ```
   The script will load the saved model and vectorizer, classify the input article, and print whether it is "Fake" or "Real."

## Future Improvements

- **Web Application**: Develop a more user-friendly web interface for real-time classification.
- **Model Improvements**: Integrate more advanced NLP models like BERT to enhance prediction accuracy.
- **Data Sources**: Expand the dataset to include more recent articles from diverse news outlets.

## Acknowledgments

- Kaggle for providing the dataset used for training.
- Special thanks to the contributors of open-source NLP and machine learning libraries that made this project possible.

## Contact

If you have any questions, suggestions, or feedback, feel free to reach out via GitHub or email: [rishi.desai@example.com](mailto:rishiamishdesai@gmail.com).

---

Thank you for visiting this repository! If you found it interesting or useful, please give it a star ⭐.