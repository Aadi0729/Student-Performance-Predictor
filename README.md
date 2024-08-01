# Student Performance Predictor

Welcome to the **Student Performance Predictor** project! This repository contains the source code, documentation, and other relevant materials for the Student Performance Predictor, a machine learning project designed to analyze and predict student performance.

## Table of Contents
* [Project Overview](#project-overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Model Training](#model-training)
* [Prediction Pipeline](#prediction-pipeline)
* [User Interface](#user-interface)
* [Deployment](#deployment)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Project Overview
The **Student Performance Predictor** is an innovative machine learning application that predicts student performance based on various features such as study time, class participation, and more. This tool aims to assist educators in making data-driven decisions to improve educational outcomes.

## Features
* **Data Ingestion & Transformation:** Efficient handling and processing of raw data.
* **Model Training:** Implementation of robust machine learning models.
* **Prediction Pipeline:** Streamlined pipeline for making real-time predictions.
* **User Interface:** Intuitive and aesthetically pleasing web interface built with Flask.
* **Deployment:** Live deployment on Render for easy access and interaction.

## Installation
To get a local copy up and running, follow these simple steps:

### Prerequisites
* Python 3.7 or higher
* Flask
* Pandas
* Scikit-learn

### Installation Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/student-performance-predictor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd student-performance-predictor
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the application locally, use the following command:
```bash
python app.py
````

Visit http://127.0.0.1:5000/ in your web browser to access the web interface.

## Project Structure
````bash
project_folder/
│
├── app.py
├── requirements.txt
├── README.md
│
├── static/
│   └── css/
│       └── styles.css
│
├── templates/
│   ├── index.html
│   └── home.html
│
└── src/
    ├── exception.py
    ├── logger.py
    ├── pipeline/
    │   ├── predict_pipeline.py
    │   └── __init__.py

````

## Model Training
The model training process involves several steps, including data preprocessing, feature engineering, model selection, and evaluation. The `src/pipeline/predict_pipeline.py` file contains the implementation of the prediction pipeline.

## Prediction Pipeline
The prediction pipeline is designed to take input data, preprocess it, and generate predictions using the trained model. It ensures that the entire process is efficient and scalable.

## User Interface
The web interface is built using Flask and provides a user-friendly experience. The main features include:
* [Home Page](#homepage)
* [Prediction Page](#predictionpage)

## Deployment
The application is deployed on Render, making it accessible to a wider audience. You can explore the live project here: 🌟 https://edupredict.onrender.com/

## ⭐️ Show Your Support

If you find this project helpful or interesting, please give it a star! ⭐️

Your support is greatly appreciated and helps to keep the project going.

## Contact
For any inquiries or feedback, please reach out via LinkedIn - https://www.linkedin.com/in/aditya-pateriya7781/ or email - adityapateriya7986@gmail.com.
