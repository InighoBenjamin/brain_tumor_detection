# Brain Tumor Detection & Analysis

An end-to-end medical imaging solution that leverages Deep Learning to segment and detect brain tumors from MRI scans. Developed with a Python-based AI model (2D Attention U-Net), a robust Spring Boot backend, and a MySQL database for storing patient and analysis records.

## 🚀 Key Features

* **Deep Learning Model:** Implements a 2D Attention U-Net architecture for high-accuracy brain tumor segmentation.
* **Backend API:** Java-based Spring Boot REST API for processing medical images, orchestrating inference from the model, and serving results.
* **Database Management:** MySQL schemas to efficiently manage patient metadata, scan histories, and prediction results.
* **Scalable Architecture:** Designed with distinct separations between the deep learning logic and the backend web application server.

## 📂 Project Structure

```text
brain_tumor_detection/
├── java_app/          # Java Spring Boot application (REST API & services)
├── data_base/         # MySQL schemas, configuration, and migration scripts 
├── docs/              # Project documentation and architectural charts
├── dataset.py         # PyTorch dataset loaders & data augmentation scripts
├── model.py           # 2D Attention U-Net neural network architecture definition
├── train.py           # Script for training the Deep Learning model
├── inference.py       # Script to run predictions (inference) on new MRI scans
└── requirements.txt   # Python dependencies
```

## 🛠️ Technology Stack

* **AI/Machine Learning:** Python, PyTorch / TensorFlow (Deep Learning), OpenCV
* **Model Architecture:** 2D Attention U-Net
* **Backend:** Java, Spring Boot, Maven/Gradle
* **Database:** MySQL
* **Tools:** RESTful APIs, Git

## ⚙️ Getting Started

### Prerequisites

* **Python 3.8+** (for Deep Learning scripts)
* **Java Development Kit (JDK) 17+** (for Spring Boot backend)
* **MySQL Server** (ensure the database is running locally or remotely)
* **Maven or Gradle** for managing Java dependencies

### 1. Database Setup
Navigate to the `data_base/` folder and execute the provided `.sql` schemas in your MySQL server to construct the necessary tables. Update the respective `application.properties` or `application.yml` file in `java_app/` with your database credentials.

### 2. Deep Learning Model (Python)
It is recommended to run the ML code using a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt

# To train the model from scratch (assuming dataset is configured)
python train.py

# To run a standalone test inference
python inference.py --image_path path/to/sample_mri.jpg
```

### 3. Spring Boot Backend (Java)
Open a terminal in the `java_app/` directory and use your build tool to compile and run the backend.

```bash
# Using Maven
mvn clean install
mvn spring-boot:run

# The API should start running on http://localhost:8080 (default)
```

## 🧠 Model Pipeline (2D Attention U-Net)

The AI engine utilizes an **Attention U-Net**, an evolution of the traditional U-Net model designed specifically for medical image segmentation. This model uses an attention gate structure to suppress irrelevant background regions in MRI scans while highlighting salient features (the tumor), yielding higher accuracy and fewer false positives in medical diagnostics.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License.
