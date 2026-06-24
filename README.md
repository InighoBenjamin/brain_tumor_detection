# 🧠 Brain Tumor Detection & Analysis System

An AI-powered medical imaging platform designed to detect and segment brain tumors from MRI scans using Deep Learning techniques. The system combines a **2D Attention U-Net model**, a **Spring Boot REST API**, and a **MySQL database** to provide accurate tumor analysis, patient record management, and scalable deployment.

---

## 🚀 Features

### 🔍 AI-Based Tumor Detection

* Utilizes a **2D Attention U-Net** architecture for precise brain tumor segmentation.
* Processes MRI scans and highlights tumor regions with high accuracy.
* Reduces false positives through attention-based feature extraction.

### ⚙️ Backend Services

* Developed using **Java Spring Boot**.
* RESTful APIs for image upload, prediction requests, and result retrieval.
* Seamless integration with the Deep Learning inference engine.

### 🗄️ Database Management

* MySQL database for storing:

  * Patient information
  * MRI scan records
  * Prediction history
  * Analysis reports

### 📈 Scalable Architecture

* Modular design separating:

  * AI Model Layer
  * Backend Service Layer
  * Database Layer
* Supports future enhancements and cloud deployment.

---

## 📂 Project Structure

```text
brain_tumor_detection/
├── java_app/          # Spring Boot backend application
├── data_base/         # MySQL scripts and database configuration
├── docs/              # Documentation and architecture diagrams
├── dataset.py         # Dataset loading and preprocessing
├── model.py           # Attention U-Net architecture
├── train.py           # Model training script
├── inference.py       # MRI prediction and segmentation
└── requirements.txt   # Python dependencies
```

---

## 🛠️ Technology Stack

### Artificial Intelligence & Machine Learning

* Python
* Deep Learning
* PyTorch / TensorFlow
* OpenCV
* NumPy

### Backend Development

* Java
* Spring Boot
* Maven / Gradle
* REST APIs

### Database

* MySQL

### Version Control & Tools

* Git
* GitHub

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.8+
* Java JDK 17+
* MySQL Server
* Maven or Gradle
* Git

---

### Step 1: Configure Database

Import the SQL scripts located in the `data_base/` directory into MySQL.

Update database credentials in:

```properties
application.properties
```

or

```yaml
application.yml
```

inside the `java_app/` folder.

---

### Step 2: Setup Deep Learning Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Train Model

```bash
python train.py
```

#### Run Prediction

```bash
python inference.py --image_path path/to/mri_image.jpg
```

---

### Step 3: Run Spring Boot Application

```bash
cd java_app

mvn clean install
mvn spring-boot:run
```

Backend API will be available at:

```text
http://localhost:8080
```

---

## 🧠 Deep Learning Model

The system employs an **Attention U-Net**, an advanced variation of the U-Net architecture specifically designed for medical image segmentation.

### Benefits

* Focuses on relevant tumor regions.
* Suppresses irrelevant background information.
* Improves segmentation accuracy.
* Produces detailed tumor boundary predictions.

### Workflow

```text
MRI Scan
   ↓
Image Preprocessing
   ↓
Attention U-Net Model
   ↓
Tumor Segmentation
   ↓
Prediction Generation
   ↓
Storage in MySQL Database
   ↓
Result Visualization
```

---

## 📊 Applications

* Brain Tumor Detection
* Medical Image Analysis
* Clinical Decision Support
* Healthcare Research
* Diagnostic Assistance Systems

---

## 🔮 Future Enhancements

* Multi-class tumor classification
* Real-time prediction dashboard
* Cloud deployment support
* Patient report generation
* Integration with hospital management systems

---

## 👥 Contributors

Developed as a collaborative academic project focused on applying Artificial Intelligence and Medical Imaging techniques for healthcare solutions.

---

## 📄 License

This project is licensed under the MIT License.
