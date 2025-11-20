## **Rock Paper Scissors Classification API**

This is a simple **Flask-based API** for classifying images of **Rock, Paper, or Scissors** using a **Machine Learning** model built with **TensorFlow/Keras**. The application is packaged with **Docker** for easy deployment and execution.

---

### **Key Features**
- Predict whether an image is **Rock, Paper, or Scissors** using a classification model.  
- Built with **Flask** as the backend API.  
- Simple to deploy and run using **Docker**.  

---

### **Prerequisites**
Ensure you have the following installed:  
1. **Git**  
2. **Docker** (version 20 or higher)  

---

### **How to Run the Application**

#### 1. **Clone the Repository**
Clone this project to your local machine:
```bash
git clone https://github.com/Rzq12/Machine-Learning-Project.git
cd "Rock Paper Scissors Classification/Deployment"
```

#### 2. **Build the Docker Image**
Run the following command to build the Docker image:
```bash
docker build -t rps-classification-api .
```

#### 3. **Run the Docker Container**
Start the application with the command:
```bash
docker run -p 5000:5000 rps-classification-api
```

The application will be available at **http://localhost:5000**.

---

### **API Usage**

#### **1. Main Endpoint**
- **Endpoint**: `POST /predict`  
- **Description**: Accepts an image file and predicts whether it is **Rock**, **Paper**, or **Scissors**.

#### **2. How to Send a Request**

Use **Postman** or **cURL** to send a request with an image file:

**Example with cURL**:
```bash
curl -X POST -F "file=@path_to_image.jpg" http://localhost:5000/predict
```

- Replace `path_to_image.jpg` with the path to your image file.

#### **3. Example Response**
If the request is successful, you will receive a response in JSON format like this:

```json
{
  "class": "Rock",
  "confidence": 0.98
}
```

---


