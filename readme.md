# MNIST Digit Classifier

This project is a simple web-based MNIST digit classifier using a **Flask** backend and a **PyTorch** deep learning model. The model predicts handwritten digits from 0 to 9 based on user-uploaded images.

## Features
- A **Flask API** that processes uploaded images and returns a digit prediction.
- A **fully connected neural network** built with PyTorch for digit classification.
- A **simple web interface** (HTML, CSS, JavaScript) for users to upload images and see predictions.
- **CORS support** to allow communication between frontend and backend.

---
## Project Structure
```
IAS_LAB6_2023201033/
│── app.py           # Flask server for handling API requests
│── model.py         # PyTorch model definition and training script
│── mnist_model.pth  # Saved trained model weights
│── static/
│   ├── styles.css   # Frontend CSS styling
│── templates/
│   ├── index.html   # Frontend HTML page
│── README.md        # Project documentation
```

---
## Requirements
To run this project, you need to have the following installed:
- Python 3.x
- Flask
- Flask-CORS
- PyTorch
- Torchvision
- PIL (Pillow)

Install dependencies using:
```sh
pip install flask flask-cors torch torchvision pillow
```

---
## Model Training
To train the model from scratch, run:
```sh
python model.py
```
This script will:
- Load the MNIST dataset.
- Train a fully connected neural network for 5 epochs.
- Save the trained model as `mnist_model.pth`.
- **I have already added the saved model. So need not run this code**

---
## Running the Flask Server
To start the API, run:
```sh
python app.py
```
The API will be available at `http://127.0.0.1:5000/predict`.

---
## Using the Web Interface
1. Open `index.html` in a browser.
2. Upload an image of a handwritten digit.
3. Click the **Predict** button.
4. The predicted digit will be displayed on the page.

---
## API Usage
### Endpoint: `/predict`
- **Method:** `POST`
- **Request:**
  - Upload an image file (PNG, JPG, etc.)
- **Response:**
  ```json
  {
    "prediction": 3
  }
  ```

---


