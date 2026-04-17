# Plant Disease Prediction using AI

## Project Overview

This project is an AI-based image classification system developed to predict plant diseases from leaf images using a Convolutional Neural Network (CNN). It helps in early detection of diseases and supports better decision-making in agriculture.

## Features

* Detects plant diseases from leaf images
* Uses deep learning (CNN) for prediction
* Provides fast and automated results
* Simple and user-friendly interface

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* Kaggle Dataset

## Project Structure

```
├── dataset/                # Plant leaf images  
├── trained_model.keras     # Saved trained model  
├── train.py                # Model training script  
├── predict.py              # Prediction script  
├── app.py                  # Streamlit UI  
├── requirements.txt        # Required libraries  
└── README.md               # Project documentation  
```

## Installation and Setup

1. Clone the repository:

```
git clone https://github.com/your-username/plant-disease-prediction.git
cd plant-disease-prediction
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
streamlit run app.py
```

## Usage

Upload a plant leaf image through the interface. The system processes the image and displays the predicted disease as output.

## Model Details

The model is based on a Convolutional Neural Network (CNN) trained on plant leaf images. The input image size is 128x128 pixels, and the output is the predicted disease class.

## Future Improvements

* Integration with mobile applications
* Real-time detection using camera
* Expansion to include more plant species and diseases

## Contributing

Contributions are welcome. You can fork the repository and improve the project.

## License

This project is created for educational purposes.

## Acknowledgement

This project was developed independently as part of a self-paced AI learning internship.
"Plant Disease Prediction using AI"
