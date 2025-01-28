# Brain Tumor MRI Image Dataset: Object Detection and Localization

## **Introduction**
This dataset contains MRI scans of the brain categorized into four classes of brain tumors: **Glioma**, **Meningioma**, **Pituitary**, and a "No Tumor" class for healthy scans. These MRI images are crucial for developing and testing machine learning models for tumor detection and classification. Automating this process can play a significant role in early diagnosis and treatment of brain tumors, improving patient outcomes.

---

## **Dataset Overview**
The **Brain Tumor MRI Image Dataset** includes:
- **Glioma Tumor**: Images containing glioma tumors, originating from glial cells in the brain.
- **Meningioma Tumor**: Images featuring meningioma tumors, forming in the meninges surrounding the brain.
- **Pituitary Tumor**: Images showing pituitary tumors located at the base of the brain.
- **No Tumor**: Healthy brain scans without any signs of tumors.

### Key Features:
- Improved image diversity and quality.
- Labeled MRI scans organized into directories for seamless usage in machine learning projects.
- Suitable for **classification**, **object detection**, and **localization** tasks in medical imaging research.

---

## **Use Case**
The dataset is ideal for:
- Training **Convolutional Neural Networks (CNNs)** for tumor classification.
- Implementing **object detection and localization** for identifying and outlining tumor regions.
- Research in medical image analysis and **AI-based diagnostic systems**.

---

## **Code Implementation**
### **1. Data Preparation**
We load the dataset, preprocess images, and split them into training and testing sets. Images are normalized to speed up model convergence.

```python
# Importing libraries and loading the dataset
import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
data_dir = '/path/to/brain-tumor-dataset'

# Preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0  # Normalize
    return img

# Loading images
yes_dir = os.path.join(data_dir, 'yes')
no_dir = os.path.join(data_dir, 'no')

yes_images = [os.path.join(yes_dir, img) for img in os.listdir(yes_dir)]
no_images = [os.path.join(no_dir, img) for img in os.listdir(no_dir)]

images = yes_images + no_images
labels = [1] * len(yes_images) + [0] * len(no_images)

# Shuffling and splitting data
data = list(zip(images, labels))
np.random.shuffle(data)
images, labels = zip(*data)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

train_images = np.array([preprocess_image(img) for img in train_images])
test_images = np.array([preprocess_image(img) for img in test_images])
train_labels = to_categorical(train_labels, 2)
test_labels = to_categorical(test_labels, 2)
```

---

### **2. Model Design**
We design a **multi-task CNN model** that performs both:
- **Classification**: Determines if a tumor is present.
- **Localization**: Identifies the tumor's bounding box in the image.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    
    classification_output = Dense(2, activation='softmax', name='classification')(x)
    localization_output = Dense(4, activation='linear', name='localization')(x)
    
    return Model(inputs=input_layer, outputs=[classification_output, localization_output])

model = create_model()
model.compile(optimizer='adam', 
              loss={'classification': 'categorical_crossentropy', 'localization': 'mse'},
              metrics={'classification': 'accuracy'})
```

---

### **3. Model Training**
We train the model using dummy bounding box values for simplicity. Real bounding boxes should be annotated for practical applications.

```python
# Generate dummy bounding boxes
train_bboxes = np.random.rand(len(train_images), 4)
test_bboxes = np.random.rand(len(test_images), 4)

# Train the model
history = model.fit(
    train_images,
    {'classification': train_labels, 'localization': train_bboxes},
    validation_data=(test_images, {'classification': test_labels, 'localization': test_bboxes}),
    epochs=10,
    batch_size=32
)
```

---

### **4. Evaluation**
Evaluate the model's classification accuracy and localization performance.

```python
results = model.evaluate(test_images, {'classification': test_labels, 'localization': test_bboxes})
print(f"Classification Accuracy: {results[3]}")
print(f"Localization Loss (MSE): {results[4]}")
```

---

### **5. Testing on a Single Image**
Visualize the model's predictions on an individual image, including the bounding box.

```python
def test_on_single_image(image_path, model):
    img = preprocess_image(image_path)
    class_pred, bbox_pred = model.predict(np.expand_dims(img, axis=0))
    
    class_idx = np.argmax(class_pred)
    bbox = bbox_pred[0]
    
    # Visualize predictions
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Class: {['No', 'Yes'][class_idx]}")
    plt.gca().add_patch(plt.Rectangle(
        (bbox[0] * IMG_WIDTH, bbox[1] * IMG_HEIGHT),
        (bbox[2] * IMG_WIDTH, bbox[3] * IMG_HEIGHT),
        fill=False, edgecolor='red', linewidth=2))
    plt.show()

# Test
test_image_path = '/path/to/sample_image.jpg'
test_on_single_image(test_image_path, model)
```

---

## **Next Steps**
- Use real bounding box annotations for better localization accuracy.
- Augment the dataset for improved generalization.
- Explore pre-trained models like **YOLO**, **Faster R-CNN**, or **EfficientDet** for advanced tumor detection.

---

## **Contribute**
If you'd like to improve this project or add features, feel free to fork the repository and submit a pull request.

---

## **License**
This project is distributed under the MIT License.

---

## 🧑🏻‍💻 About the Author  
**Name:** Arif Mia  

🎓 **Profession:** Machine Learning Engineer & Data Scientist  

---

### 🔭 **Career Objective**  
🚀 My goal is to contribute to groundbreaking advancements in artificial intelligence and data science, empowering companies and individuals with data-driven solutions. I strive to simplify complex challenges, craft innovative projects, and pave the way for a smarter and more connected future.  

🔍 As a **Machine Learning Engineer** and **Data Scientist**, I am passionate about using machine learning, deep learning, computer vision, and advanced analytics to solve real-world problems. My expertise lies in delivering impactful solutions by leveraging cutting-edge technologies.  

---

### 💻 **Skills**  
- 🤖 **Artificial Intelligence & Machine Learning**  
- 👁️‍🗨️ **Computer Vision & Predictive Analytics**  
- 🧠 **Deep Learning & Natural Language Processing (NLP)**  
- 🐍 **Python Programming & Automation**  
- 📊 **Data Visualization & Analysis**  
- 🚀 **End-to-End Model Development & Deployment**  

---

### 🚧 **Featured Projects**  

📊 **Lung Cancer Prediction with Deep Learning**  
Achieved 99% accuracy in a computer vision project using 12,000 medical images across three classes. This project involved data preprocessing, visualization, and model training to detect cancer effectively.  

🌾 **Ghana Crop Disease Detection Challenge**  
Developed a model using annotated images to identify crop diseases with bounding boxes, addressing real-world agricultural challenges and disease mitigation.  

🛡️ **Global Plastic Waste Analysis**  
Utilized GeoPandas, Matplotlib, and machine learning models like RandomForestClassifier and CatBoostClassifier to analyze trends in plastic waste management.  

🎵 **Twitter Emotion Classification**  
Performed exploratory data analysis and built a hybrid machine learning model to classify Twitter sentiments, leveraging text data preprocessing and visualization techniques.  

---

### ⚙️ **Technical Skills**  

- 💻 **Programming Languages:** Python 🐍, SQL 🗃️, R 📈  
- 📊 **Data Visualization Tools:** Matplotlib 📉, Seaborn 🌊, Tableau 📊, Power BI 📊  
- 🧠 **Machine Learning & Deep Learning:** Scikit-learn 🤖, TensorFlow 🔥, PyTorch 🧩  
- 🗂️ **Big Data Technologies:** Hadoop 🏗️, Spark ⚡  
- 🚀 **Model Deployment:** Flask 🌐, FastAPI ⚡, Docker 🐳  

---

### 🌐 **Connect with Me**  

📧 **Email:** arifmiahcse@gmail.com 

🔗 **LinkedIn:** [www.linkedin.com/in/arif-miah-8751bb217](#)  

🐱 **GitHub:** [https://github.com/Arif-miad](#)  

📝 **Kaggle:** [https://www.kaggle.com/arifmia](#)  

🚀 Let’s turn ideas into reality! If you’re looking for innovative solutions or need collaboration on exciting projects, feel free to reach out.  

---

