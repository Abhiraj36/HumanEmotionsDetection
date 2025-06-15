Project Overview: 

This project implements a highly accurate Emotion Detection System using state-of-the-art deep learning techniques. By leveraging the powerful Vision Transformer (ViT) architecture, the model classifies human facial emotions into categories such as Happy, Sad, and Angry. The system is capable of analyzing image inputs and making real-time predictions, with extensions for downstream applications like emotion-based music recommendation.

Objectives:

Build a high-performing image classification model for human emotion recognition.

Compare traditional CNN architectures (LeNet) with Transformer-based models (ViT).

Implement robust data preprocessing and augmentation pipelines.

Evaluate model performance using top-k accuracy, validation metrics, and confusion matrix.

Showcase real-world applicability through a minor deployment demo.

Achievements:

 Achieved ~85% validation accuracy using ViT on facial emotion dataset.

 Achieved top-2 accuracy of ~89%, demonstrating strong generalization.

 Trained on over 5,000+ labeled images with augmentation and regularization.

 Integrated WandB logging for real-time training insights, GPU utilization, and performance analytics.

Built a prototype for emotion-based music recommendation using model outputs.



Technologies Used:

Category	Stack

Programming Language	Python

Deep Learning Framework	TensorFlow, Keras

Vision Models	Vision Transformer (ViT), LeNet

Data Handling & Augmentation	NumPy, Pandas, OpenCV, tf.data

Monitoring & Logging	Weights & Biases (wandb)

Visualization	Matplotlib, Seaborn

Deployment Prototype	Streamlit (Webcam-based music recommendation demo)



Results:

Confusion Matrix clearly shows strong class-wise prediction performance.

Loss & Accuracy Graphs indicate stable training with effective regularization.

Top-K Accuracy consistently remained above 89% for most validation batches.

GPU Monitoring Logs helped fine-tune performance and reduce overfitting.


Real-World Use Cases:

Real-time Emotion Tracking for Mental Health Monitoring

Smart Emotion-Based Content Recommendation

Classroom or Workplace Engagement Analysis

Interactive Customer Support Systems



Contributions:

This project was individually developed and executed by Abhiraj Aryan, including:

Model Architecture (ViT, LeNet)

Data Handling & Augmentation

Training & Evaluation

Deployment Prototype

Visualizations and Documentation


Final Words:

This project demonstrates how cutting-edge machine learning techniques, when combined with real-world application ideas, can lead to powerful and practical solutions. The model's high accuracy, robust design, and extensible use cases position it as a strong foundation for advanced emotion recognition systems.

