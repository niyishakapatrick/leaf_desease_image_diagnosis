import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt



header = st.container()
model_inference = st.container()
im = st.container()
features = st.container()





with header:
    # Add custom CSS styling
    st.markdown(
        """
        <style>
        .css-1aumxhk {
            background-color: skyblue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render the navbar
    st.markdown(
        """
        <div class="css-1aumxhk">
        <h1 style="color: white;">Leaf Desease Detection</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with model_inference:
    
    st.markdown(' #### Use a machine learning model to examine images and differentiate between Healthy and Diseased leaves.')
    
    preprocess = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the model's parameters, mapping them to the CPU if necessary
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'efficient_b2.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.efficientnet_b2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    def perform_inference(image):
        # Apply the transformations to the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Perform inference on the image
        with torch.no_grad():
            input_batch = input_batch.to(device)
            output = model(input_batch)
        
        # Get the predicted class
        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()
        
        # Define the class labels
        classes = ['desease', 'healthy']
        
        # Get the predicted label and probabilities
        predicted_class = classes[predicted_label]
        probabilities = torch.softmax(output, dim=1)
        prob_covid = probabilities[0][0].item()
        prob_normal = probabilities[0][1].item()
        
        return predicted_class, prob_covid, prob_normal

    def upload_image():
        # Upload and display the image
        uploaded_image = st.file_uploader("Upload an image (image must contain the leaf portion)", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            # Convert grayscale image to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform inference on the image
            predicted_class, prob_covid, prob_normal = perform_inference(image)

            # Get image dimensions
            width, height = image.size

            # Display the image and inference results
            st.image(image, caption="Uploaded Image", width=300)
            # Convert probabilities to percentages
            prob_covid_percent = prob_covid * 100
            prob_normal_percent = prob_normal * 100

            # Plot the probabilities
            labels = ['desease', 'healthy']
            probabilities = [prob_covid_percent, prob_normal_percent]
            colors = ['red', 'blue']

            fig, ax = plt.subplots()
            ax.barh(labels, probabilities, color=colors)
            ax.set_xlim(0, 100)  # Set x-axis limit from 0 to 100 (percentage range)
            ax.set_xlabel('Probability (%)')

            # Display the number values on the plot
            for i, v in enumerate(probabilities):
                ax.text(v + 1, i, str(round(v, 2)), color='black', va='center')

            # Display the image and the plot
            #st.image(image, caption="Uploaded Image", width=300)
            st.pyplot(fig)

    # Call the function to run the Streamlit app
    upload_image()


with im:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image('static/desease.png', caption='First column displays healthy leaves, while the second and third columns exhibit diseased leaves.', use_column_width=True)




with features:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Leaf desease detection")
    st.markdown("""Image-Based Automated Detection: With advancements in computer vision and image processing techniques, automated systems have been developed to detect leaf diseases. This approach involves capturing high-resolution images of plant leaves and analyzing them using algorithms to detect disease symptoms. Machine learning techniques, such as deep learning and convolutional neural networks (CNNs), are commonly employed for image classification and disease identification.

a. Leaf Segmentation: Initially, the leaf is segmented from the background using image processing techniques, isolating the region of interest for disease detection.

b. Feature Extraction: Relevant features are extracted from the segmented leaf images, such as texture, shape, color, or vein patterns. These features provide valuable information for disease classification.

c. Disease Classification: Machine learning models are trained on a labeled dataset containing images of healthy and diseased leaves. The models learn to classify the leaves into different disease categories based on the extracted features. Support Vector Machines (SVM), Random Forests, or deep learning models like Convolutional Neural Networks (CNNs) are commonly used for classification..""")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Salient features for leaf desease  detection")
    st.markdown("""Salient features for leaf disease detection from images include color-based features, texture-based features, shape-based features, and vein patterns. These features capture changes in color, texture, shape, and vein structure caused by diseases..""")
    



