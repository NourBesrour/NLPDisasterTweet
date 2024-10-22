import gradio as gr
import joblib

# Load the trained models
nb_pipeline = joblib.load('nb_pipeline.pkl')
lsvm_pipeline = joblib.load('LSVM.pkl')
logreg_pipeline = joblib.load('logreg.pkl')

# Define a function to classify the input text using the selected model
def classify_text(model_choice, input_text):
    if model_choice == "Naive Bayes":
        prediction = nb_pipeline.predict([input_text])[0]
    elif model_choice == "Linear SVM":
        prediction = lsvm_pipeline.predict([input_text])[0]
    elif model_choice == "Logistic Regression":
        prediction = logreg_pipeline.predict([input_text])[0]
    
    # Return a human-readable result
    return "it's not a disaster" if prediction == 1 else "it's a disaster"

# Set up the Gradio interface
model_choices = ["Naive Bayes", "Linear SVM", "Logistic Regression"]
input_text = gr.Textbox(lines=2, placeholder="Enter text to classify...")
output_text = gr.Textbox()

# Create the Gradio interface
gr.Interface(
    fn=classify_text,
    inputs=[
        gr.Dropdown(model_choices, label="Select Model"),
        input_text
    ],
    outputs=output_text,
    title="Text Classification",
    description="Enter text and select a classification algorithm to determine if the sentiment is positive or negative.",
).launch(share=True)
