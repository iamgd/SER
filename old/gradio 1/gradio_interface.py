import gradio as gr
from ensemble_predict import ensemble_predict

def predict_emotion(audio_file):
    return ensemble_predict(audio_file)

interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Emotion Recognition",
    description="Predict emotion from audio using ensemble of SVM, CNN, and LSTM."
)

if __name__ == "__main__":
    interface.launch()
