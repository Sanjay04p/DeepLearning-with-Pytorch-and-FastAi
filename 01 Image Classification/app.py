import gradio as gr
from fastai.vision.all import *
model = load_learner('image_classifier.pkl')
categories=('Cat','Dog')
def classify(img):
    pred,_,prob=model.predict(img)
    return dict(zip(categories,map(float,prob)))

image = gr.Image(height=192, width=192)
label = gr.Label()
classify = gr.Interface(fn=classify, inputs=image, outputs=label)
classify.launch(inline=False)