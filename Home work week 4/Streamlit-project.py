import streamlit as st
import cv2
import numpy as np
from PIL import Image
from hugchat import hugchat
from hugchat.login import Login

# Function to load vocabulary
def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words

# Function to calculate Levenshtein distance
def levenshtein_distance(token1, token2):
    distances = [[0]*(len(token2)+1) for i in range(len(token1)+1)]
    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1
    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                distances[t1][t2] = min(a, b, c) + 1
    return distances[len(token1)][len(token2)]

# Object Detection Functions
MODEL = r"D:\dev\AIO-189-phatpham\Home work week 4\MobileNetSSD_deploy.caffemodel"
PROTOTXT = r"Home work week 4/MobileNetSSD_deploy.prototxt.txt"

def process_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections

def annotate_image(image, detections, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), 70, 2)
    return image

# Main Function
def main():
    st.title("AIO_189 Phatpham Projects")

    # Word Correction Section
    st.header("Word Correction")
    word = st.text_input('Word:')
    if st.button("Compute"):
        vocabs = load_vocab(file_path=r'D:\dev\AIO-189-phatpham\Home work week 4\vocab.txt')
        leven_distances = {vocab: levenshtein_distance(word, vocab) for vocab in vocabs}
        sorted_distances = dict(sorted(leven_distances.items(), key=lambda item: item[1]))
        correct_word = list(sorted_distances.keys())[0]
        st.write('Correct word: ', correct_word)

    # Object Detection Section
    st.header("Object Detection")
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")
        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image)
        processed_image = annotate_image(image, detections)
        st.image(processed_image, caption="Processed Image")

    # Chatbot Section
    st.header("Chatbot")
    with st.sidebar:
        st.title('Login HugChat')
        hf_email = st.text_input('Enter E-mail:')
        hf_pass = st.text_input('Enter Password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your account!')
        else:
            st.success('Proceed to entering your prompt message!')

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def generate_response(prompt_input, email, passwd):
        sign = Login(email, passwd)
        cookies = sign.login()
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        return chatbot.chat(prompt_input)

    if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(prompt, hf_email, hf_pass)
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)

if __name__ == "__main__":
    main()


####### Pull the file out of Home work week 4 to run streamlit#########


