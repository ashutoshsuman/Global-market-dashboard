import streamlit as st

st.set_page_config(page_title="NewsFlash App", layout="centered")

st.title(" Streamlit App is Working")

st.write("If you can see this, your deployment is successful.")

name = st.text_input("Enter your name:")

if name:

    st.success(f"Hello {name}, your Streamlit app is live ")

st.button("Click me")
