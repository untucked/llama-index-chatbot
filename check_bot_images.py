import streamlit as st
from htmlTemplates import css, bot_template, user_template, user_icon_base64, bot_icon_base64
# streamlit run check_bot_images.py
st.markdown(css, unsafe_allow_html=True)

st.markdown(user_template.format(user_icon=user_icon_base64, message="Hello from user!"), unsafe_allow_html=True)
st.markdown(bot_template.format(bot_icon=bot_icon_base64, message="Hello from bot!"), unsafe_allow_html=True)
