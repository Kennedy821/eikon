import streamlit as st
import pandas as pd
import bcrypt
from PIL import Image
import jwt  # To generate and decode tokens
import time
import tempfile
from datetime import datetime
from uuid import uuid1
import requests

current_date = datetime.now()
formatted_date = current_date.strftime("%d-%m-%Y")
simple_date = str(current_date).split(' ')[0]

im = Image.open('slug_logo.png')
st.set_page_config(
    page_title="eikon",
    page_icon=im,
    initial_sidebar_state="collapsed",
    ) 

# define functions for registration form

def check_if_user_is_existing_user(user_email):
    # send the user email to the backend to check if the user already has an account
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["existing_user_check_endpoint"]}'
    payload = {"user_email": user_email}
    r = requests.post(base_api_address, json=payload, timeout=120)
    print(r)
    if r.ok:
        try:
            user_email_flag_value = r.json().get("verification_flag")
        except Exception as e:
            st.error(f"Failed to parse existing user check response: {e}")
            return None
        if user_email_flag_value == 1:
            return "existing_user"
        else:
            return "not_an_eikon_user"


def register_new_user( user_id,
    first_name,
    last_name,
    email_address,
    password,
    account_type,
    date):
    # send the user email to the backend to check if the user already has an account
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["add_new_user_endpoint"]}'
    payload = {"user_id": user_id,
    "first_name":first_name,
    "last_name":last_name,
    "email_address": email_address,
    "password":password,
    "account_type":account_type,
    "registration_date":date}

    r = requests.post(base_api_address, json=payload, timeout=120)
    print(r)
    if r.ok:
        try:
            user_registration_confirmation = r.json().get("updated_user_data")
        except Exception as e:
            st.error(f"Failed to parse registration response: {e}")
            return "error: failed to register your account."
        if user_registration_confirmation is True:
            return "success: registered your account."
        else:
            return "error: failed to register your account."

with st.container(border=1):
    # Login form
    st.title("Registration form")


    first_name = st.text_input("First name").strip()

    last_name = st.text_input("Last name").strip()


    email = st.text_input("Email")
    if email:
        if "@" in email:
            if ".com" in email or ".ac.uk" in email:
                pass
        else:
            st.markdown("it looks like your email isn't valid. Please try put in another email")
            

    confirm_email = st.text_input("Confirm Email")
    if email and confirm_email:
        if email != confirm_email:
            st.markdown("please make sure your emails match")


    password = st.text_input("Password", type="password")

    confirm_password = st.text_input("Confirm Password", type="password")

    if password and confirm_password:
        if password != confirm_password:
            st.markdown("please make sure your passwords match!")

    account_type = st.selectbox("select what you want to use the account for",["","student","commercial"])

    # Display the payment initiation UI
    st.write("Subscribe to access eikon API.")
    if st.button("Register"):
        with st.spinner("Registering your account..."):

            time.sleep(3)

            info_placeholder = st.empty()

            # # now check that the user has provided appropriate information to register to eikon
            # if first_name:
            #     if last_name:
            #         if email:
            #             if confirm_email:
            #                 if password:
            #                     if confirm_password:
            #                         if account_type!="":

            info_placeholder.info("form completed correctly")

            time.sleep(2)
            info_placeholder.empty()
            if check_if_user_is_existing_user(email)!="existing_user":

                # create a user_id
                reg_user_id = "eikon_user_"+str(uuid1())
                reg_attempt = register_new_user(
                                user_id=reg_user_id,
                                first_name=first_name,
                                last_name=last_name,
                                email_address=email,
                                password=password,
                                account_type=account_type,
                                date=formatted_date
                                )
                # st.write(reg_attempt)
                                
                # if reg_attempt == "existing_user":
                #     info_placeholder.error("It looks like this email address already has an account.")
                # else:
                info_placeholder.success(f"Successfully registered an account for {first_name}")
            else:
                info_placeholder.error("It looks like this email address already has an account.")
            
            
        st.stop()
