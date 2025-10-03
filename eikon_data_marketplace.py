import os, json, time, requests
import streamlit as st
import stripe
import requests
import pandas as pd
import json
from PIL import Image
from datetime import datetime
from uuid import uuid1

# Functions

def get_api_key_from_credentials_basic(email):
    # send the user email to the backend to check if the user already has an account
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["basic_api_access_endpoint"]}'
    payload = {"email_address": email
              }
    r = requests.post(base_api_address, json=payload, timeout=120)
    if r.ok:
        api_key = r.json()["api_key"]
        return api_key

def get_api_key_from_credentials_secure(email, password):
    import requests
    # send the user email to the backend to check if the user already has an account
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["secure_api_access_endpoint"]}'
    payload = {"email_address": email,
              "password":password
              }
    r = requests.post(base_api_address, json=payload, timeout=120)
    if r.ok:
        api_key = r.json()["api_key"]
        return api_key

def check_users_current_credit_balance(api_key):
    # send the user email to the backend to check if the user already has an account
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["check_api_credits_endpoint"]}'
    payload = {"api_key": api_key}
    r = requests.post(base_api_address, json=payload, timeout=120)
    print(r)
    if r.ok:
        current_api_balance = r.json()["current_api_credit_balance"]
        return current_api_balance

def get_payments_link_to_add_credits():

    get_payments_link_endpoint = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["payments_link_endpoint"]}'
    
    payload = {"prompt": "_",
              }
    r = requests.post(get_payments_link_endpoint, json=payload, timeout=120)
    if r.ok:
        valid_payment_link_url = r.json()["payment_link"]
        return valid_payment_link_url

def update_users_api_credit_balance_based_on_latest_consumption_location_datasets(api_key,dataset_id,consumption_credit_value):
    import requests
    # this function takes in a string of data and calculates how many credits should be consumed from processing this data
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["consume_api_credits_for_location_dataset_purchase"]}'
    payload = {"api_key": api_key,
    "consumption_credit_value":consumption_credit_value,
    "dataset_id":dataset_id}
    r = requests.post(base_api_address, json=payload, timeout=120)
    print(r)
    if r.ok:
        update_response = r.json()["success"]
        if update_response=="Successfully updated user's credit balance.":
            return "success"  


def load_user_entitlements(api_key):

    get_user_entitlements_endpoint = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["user_entitlements_endpoint"]}'

    payload = {"api_key":api_key,
              }
    r = requests.post(get_user_entitlements_endpoint, json=payload, timeout=120)
    if r.ok:
        valid_user_entitlements_str = r.json()["user_entitlements"]
        return valid_user_entitlements_str
        
def load_entitled_datasets(single_dataset_id):

    get_entitled_dataset_endpoint = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["load_entitled_datasets_endpoint"]}'

    payload = {"dataset_id":single_dataset_id,
              }
    r = requests.post(get_entitled_dataset_endpoint, json=payload, timeout=120)
    if r.ok:
        valid_dataset_json = r.json()["requested_dataset"]
        return valid_dataset_json
        
def collect_user_entitlements_data(api_key):
    # first we're going to load their entitlements
    user_entitlements = load_user_entitlements(api_key)

    entitlements_list = user_entitlements.split("|")
    dataset_containers = []
    for dataset_id in entitlements_list:
        entitled_dataset_json_dict = load_entitled_datasets(dataset_id)
        dataset_containers.append(entitled_dataset_json_dict)
    return entitlements_list, dataset_containers

def convert_for_download(download_dataset_obj):
    df = pd.DataFrame.from_dict(json.loads(download_dataset_obj))
    return df.to_csv().encode("utf-8")


# ── "DB": replace with your catalog / DB as you like ─────────────────────────
PRODUCTS = [
    {
        "id": "low_resolution_visual_embeddings_res_7_united_kingdom",
        "title": "UK embeddings low res",
        "price_gbp": 1_00,
        "summary": "Location embeddings for UK at H3 resolution 7.",
        "filesize_mb": 520,
    },
    {
        "id": "london_buildings_v2",
        "title": "London Buildings (v2)",
        "price_gbp": 5_00,
        "summary": "Building footprints & attributes for Greater London.",
        "filesize_mb": 780,
    },
]

def get_product(pid): return next(p for p in PRODUCTS if p["id"] == pid)

# ── STATE ─────────────────────────────────────────────────────────────────────
if "cart" not in st.session_state: st.session_state.cart = {}  # {product_id: qty}

def add_to_cart(pid, qty=1):
    st.session_state.cart[pid] = st.session_state.cart.get(pid, 0) + qty

def remove_from_cart(pid):
    st.session_state.cart.pop(pid, None)



# ---------------------------------------------------------------------------------
# actual app
# -----------
im = Image.open('slug_logo.png')
st.set_page_config(page_title="EIKON Data Shop",
                   page_icon=im,
                   layout="wide")

# Inject CSS for banner
st.markdown(
    """
    <style>
    .banner {
        background-image: url("https://picsum.photos/id/122/1200/200/");
        background-size: cover;
        background-position: center;
        height: 200px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render the banner
st.markdown('<div class="banner"></div>', unsafe_allow_html=True)

st.title("EIKON Data Shop")

tab_reg, tab_catalog, tab_cart, tab_downloads = st.tabs(["Registration","Catalog", "Cart", "Downloads"])

# -REGISTRATION ----------------------------------------------------------------
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
with tab_reg:
    with st.container(border=1):
        # Login form
        st.title("Registration form")


        first_name = st.text_input("First name", key="first_n").strip()

        last_name = st.text_input("Last name", key="last_n").strip()


        email = st.text_input("Email", key="email")
        if email:
            if "@" in email:
                if ".com" in email or ".ac.uk" in email:
                    pass
            else:
                st.markdown("it looks like your email isn't valid. Please try put in another email")
                

        confirm_email = st.text_input("Confirm Email", key="email_confirm")
        if email and confirm_email:
            if email != confirm_email:
                st.markdown("please make sure your emails match")


        password = st.text_input("Password", type="password", key="init_pw")

        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pw")

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
# ── CATALOG ───────────────────────────────────────────────────────────────────
with tab_catalog:
    st.subheader("Browse datasets")
    for p in PRODUCTS:
        with st.container(border=1):

            col1, col2 = st.columns([5,1])
            with col1:
                st.markdown(f"**{p['title']}** — £{p['price_gbp']/100:.2f}")
                st.caption(p["summary"])
                st.text(f"Size: {p['filesize_mb']} MB")
            with col2:
                if st.button("Add to cart", key=f"add_{p['id']}"):
                    add_to_cart(p["id"])
                    st.success("Added!")

# ── CART + STRIPE CHECKOUT ───────────────────────────────────────────────────
with tab_cart:
    st.subheader("Your cart")
    if not st.session_state.cart:
        st.info("Cart is empty.")
    else:
        total = 0
        for pid, qty in st.session_state.cart.items():
            p = get_product(pid)
            with st.container(border=1):

                col1, col2 = st.columns([5,1])
                with col1: st.write(f"{p['title']} × {qty} — £{(p['price_gbp']*qty)/100:.2f}")
                with col2:
                    if st.button("Remove", key=f"rm_{pid}"):
                        remove_from_cart(pid)
                        st.rerun()
            total += p["price_gbp"] * qty
        
        actual_credit_value_of_basket = total / 100

        st.write("---")
        st.write(f"**Total: £{total/100:.2f}**")
        buyer_email = st.text_input("Your email (for receipts & downloads)")
        if buyer_email:
            users_api_key = get_api_key_from_credentials_basic(buyer_email)
            users_current_balance = check_users_current_credit_balance(users_api_key)
            st.info(f"Your current credits balance is: £{users_current_balance}")
            # st.markdown(total/100)
            if users_current_balance > actual_credit_value_of_basket:
                st.success("You have enough credits to make this purchase")

                with st.expander("View & Confirm Agreement"):
                    terms_and_conditions = """

                                            By checking this box and proceeding with your purchase, you agree to the following:
                                            **1.Payment and Credits**
                                            •   You authorise SlugAI to deduct the relevant number of credits from your account balance immediately upon confirming your purchase.
                                            •	You acknowledge that all transactions are final at the point of confirmation.
                                            **2.Access and Delivery**
                                            •	Upon completion of your purchase, you will be granted access to download the data associated with your order.
                                            •	Access will be available immediately, subject to technical availability.
                                            **3.No Refunds**
                                            •	Once you have accessed or downloaded the purchased data, you are not entitled to any refund, reversal, or credit reinstatement.
                                            •	SlugAI does not guarantee that the data will meet your specific requirements or be error-free, and no refunds will be provided on these grounds.
                                            **4.Use of Data**
                                            •	The data provided is for your personal or business use in accordance with the applicable licence terms.
                                            •	You may not redistribute, resell, or otherwise make the data available to third parties without prior written consent from SlugAI.
                                            **5.Liability**
                                            •	SlugAI provides the data “as is” and makes no warranties, express or implied, regarding its accuracy, completeness, or fitness for a particular purpose.
                                            •	To the maximum extent permitted by law, SlugAI will not be liable for any loss, damage, or costs arising from your use of the data.
                                            """
                    terms_state = False
                    st.write(terms_and_conditions)
                    if st.checkbox("I agree to the Terms and Conditions", value=terms_state):
                        terms_state = True

                        # if terms_state==True:

                if st.button("Buy", disabled=not buyer_email or not terms_state):
                    line_items = []
                    for pid, qty in st.session_state.cart.items():
                        p = get_product(pid)
                        line_items.append({
                            "quantity": qty,
                            "price_data": {
                                "currency": "gbp",
                                "unit_amount": p["price_gbp"],
                                "product_data": {
                                    "name": p["title"],
                                    "metadata": {"product_id": pid},
                                },
                            },
                        })
                        item_unit_price = p["price_gbp"]/100
                        item_dataset_id = pid

                        # first we're going to charge the account for the full total
                        updated_credit_balance_response = update_users_api_credit_balance_based_on_latest_consumption_location_datasets(api_key=users_api_key,
                                                                                                                                        consumption_credit_value=item_unit_price,
                                                                                                                                        dataset_id=item_dataset_id)
                        if updated_credit_balance_response=="success":

                            st.write(pid)
                            st.success(f"You can now access: {p['title']} in downloads")

                            # once we've done this we'll update the entitlements for this user



                        
            else:
                st.error("You need to add some credits to your account to buy this data")
                add_credits_payments_url = get_payments_link_to_add_credits()
                st.markdown(f"[Click here to add some credits to your account]({add_credits_payments_url})")
                # trigger the function to generate the payments link to top up the user's account


            # st.write("Redirecting to checkout…")
            # st.markdown(f"[Open Stripe Checkout]({session.url})")
            st.stop()

# Redirect flags (Stripe → success_url lands here)
qs = st.query_params
if qs.get("success") == "true" and qs.get("session_id"):
    st.success("Payment received! Your downloads will appear in the Downloads tab after webhook processing (usually a few seconds).")
elif qs.get("canceled") == "true":
    st.warning("Checkout canceled.")

# ── DOWNLOAD CENTER ──────────────────────────────────────────────────────────
with tab_downloads:
    st.subheader("Your downloads")
    email = st.text_input("Enter your email", key="dl_email")
    password = st.text_input("Password", type="password")
    if email and password:
        user_api_key = get_api_key_from_credentials_secure(email=email,password=password )

        entitlements_list , user_entitled_datasets = collect_user_entitlements_data(user_api_key)
        for idx in range(len(user_entitled_datasets)):
            with st.container(border=1):
                col1, col2 = st.columns([6,2])
                with col1:
                    st.markdown(f"**{entitlements_list[idx]}**")
                with col2:
                    dataset_n = convert_for_download(user_entitled_datasets[idx])
                    
                    # json.loads(dataset_n)
                    # st.markdown(dataset_n)
                    st.download_button(
                        label="Download CSV",
                        data=dataset_n,
                        file_name="data.csv",
                        mime="text/csv",
                        icon=":material/download:",
                    )
                        
                        
