import streamlit as st
import pyotp

from db.auth_db import (
    verify_login,
    reset_password,
    get_user_details,
    create_user,
)


def login_form():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        totp = st.text_input("MFA Code")
        submit = st.form_submit_button("Login")
        if submit:
            if not username or not password:
                st.toast("Please enter both username and password.")

            if not totp:
                st.toast("Please enter the MFA code.")

            return verify_login(username, password, totp)

    return False, None


def reset_password_form():
    with st.expander("Forgot Password?"):
        with st.form("reset_password_form"):
            reset_username = st.text_input("Enter Username")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm Password", type="password")
            reset_totp = st.text_input("MFA OTP")
            reset_submit = st.form_submit_button("Reset Password")
            if reset_submit:
                if new_password != confirm_new_password:
                    st.toast("Passwords don't match", icon="🚨")
                else:
                    user_details = get_user_details(reset_username)
                    if not user_details:
                        st.toast(
                            "User not Found!\n Contact Admin for registration",
                            icon="‼️",
                        )
                    else:
                        totp = pyotp.TOTP(user_details["mfa_secret"])
                        if totp.verify(reset_totp, valid_window=1):
                            if reset_password(reset_username, new_password):
                                st.success("Password Updated Successfully")
                            else:
                                st.error("Failed to update the password")
                        else:
                            st.toast("Invalid MFA Code")


def user_register_form():
    with st.form("register_form"):
        st.subheader("👤 Register New User")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        new_role = st.selectbox("Role", ["admin", "user"])
        register_submit = st.form_submit_button("Register")

        if register_submit:
            if new_password != confirm_password:
                st.error("Passwords don't match")
                return False, None
            else:
                return create_user(new_username, new_password, new_role)

    return False, None
