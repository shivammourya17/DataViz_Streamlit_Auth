import streamlit as st
import pyotp, qrcode
import time
from io import BytesIO
from db.auth_db import get_user_details, SessionManagement
from ui.auth_ui import login_form, reset_password_form, user_register_form
from ui.dashboard_ui import show_dashboard

st.set_page_config(
        page_title="AI-Powered Data Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# Session timeout (30 minutes)
SESSION_TIMEOUT = 1800
sm = SessionManagement()

# Initialize session management
if "session_initialized" not in st.session_state:
    # Try to recover session from query params
    session_id = st.query_params.get("session_id", None)

    if session_id:
        session_data = sm.get_session(session_id)
        if session_data and (
            time.time() - session_data["created_at"] < SESSION_TIMEOUT
        ):
            # Valid session found
            st.session_state.session_id = session_id
            st.session_state.username = session_data["username"]
            st.session_state.role = session_data["role"]
            st.session_state.authenticated = True
            st.session_state.last_activity = time.time()
        else:
            # Invalid or expired session
            sm.delete_session(session_id)
            st.query_params.clear()

    st.session_state.session_initialized = True

# Check session timeout
if st.session_state.get("authenticated", False):
    current_time = time.time()
    elapsed_time = current_time - st.session_state.get("last_activity", current_time)

    if elapsed_time > SESSION_TIMEOUT:
        sm.logout(st.session_state.session_id)
        st.session_state.clear()
        st.warning("Your session has expired. Please log in again.")
        st.query_params.clear()
    else:
        # Update last activity
        st.session_state.last_activity = current_time
        sm.update_session(st.session_state.session_id)



# Logout button
if st.sidebar.button("Logout", type="primary"):
    if st.session_state.get("session_id"):
        sm.logout(st.session_state.session_id)
    st.session_state.clear()
    st.query_params.clear()
    st.rerun()

# ---------------- LOGIN + SIGNUP FLOW ---------------- #
if not st.session_state.get("authenticated", False):

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    # ---------- LOGIN ---------- #
    with tab1:
        success, response = login_form()
        if success:
            session_id = sm.create_session(response["username"], response["role"])
            st.session_state.session_id = session_id
            st.session_state.username = response["username"]
            st.session_state.role = response["role"]
            st.session_state.authenticated = True
            st.session_state.last_activity = time.time()
            st.query_params["session_id"] = session_id
            st.rerun()
        else:
            if response:
                st.error(response)
            reset_password_form()

    # ---------- SIGNUP ---------- #
    with tab2:
        is_success, user = user_register_form()
        if is_success and user.get("mfa_secret"):
            st.session_state.new_mfa_secret = user["mfa_secret"]
            st.session_state.new_username = user["username"]
            st.session_state.show_qr = True

        if st.session_state.get("show_qr", False):
            st.subheader("🔐 MFA Setup Required")
            st.info("Scan this QR code with your authenticator app")

            # Generate QR code
            otp_uri = pyotp.totp.TOTP(st.session_state.new_mfa_secret).provisioning_uri(
                name=st.session_state.new_username, issuer_name="streamlit auth"
            )
            img = qrcode.make(otp_uri)
            buff = BytesIO()
            img.save(buff, format="PNG")
            img_bytes = buff.getvalue()

            st.image(img_bytes, caption="Scan the QR with Authenticator App", width=200)
            st.text(f"Manual Entry Secret: {st.session_state.new_mfa_secret}")

            if st.button("Done"):
                del st.session_state["show_qr"]
                st.success("User created successfully! Please log in.")

# ---------------- MAIN DASHBOARD ---------------- #
if st.session_state.get("authenticated", False):
    user_profile = get_user_details(st.session_state.username)
    if user_profile:
        st.sidebar.subheader(f"Welcome, {user_profile['username']}")
        st.sidebar.write(f"Role: {user_profile['role']}")

    # Show dashboard directly (no admin-only registration)
    show_dashboard()
