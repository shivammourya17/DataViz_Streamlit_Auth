---
# Dataviz - AI Powered Data Visualization 🔐📊

A **secure, AI-powered data visualization and analytics platform** built with Streamlit.
It combines a **production-ready authentication system** (with MFA & RBAC) and an **intelligent data visualization engine** that lets users upload datasets, analyze, visualize, and interact with their data in real-time.

---

## ✨ Features

### 🔒 Authentication & Security

* **Secure Authentication** with password hashing (bcrypt)
* **Multi-Factor Authentication (MFA)** via TOTP (Google Authenticator, Authy, etc.)
* **Role-Based Access Control (RBAC):** Admin & user roles with different permissions
* **Session Management:** Secure sessions with configurable timeouts
* **Password Reset:** Self-service reset with MFA verification
* **Database Integration:** Postgres with proper connection pooling

### 📊 AI-Powered Data Visualization

* **CSV Uploads** (auto-detects delimiters & encoding)
* **Automatic Data Summary** (rows, columns, missing values, stats)
* **Visualization Generation:** Charts & graphs created dynamically with AI assistance
* **Statistical Analysis** for numeric & categorical data
* **Download Reports** (summaries + visualizations)
* **Interactive Chat Interface** to query and explore data
* **Save & Load Chat History** for later use
* **Clear Data/Chat Option** to start fresh

---

## 🚀 Quick Start

### 🔧 Prerequisites

* Python **3.8+**
* Postgres Database

---

### ⚡ Installation

1️⃣ Clone the repository:

```bash
git clone https://github.com/shivammourya17/DataViz_StreamlitMFA_Auth.git
cd DataViz_StreamlitMFA_Auth
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Create a `.env` file in the root directory:

```ini
DB_HOST=your_Postgres_host
DB_USER=your_Postgres_user
DB_PASSWORD=your_Postgres_password
DB_DATABASE=your_database_name
```

4️⃣ Initialize the database:
The system will automatically create required tables (`users`, `user_sessions`) on first run.

5️⃣ Run the app:

```bash
streamlit run dashboard_app.py
```

---

## 📋 Requirements

```
streamlit>=1.28.0
psycopg2-binary
bcrypt>=4.0.0
pyotp>=2.9.0
qrcode>=7.4.0
python-dotenv>=1.0.0
Pillow>=10.0.0
pandas>=2.0.0
matplotlib>=3.8.0
openai>=1.0.0   # if using GPT-powered insights
```

---

## 🏗️ System Architecture

### 🔐 Authentication Flow

* User login → MFA (TOTP) → Role-based access → Session created

### 📊 Data Flow

* User uploads CSV → AI generates summary → Visualization + Statistics → Report download

---

## 📁 Project Structure

```
Dataviz-StreamlitMFA-Auth/
├── dashboard_app.py        # Main application entry point
├── db/
│   ├── __init__.py
│   ├── auth_db.py          # Authentication database operations
│   └── setup_db.py         # Database initialization
├── ui/
│   ├── __init__.py
│   ├── auth_ui.py          # Authentication UI components
│   └── dashboard_ui.py     # Visualization dashboard UI
├── db_backups/
│   └── auth_db_setup.sql   # Database schema backup
├── .env                    # Environment variables (not committed)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔐 Environment Variables

| Variable     | Description         | Required |
| ------------ | ------------------- | -------- |
| DB\_HOST     | Postgres database host | ✅        |
| DB\_USER     | Postgres username      | ✅        |
| DB\_PASSWORD | Postgres password      | ✅        |
| DB\_DATABASE | Postgres database name | ✅        |

---

## 📊 Demo Use Case

1. Login securely with MFA
2. Upload a CSV dataset
3. Get instant **data summary + AI-generated insights**
4. Explore interactive charts & graphs
5. Download full **analysis report**
6. Save your chat-based exploration for later

---

## 🌟 Future Roadmap

* Support for **Excel & JSON uploads**
* Advanced **data cleaning & preprocessing** tools
* Integration with **cloud databases** (BigQuery, Snowflake)
* Customizable **user dashboards**

---

## 🛡️ License

MIT License © 2025 Shivam Mourya

---
