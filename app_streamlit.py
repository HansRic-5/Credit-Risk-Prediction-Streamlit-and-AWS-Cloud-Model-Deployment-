import streamlit as st
import pandas as pd
import json
import os
import boto3
from botocore.exceptions import ClientError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "UAS-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

st.set_page_config(page_title="Credit Score Predictor", layout="wide")

st.sidebar.title("Informasi Aplikasi")
st.sidebar.info("Aplikasi prediksi skor kredit berbasis Cloud (AWS SageMaker).")
st.sidebar.success("Terhubung dengan Endpoint")

st.title("Credit Score Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(["Demografi & Pemasukan", "Informasi Kartu & Rekening", "Portofolio Pinjaman"])

with tab1:
    st.header("Profil & Keuangan Bulanan")
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Bulan", ["January", "February", "March", "April", "May", "June", "July", "August"])
        age = st.number_input("Usia", min_value=18, max_value=100, value=25)
        occupation = st.selectbox("Pekerjaan", [
            'Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
            'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
            'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect', 'Unknown'
        ])
    with col2:
        annual_income = st.number_input("Pendapatan Tahunan", min_value=0.0, value=50000.0, step=1000.0)
        monthly_inhand_salary = st.number_input("Gaji Bersih Bulanan", min_value=0.0, value=4000.0, step=100.0)
        monthly_balance = st.number_input("Saldo Akhir Bulanan", min_value=0.0, value=500.0, step=50.0)

with tab2:
    st.header("Rekam Jejak Bank, Kartu & Tagihan")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, max_value=20, value=2)
        num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=20, value=1)
        interest_rate = st.number_input("Suku Bunga Rata-rata (%)", min_value=0, max_value=100, value=10)
        num_of_loan = st.number_input("Total Jumlah Pinjaman", min_value=0, max_value=20, value=1)
        delay_from_due_date = st.number_input("Rata-rata Telat Bayar (Hari)", min_value=0, max_value=365, value=5)
        num_of_delayed_payment = st.number_input("Banyaknya Pembayaran Telat", min_value=0, max_value=100, value=1)
        
    with c2:
        changed_credit_limit = st.number_input("Perubahan Limit Kredit (%)", min_value=-100.0, max_value=100.0, value=1.0)
        num_credit_inquiries = st.number_input("Banyaknya Inquiry Kredit", min_value=0, max_value=50, value=0)
        credit_mix = st.selectbox("Mix Kredit", ["Good", "Standard", "Bad", "Unknown"])
        outstanding_debt = st.number_input("Hutang Belum Dibayar ($)", min_value=0.0, value=1000.0)
        credit_utilization_ratio = st.slider("Rasio Penggunaan Kredit (%)", min_value=0.0, max_value=100.0, value=30.0)
        credit_history_age = st.number_input("Lama Riwayat Kredit (Dalam Bulan)", min_value=0.0, value=24.0)
        
    with c3:
        payment_of_min_amount = st.radio("Bayar Nominal Minimum?", ["Yes", "No", "NM"])
        total_emi_per_month = st.number_input("Total EMI Bulanan", min_value=0.0, value=150.0)
        amount_invested_monthly = st.number_input("Investasi Bulanan ($)", min_value=0.0, value=100.0)
        spent_level = st.selectbox("Level Pengeluaran", ["Low", "High"])
        payment_value = st.selectbox("Nilai Pembayaran", ["Small", "Medium", "Large"])

with tab3:
    st.header("Jumlah Frekuensi Setiap Pinjaman")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        auto_loan_freq = st.number_input("Auto Loan", min_value=0, max_value=10, value=0)
        credit_builder_loan_freq = st.number_input("Credit-Builder Loan", min_value=0, max_value=10, value=0)
    with p2:
        debt_consolidation_loan_freq = st.number_input("Debt Consolidation Loan", min_value=0, max_value=10, value=0)
        home_equity_loan_freq = st.number_input("Home Equity Loan", min_value=0, max_value=10, value=0)
    with p3:
        mortgage_loan_freq = st.number_input("Mortgage Loan", min_value=0, max_value=10, value=0)
        payday_loan_freq = st.number_input("Payday Loan", min_value=0, max_value=10, value=0)
    with p4:
        personal_loan_freq = st.number_input("Personal Loan", min_value=0, max_value=10, value=0)
        student_loan_freq = st.number_input("Student Loan", min_value=0, max_value=10, value=0)

st.markdown("---")
submit_btn = st.button("Prediksi Skor Kredit via SageMaker", type="primary", use_container_width=True)

if submit_btn:
    input_dict = {
        'Month': month,
        'Age': float(age),
        'Occupation': occupation,
        'Annual_Income': float(annual_income),
        'Monthly_Inhand_Salary': float(monthly_inhand_salary),
        'Num_Bank_Accounts': int(num_bank_accounts),
        'Num_Credit_Card': int(num_credit_card),
        'Interest_Rate': int(interest_rate),
        'Num_of_Loan': int(num_of_loan),
        'Delay_from_due_date': int(delay_from_due_date),
        'Num_of_Delayed_Payment': int(num_of_delayed_payment),
        'Changed_Credit_Limit': float(changed_credit_limit),
        'Num_Credit_Inquiries': int(num_credit_inquiries),
        'Credit_Mix': credit_mix if credit_mix != "Unknown" else float('nan'),
        'Outstanding_Debt': float(outstanding_debt),
        'Credit_Utilization_Ratio': float(credit_utilization_ratio),
        'Credit_History_Age': float(credit_history_age),
        'Payment_of_Min_Amount': payment_of_min_amount,
        'Total_EMI_per_month': float(total_emi_per_month),
        'Amount_invested_monthly': float(amount_invested_monthly),
        'Monthly_Balance': float(monthly_balance),
        'Spent_Level': spent_level,
        'Payment_Value': payment_value,
        'personal_loan_freq': int(personal_loan_freq),
        'student_loan_freq': int(student_loan_freq),
        'mortgage_loan_freq': int(mortgage_loan_freq),
        'auto_loan_freq': int(auto_loan_freq),
        'payday_loan_freq': int(payday_loan_freq),
        'credit-builder_loan_freq': int(credit_builder_loan_freq),
        'home_equity_loan_freq': int(home_equity_loan_freq),
        'debt_consolidation_loan_freq': int(debt_consolidation_loan_freq)
    }

    with st.spinner("Mengirim data ke AWS SageMaker..."):
        try:
            result = invoke_endpoint(input_dict)
            
            final_status = result.get("credit_score_prediction", "Unknown")
            probabilities = result.get("probabilities", [])
            
            if final_status == "Good":
                st.success(f"Hasil Prediksi: **{final_status}** Credit Score")
            elif final_status == "Standard":
                st.warning(f"Hasil Prediksi: **{final_status}** Credit Score")
            elif final_status == "Poor":
                st.error(f"Hasil Prediksi: **{final_status}** Credit Score")
            else:
                st.error("Gagal membaca status prediksi dari model.")
                
            if probabilities:
                st.caption(f"Probabilitas Model (Poor, Standard, Good): {probabilities}")
                
        except ClientError as e:
            st.error(f"Eror Kredensial/Akses AWS: Pastikan EC2 memiliki IAM Role yang diizinkan untuk 'sagemaker:InvokeEndpoint'. Detail: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memanggil SageMaker: {e}")