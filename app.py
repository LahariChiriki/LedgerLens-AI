
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from database import (
    add_transaction,
    get_transactions,
    update_transaction,
    delete_transaction,
    get_summary,
)

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LedgerLens AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
MODEL_NAME = "gemini-2.5-flash"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0B1120;
        color: #E5E7EB;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #172238, #111827);
        border: 1px solid #26354D;
        border-radius: 16px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #A7B4C8;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC;
    }

    [data-testid="stMetricDelta"] {
        color: #6EE7B7;
    }

    div[data-testid="stForm"] {
        background: #111B2D;
        border: 1px solid #26354D;
        border-radius: 14px;
        padding: 20px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: #4F7DF3;
        color: white;
        border: 1px solid #4F7DF3;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background: #111827;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #26354D;
        border-radius: 12px;
        overflow: hidden;
    }

    .hero-card {
        background: linear-gradient(120deg, #172B4D, #172238 55%, #19233A);
        border: 1px solid #31466B;
        padding: 26px;
        border-radius: 18px;
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 7px;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #B7C5DB;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #94A3B8;
        font-size: 13px;
    }

    .ai-card {
        background: #111B2D;
        border: 1px solid #31466B;
        padding: 18px;
        border-radius: 14px;
        margin: 8px 0;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_inr(amount):
    return f"₹{float(amount):,.2f}"


def get_dataframe():
    records = get_transactions()
    columns = [
        "id", "date", "type", "category", "amount", "description"
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)

    for column in columns:
        if column not in frame.columns:
            frame[column] = None

    frame["amount"] = pd.to_numeric(
        frame["amount"], errors="coerce"
    ).fillna(0)

    frame["date"] = pd.to_datetime(
        frame["date"], errors="coerce"
    ).dt.date

    frame = frame.dropna(subset=["date"])
    return frame[columns]


def calculate_summary(frame):
    if frame.empty:
        return {"income": 0.0, "expenses": 0.0, "balance": 0.0}

    income = float(
        frame.loc[frame["type"] == "Income", "amount"].sum()
    )
    expenses = float(
        frame.loc[frame["type"] == "Expense", "amount"].sum()
    )

    return {
        "income": income,
        "expenses": expenses,
        "balance": income - expenses,
    }


def make_financial_context(frame):
    summary = calculate_summary(frame)

    transaction_lines = []

    for _, row in frame.iterrows():
        transaction_lines.append(
            f"Date: {row['date']}, "
            f"Type: {row['type']}, "
            f"Category: {row['category']}, "
            f"Amount: ₹{row['amount']:.2f}, "
            f"Description: {row['description'] or 'None'}"
        )

    transaction_text = (
        "\n".join(transaction_lines)
        if transaction_lines
        else "No transactions recorded for the selected period."
    )

    return f"""
LedgerLens AI financial records for the selected period:

Total Income: ₹{summary['income']:.2f}
Total Expenses: ₹{summary['expenses']:.2f}
Net Balance (Income minus Expenses): ₹{summary['balance']:.2f}
Number of transactions: {len(frame)}

Transactions:
{transaction_text}
"""


def ask_gemini(prompt):
    if client is None:
        raise ValueError(
            "Gemini API key not found. Check GEMINI_API_KEY in your .env file."
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        return "No AI response was returned. Please try again."

    return response.text


def show_ai_error(error):
    st.error(f"AI request failed: {error}")


# =========================================================
# HEADER AND TOP NAVIGATION
# =========================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="
                background:#4F7DF3;
                width:44px;height:44px;border-radius:13px;
                display:flex;align-items:center;justify-content:center;
                font-size:24px;color:white;">
                ◈
            </div>
            <div>
                <div style="font-size:25px;font-weight:800;color:#F8FAFC;">
                    LedgerLens <span style="color:#70A0FF;">AI</span>
                </div>
                <div style="font-size:12px;color:#94A3B8;">
                    SMART FINANCIAL TRACKING
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right;padding-top:7px;">
            <span style="
                color:#86EFAC;background:#123328;
                border:1px solid #22543D;border-radius:20px;
                padding:7px 12px;font-size:12px;">
                ● Level 1
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

page = st.radio(
    "Main navigation",
    [
        "Dashboard",
        "Transactions",
        "AI Summary",
        "AI Assistant",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)

st.divider()


# =========================================================
# DATE FILTER
# =========================================================

with st.expander("📅 Filter financial data by date", expanded=False):
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

    with filter_col1:
        start_date = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=6),
            key="global_start_date",
        )

    with filter_col2:
        end_date = st.date_input(
            "End date",
            value=date.today(),
            key="global_end_date",
        )

    with filter_col3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Reset dates"):
            st.session_state["global_start_date"] = date.today() - timedelta(days=6)
            st.session_state["global_end_date"] = date.today()
            st.rerun()

if start_date > end_date:
    st.error("Start date cannot be later than end date.")
    st.stop()


# =========================================================
# PREPARE FILTERED DATA
# =========================================================

all_df = get_dataframe()

if not all_df.empty:
    filtered_df = all_df[
        (all_df["date"] >= start_date)
        & (all_df["date"] <= end_date)
    ].copy()
else:
    filtered_df = all_df.copy()

summary = calculate_summary(filtered_df)
financial_context = make_financial_context(filtered_df)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Financial Overview</div>
            <div class="hero-subtitle">
                Welcome back! Here is your financial snapshot
                for {start_date.strftime('%d %b %Y')}
                – {end_date.strftime('%d %b %Y')}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "TOTAL INCOME",
            format_inr(summary["income"]),
            help="Sum of recorded income in the selected period.",
        )

    with metric2:
        st.metric(
            "TOTAL EXPENSES",
            format_inr(summary["expenses"]),
            help="Sum of recorded expenses in the selected period.",
        )

    with metric3:
        st.metric(
            "NET BALANCE",
            format_inr(summary["balance"]),
            help="Income minus expenses in the selected period.",
        )

    st.write("")
    chart_col, profit_col = st.columns([1.6, 1])

    with chart_col:
        st.markdown(
            '<div class="section-title">Income vs Expenses</div>',
            unsafe_allow_html=True,
        )

        comparison_df = pd.DataFrame({
            "Type": ["Income", "Expenses"],
            "Amount": [summary["income"], summary["expenses"]],
        })

        st.bar_chart(
            comparison_df.set_index("Type"),
            use_container_width=True,
        )

    with profit_col:
        st.markdown(
            '<div class="section-title">Profit / Loss Overview</div>',
            unsafe_allow_html=True,
        )

        if summary["balance"] > 0:
            status = "Positive balance"
            status_color = "#86EFAC"
            status_icon = "↗"
        elif summary["balance"] < 0:
            status = "Negative balance"
            status_color = "#FCA5A5"
            status_icon = "↘"
        else:
            status = "Break-even"
            status_color = "#93C5FD"
            status_icon = "—"

        st.markdown(
            f"""
            <div class="ai-card">
                <div class="small-muted">Net result for selected period</div>
                <div style="font-size:30px;font-weight:800;
                    color:{status_color};margin:12px 0;">
                    {status_icon} {format_inr(summary['balance'])}
                </div>
                <div style="color:{status_color};font-size:14px;">
                    {status}
                </div>
                <hr style="border-color:#26354D;">
                <div class="small-muted">
                    Calculated as total income minus total expenses.
                    This is not a full accounting profit calculation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Expense Breakdown by Category</div>',
        unsafe_allow_html=True,
    )

    expense_df = filtered_df[
        filtered_df["type"] == "Expense"
    ] if not filtered_df.empty else filtered_df

    if not expense_df.empty:
        category_totals = (
            expense_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(category_totals, use_container_width=True)
    else:
        st.info("No expenses found for the selected date range.")

    st.markdown(
        '<div class="section-title">Recent Transactions</div>',
        unsafe_allow_html=True,
    )

    if not filtered_df.empty:
        recent = filtered_df.sort_values(
            ["date", "id"], ascending=[False, False]
        ).head(5).copy()

        recent["date"] = recent["date"].astype(str)
        recent["amount"] = recent["amount"].map(format_inr)

        st.dataframe(
            recent[["date", "type", "category", "amount", "description"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No transactions found for this date range.")

    if st.button("Generate AI Financial Summary", type="primary"):
        st.session_state["generate_dashboard_summary"] = True

    if st.session_state.get("generate_dashboard_summary", False):
        try:
            with st.spinner("Analyzing your financial records..."):
                answer = ask_gemini(
                    "Summarize these financial records in simple English. "
                    "Mention the income, expenses, net balance, and one "
                    "observation supported by the data. Do not invent facts. "
                    "Clearly state that net balance is income minus expenses.\n\n"
                    + financial_context
                )

            st.markdown(
                '<div class="section-title">AI Insights</div>',
                unsafe_allow_html=True,
            )
            st.markdown(answer)

        except Exception as error:
            show_ai_error(error)


# =========================================================
# TRANSACTIONS: ADD, FILTER, EDIT, DELETE
# =========================================================

elif page == "Transactions":

    st.markdown(
        '<div class="section-title">Transaction Management</div>',
        unsafe_allow_html=True,
    )
    st.caption("Record, search, edit, and remove financial entries.")

    add_tab, history_tab = st.tabs(
        ["➕ Add Transaction", "📋 Transaction History"]
    )

    with add_tab:
        st.subheader("New Ledger Entry")

        with st.form("add_transaction_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                transaction_date = st.date_input(
                    "Transaction date",
                    value=date.today(),
                )

                transaction_type = st.selectbox(
                    "Transaction type",
                    ["Income", "Expense"],
                )

            with col2:
                amount = st.number_input(
                    "Amount (₹)",
                    min_value=0.0,
                    step=100.0,
                    format="%.2f",
                )

                description = st.text_input(
                    "Description",
                    placeholder="Example: Payment received from customer",
                )

            if transaction_type == "Income":
                categories = [
                    "Sales",
                    "Service Revenue",
                    "Customer Payment",
                    "Interest Received",
                    "Capital Introduced",
                    "Loan Received",
                    "Other Income",
                ]
            else:
                categories = [
                    "Purchases",
                    "Supplier Payment",
                    "Rent",
                    "Salaries Paid",
                    "Electricity",
                    "Water Bill",
                    "Transport",
                    "Office Expenses",
                    "Repairs and Maintenance",
                    "Taxes",
                    "Bank Charges",
                    "Loan Repayment",
                    "Food",
                    "Other Expense",
                ]

            category_col, custom_col = st.columns(2)

            with category_col:
                category = st.selectbox("Category", categories)

            with custom_col:
                custom_category = st.text_input(
                    "Custom category (optional)",
                    placeholder="Enter a different category",
                )

            submitted = st.form_submit_button(
                "Save Transaction",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                final_category = (
                    custom_category.strip()
                    if custom_category.strip()
                    else category
                )

                if amount <= 0:
                    st.error("Amount must be greater than zero.")
                elif not final_category:
                    st.error("Please provide a category.")
                else:
                    add_transaction(
                        str(transaction_date),
                        transaction_type,
                        final_category,
                        amount,
                        description.strip(),
                    )
                    st.success("Transaction saved successfully.")
                    st.rerun()

    with history_tab:
        st.subheader("Saved Transactions")

        if all_df.empty:
            st.info("No transactions yet. Add your first ledger entry.")
        else:
            search_text = st.text_input(
                "Search by category or description",
                placeholder="e.g. Rent, Sales, supplier",
            )

            type_filter = st.selectbox(
                "Transaction type",
                ["All", "Income", "Expense"],
                key="history_type_filter",
            )

            history_df = filtered_df.copy()

            if type_filter != "All":
                history_df = history_df[
                    history_df["type"] == type_filter
                ]

            if search_text.strip():
                term = search_text.strip()
                matches = (
                    history_df["category"].astype(str).str.contains(
                        term, case=False, na=False
                    )
                    | history_df["description"].astype(str).str.contains(
                        term, case=False, na=False
                    )
                )
                history_df = history_df[matches]

            display_df = history_df.copy()
            display_df["date"] = display_df["date"].astype(str)
            display_df["amount"] = display_df["amount"].map(format_inr)

            st.dataframe(
                display_df[
                    ["id", "date", "type", "category", "amount", "description"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.subheader("Edit or Delete Entry")

            editable_df = filtered_df if not filtered_df.empty else all_df

            transaction_ids = editable_df["id"].astype(int).tolist()

            selected_id = st.selectbox(
                "Choose transaction ID",
                transaction_ids,
            )

            selected_row = all_df[
                all_df["id"] == selected_id
            ].iloc[0]

            with st.form("edit_transaction_form"):
                edit_col1, edit_col2 = st.columns(2)

                with edit_col1:
                    new_date = st.date_input(
                        "Date",
                        value=selected_row["date"],
                    )

                    new_type = st.selectbox(
                        "Type",
                        ["Income", "Expense"],
                        index=0 if selected_row["type"] == "Income" else 1,
                    )

                    new_category = st.text_input(
                        "Category",
                        value=str(selected_row["category"]),
                    )

                with edit_col2:
                    new_amount = st.number_input(
                        "Amount (₹)",
                        min_value=0.01,
                        value=max(float(selected_row["amount"]), 0.01),
                        format="%.2f",
                    )

                    new_description = st.text_area(
                        "Description",
                        value=str(selected_row["description"] or ""),
                    )

                update_clicked = st.form_submit_button(
                    "Update Transaction",
                    type="primary",
                    use_container_width=True,
                )

                if update_clicked:
                    if not new_category.strip():
                        st.error("Category cannot be empty.")
                    elif new_amount <= 0:
                        st.error("Amount must be greater than zero.")
                    else:
                        update_transaction(
                            int(selected_id),
                            str(new_date),
                            new_type,
                            new_category.strip(),
                            new_amount,
                            new_description.strip(),
                        )
                        st.success("Transaction updated successfully.")
                        st.rerun()

            st.warning(
                "Deleting a transaction permanently removes that record "
                "from the local database."
            )

            confirm_delete = st.checkbox(
                f"Confirm deletion of transaction ID {selected_id}",
                key="confirm_delete_transaction",
            )

            if st.button("Delete Selected Transaction"):
                if confirm_delete:
                    delete_transaction(int(selected_id))
                    st.success("Transaction deleted.")
                    st.rerun()
                else:
                    st.error("Confirm deletion using the checkbox first.")


# =========================================================
# AI FINANCIAL SUMMARY
# =========================================================

elif page == "AI Summary":

    st.markdown(
        '<div class="section-title">AI Financial Summary</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Get a plain-English explanation of your income, expenses, "
        "and net balance for the selected dates."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", format_inr(summary["income"]))
    col2.metric("Total Expenses", format_inr(summary["expenses"]))
    col3.metric("Net Balance", format_inr(summary["balance"]))

    if st.button("✨ Generate Summary", type="primary"):
        try:
            prompt = f"""
You are LedgerLens AI, a basic financial assistant.

Explain the following financial records in simple English.
Include:
1. A short summary of total income, expenses, and net balance.
2. One observation supported by the transactions.
3. One practical suggestion, if appropriate.

Use only the supplied figures. Do not invent data.
Net balance means income minus expenses; do not call it audited profit.
Keep the answer clear and concise.

{financial_context}
"""

            with st.spinner("Preparing your AI summary..."):
                answer = ask_gemini(prompt)

            st.markdown(
                '<div class="ai-card">',
                unsafe_allow_html=True,
            )
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as error:
            show_ai_error(error)

    st.caption(
        "AI responses are explanations of recorded data, not audited accounts "
        "or professional financial advice."
    )


# =========================================================
# AI CHAT ASSISTANT
# =========================================================

elif page == "AI Assistant":

    st.markdown(
        '<div class="section-title">💬 LedgerLens AI Assistant</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Ask questions about your saved transactions and the selected date range."
    )

    st.markdown(
        """
        <div class="ai-card">
            <div style="font-size:16px;font-weight:700;color:#E5EDFF;">
                What can I help you with?
            </div>
            <div class="small-muted" style="margin-top:8px;">
                Examples: What are my total expenses? Which category has
                the highest expenses? How much income did I record?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "ai_chat_messages" not in st.session_state:
        st.session_state["ai_chat_messages"] = []

    for message in st.session_state["ai_chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about your financial records...")

    if question:
        st.session_state["ai_chat_messages"].append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        conversation_history = st.session_state["ai_chat_messages"][:-1]

        history_text = "\n".join(
            f"{'User' if item['role'] == 'user' else 'Assistant'}: "
            f"{item['content']}"
            for item in conversation_history[-8:]
        )

        prompt = f"""
You are LedgerLens AI, a basic financial question-answering assistant.

Answer in simple English using the supplied financial records.
You may answer follow-up questions using the recent conversation as context.

Rules:
- Use only supplied financial data for factual financial claims.
- Do not invent transactions or amounts.
- Net balance means income minus expenses.
- If the requested information is missing, say so.
- Do not claim to modify or save transactions.
- Do not give professional tax or investment advice.

Financial records for the selected date range:
{financial_context}

Recent conversation:
{history_text}

Current user question:
{question}
"""

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_gemini(prompt)

                st.markdown(answer)

            st.session_state["ai_chat_messages"].append(
                {"role": "assistant", "content": answer}
            )

        except Exception as error:
            with st.chat_message("assistant"):
                st.error(f"Could not get an answer: {error}")

    if st.session_state["ai_chat_messages"]:
        if st.button("Clear Chat History"):
            st.session_state["ai_chat_messages"] = []
            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

footer_left, footer_right = st.columns(2)

with footer_left:
    st.caption("LedgerLens AI · Level 1 Financial Tracker")

with footer_right:
    st.caption("Python · Streamlit · SQLite · Gemini AI")
