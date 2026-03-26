"""
AI Scanner app: Text scanner and Receipt scanner (expense tracking).
Receipt mode extracts product names and prices with strict, no-guess rules so prices are never wrong.
"""
import streamlit as st
import pandas as pd
from PIL import Image

from scanner import scan_and_generate
from receipt_scanner import (
    extract_receipt,
    add_receipt_to_expenses,
    load_expenses,
    ReceiptData,
)

st.set_page_config(page_title="AI Scanner", page_icon="📄", layout="centered")

mode = st.sidebar.radio("Mode", ["Receipt scanner (expenses)", "Text scanner"], index=0)

if mode == "Receipt scanner (expenses)":
    st.title("🧾 Receipt scanner")
    st.caption("Upload a receipt. Product names and prices are extracted and can be added to expenses. Prices are never guessed—only exact values from the receipt are used.")

    uploaded = st.file_uploader("Upload a receipt image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Receipt", use_container_width=True)

        with st.spinner("Scanning receipt…"):
            receipt = extract_receipt(image)

        st.subheader("Extracted receipt")
        if receipt.warnings:
            for w in receipt.warnings:
                st.warning(w)

        currency = getattr(receipt, "currency", "") or ""
        col1, col2 = st.columns(2)
        with col1:
            if receipt.merchant:
                st.write("**Merchant:**", receipt.merchant)
            if receipt.date:
                st.write("**Date:**", receipt.date)
            if currency:
                st.write("**Currency:**", currency)
        with col2:
            if receipt.subtotal:
                st.write("**Subtotal:**", f"{receipt.subtotal} {currency}".strip())
            if receipt.tax:
                st.write("**Tax:**", f"{receipt.tax} {currency}".strip())

        if receipt.items:
            st.write("**Line items (product name and price):**")
            rows = []
            for it in receipt.items:
                p = it.get("price", "")
                p_display = f"{p} {currency}".strip() if currency else p
                rows.append({"Product": it["product"], "Price": p_display, "Needs review": it.get("needs_review", False)})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("Add to expenses"):
                add_receipt_to_expenses(receipt)
                st.success("Receipt added to expenses.")
        else:
            st.info("No line items with valid prices were found. Check the raw text below and use a clearer image if needed.")

        with st.expander("Raw OCR text (for audit)"):
            st.text(receipt.raw_text or "(none)")

    st.subheader("Your expenses")
    expenses = load_expenses()
    if not expenses:
        st.info("No receipts added yet. Scan a receipt and click **Add to expenses**.")
    else:
        for i, entry in enumerate(expenses):
            cur = entry.get("currency", "")
            with st.expander(f"{entry.get('merchant', 'Receipt')} — {entry.get('date', '')}"):
                st.write("**Items:**")
                for it in entry.get("items", []):
                    p = it.get("price", "")
                    p_display = f"{p} {cur}".strip() if cur else p
                    st.write("-", it.get("product", ""), "—", p_display)
                if entry.get("subtotal"):
                    st.write("Subtotal:", f"{entry['subtotal']} {cur}".strip())
                if entry.get("tax"):
                    st.write("Tax:", f"{entry['tax']} {cur}".strip())

        # Export as CSV
        rows = []
        for entry in expenses:
            cur = entry.get("currency", "")
            for it in entry.get("items", []):
                p = it.get("price", "")
                rows.append({
                    "merchant": entry.get("merchant", ""),
                    "date": entry.get("date", ""),
                    "product": it.get("product", ""),
                    "price": p,
                    "price_with_currency": f"{p} {cur}".strip() if cur else p,
                    "currency": cur,
                })
        if rows:
            df_export = pd.DataFrame(rows)
            st.download_button(
                label="Export expenses as CSV",
                data=df_export.to_csv(index=False),
                file_name="expenses.csv",
                mime="text/csv",
            )

else:
    st.title("📄 Text scanner")
    st.caption("Upload an image. If it contains text, the scanner will extract it and show an output.")

    uploaded = st.file_uploader(
        "Upload an image (document, screenshot, etc.)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Your scan", use_container_width=True)

        with st.spinner("Scanning and generating output…"):
            extracted_text, ai_output = scan_and_generate(image)

        st.subheader("Extracted text")
        if extracted_text:
            st.text_area("Scanned text", value=extracted_text, height=120, disabled=True)
            st.download_button(
                label="Download as .txt",
                data=extracted_text,
                file_name="scanned_text.txt",
                mime="text/plain",
            )
        else:
            st.info("No text was detected in this image. Try a clearer or higher-resolution image with visible text.")

        st.subheader("AI-generated output")
        st.markdown(ai_output)
    else:
        st.info("👆 Upload an image above to scan.")
