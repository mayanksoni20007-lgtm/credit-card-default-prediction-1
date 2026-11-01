# Credit Card Default Prediction — Streamlit app

This app includes a customer input form for the 23 features in the Default of Credit Card Clients dataset. Enter the demographic profile, six months of repayment status, bill amounts, and previous payment amounts. Select Logistic Regression, Random Forest, or XGBoost to estimate default probability.

The app fetches the public dataset from UCI and trains the selected classifier on first use. Dataset and fitted model are cached by Streamlit.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

Keep `streamlit_app.py` and `requirements.txt` at the GitHub repository root and include the `website/` folder. Deploy `streamlit_app.py` in Streamlit Community Cloud. Commit and push updates to trigger redeployment.

Dataset source: UCI Machine Learning Repository, [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bcredit%2Bcard%2Bclients), DOI: 10.24432/C55H3H. It contains 30,000 records from Taiwan in 2005. Educational demonstration only; not for real credit or lending decisions.
