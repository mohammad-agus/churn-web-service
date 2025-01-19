import requests

customer_id = 'xyz_123'
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

url = 'http://0.0.0.0:9696/predict'
response = requests.post(url=url, json=customer).json()

print(response)
if response['churn'] == True:
    print(f"sending promo email to {customer_id}")
else:
    print(f"not sending promo email to {customer_id}")