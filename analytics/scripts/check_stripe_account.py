
import stripe
import os
import sys

# Load env manually because we are in a script
from dotenv import load_dotenv
load_dotenv(".env")

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

try:
    account = stripe.Account.retrieve()
    print(f"API Key Account ID: {account.id}")
except Exception as e:
    print(f"Error retrieving account: {e}")
