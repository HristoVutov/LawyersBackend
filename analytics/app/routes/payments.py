"""
Payment routes using Stripe.
"""
import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional

from app.config import get_settings
from app.database import get_db
from app.models import User, Transaction
from app.routes.users import get_current_user

router = APIRouter(prefix="/api/payments", tags=["payments"])

# Initialize Stripe
settings = get_settings()
stripe.api_key = settings.stripe_secret_key


class CheckoutRequest(BaseModel):
    """Request to create a checkout session."""
    pack_id: str  # e.g. "basic_10", "pro_50"


class CheckoutResponse(BaseModel):
    """Checkout session URL."""
    url: str


# Credit Packs Configuration (Hardcoded for now)
CREDIT_PACKS = {
    "basic_10": {"amount_cents": 1000, "credits": 10.0, "name": "Basic Pack (10 Credits)"},
    "pro_50": {"amount_cents": 4500, "credits": 50.0, "name": "Pro Pack (50 Credits)"},
    "enterprise_100": {"amount_cents": 8000, "credits": 100.0, "name": "Enterprise Pack (100 Credits)"},
}


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    request: CheckoutRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Stripe Checkout Session for purchasing credits.
    """
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    pack = CREDIT_PACKS.get(request.pack_id)
    if not pack:
        raise HTTPException(status_code=400, detail="Invalid pack_id")

    try:
        # Create pending transaction
        transaction = Transaction(
            user_id=current_user.id,
            amount_cents=pack["amount_cents"],
            credits_amount=pack["credits"],
            status="pending"
        )
        db.add(transaction)
        await db.commit()
        await db.refresh(transaction)

        # Create Stripe Session
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": pack["name"],
                        },
                        "unit_amount": pack["amount_cents"],
                    },
                    "quantity": 1,
                },
            ],
            mode="payment",
            success_url=f"http://localhost:5173/dashboard?payment=success&session_id={{CHECKOUT_SESSION_ID}}", # Update domain in prod
            cancel_url=f"http://localhost:5173/dashboard?payment=cancelled",
            metadata={
                "transaction_id": str(transaction.id),
                "user_id": str(current_user.id),
                "credits": str(pack["credits"]),
            },
        )
        
        # Update transaction with session ID
        transaction.stripe_session_id = checkout_session.id
        await db.commit()

        return CheckoutResponse(url=checkout_session.url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Handle Stripe webhooks to fulfill orders.
    """
    print("🔔 Webhook Endpoint Hit!") # Debug log
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    print(f"📥 Payload received. Signature: {sig_header[:10]}...") # Debug log

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
        print("✅ Signature verified.")
    except ValueError as e:
        print(f"❌ Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        print(f"❌ Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    print(f"ℹ️ Event Type: {event['type']}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        transaction_id = session.get("metadata", {}).get("transaction_id")
        print(f"🔍 Processing Session. Transaction ID from metadata: {transaction_id}")
        
        if transaction_id:
            # Fulfill order
            result = await db.execute(select(Transaction).where(Transaction.id == transaction_id))
            transaction = result.scalar_one_or_none()
            
            if transaction:
                print(f"✅ Transaction found: {transaction.id}, Status: {transaction.status}")
                if transaction.status != "completed":
                    transaction.status = "completed"
                    from datetime import datetime
                    transaction.completed_at = datetime.utcnow()
                    transaction.payment_intent_id = session.get("payment_intent")
                    
                    # Add credits to user
                    # Need to fetch user first
                    user_result = await db.execute(select(User).where(User.id == transaction.user_id))
                    user = user_result.scalar_one_or_none()
                    if user:
                        print(f"👤 User found: {user.id}. Adding {transaction.credits_amount} credits.")
                        user.available_credits += transaction.credits_amount
                    else:
                        print(f"❌ User not found for transaction search.")
                    
                    await db.commit()
                    print("💾 DB Commit successful.")
            else:
                print(f"❌ Transaction {transaction_id} not found in DB.")
        else:
             print("⚠️ No transaction_id in session metadata.")

    return {"status": "success"}
