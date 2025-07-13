from fastapi import FastAPI
import os
import praw
import requests
from transformers import pipeline
from datetime import datetime, timezone, timedelta
import gc
import torch

# Disable gradients
torch.set_grad_enabled(False)

app = FastAPI()

# ========== ENV VARIABLES ==========
REDDIT_CLIENT_ID = os.environ['REDDIT_CLIENT_ID']
REDDIT_CLIENT_SECRET = os.environ['REDDIT_CLIENT_SECRET']
REDDIT_USER_AGENT = os.environ['REDDIT_USER_AGENT']
REDDIT_USERNAME = os.environ['REDDIT_USERNAME']
REDDIT_PASSWORD = os.environ['REDDIT_PASSWORD']
TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_IDS = os.environ['TELEGRAM_CHAT_IDS'].split(',')

# ========== INIT REDDIT ==========
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

# ========== TIMEZONE ==========
sg_timezone = timezone(timedelta(hours=8))

# ========== Lazy Load Sentiment Model ==========
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        print("🧠 Lazy-loading sentiment model...")
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",
            device=-1
        )
        print("✅ Model loaded.")
    return classifier

# ========== Routes ==========
@app.get("/")
@app.post("/")
async def run_bot():
    try:
        print("🚀 Starting Reddit scan...")
        scan_reddit()
        print("✅ Scan completed.")
        return {"status": "success", "message": "Reddit bot triggered successfully"}
    except Exception as e:
        error_msg = f"🔥 ERROR: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

# ========== Core Logic ==========
def scan_reddit():
    sentiment = get_classifier()
    subreddit = reddit.subreddit("NationalServiceSG")
    keywords = ["5sir", "5 sir"]
    print("🤖 Scanning for 5SIR mentions...")

    for submission in subreddit.new(limit=20):
        post_date = datetime.fromtimestamp(submission.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
        post_text = (submission.title + " " + (submission.selftext or "")).strip()
        lower_post_text = post_text.lower()

        post_mentions_5sir = any(k in lower_post_text for k in keywords)
        post_result = sentiment(post_text[:512])[0]
        post_label = post_result["label"]
        post_score = post_result["score"]

        if post_mentions_5sir:
            send_telegram_alert(f"🚨 5SIR Post Detected: {post_label} ({post_score:.3f})\n{post_text[:300]}...")

    gc.collect()
