from fastapi import FastAPI
import os
import praw
import requests
from transformers import pipeline
from datetime import datetime, timezone, timedelta
import gc
import torch
from time import time
import html

# Disable gradients for faster inference
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

# ========== Emoji Mapping ==========
def label_to_text(label):
    mapping = {
        "NEGATIVE": ("🔴", "Negative"),
        "POSITIVE": ("🟢", "Positive")
    }
    return mapping.get(label, ("⚪", "Neutral"))

# ========== Telegram Alert ==========
def send_telegram_alert(message):
    # Escape HTML and enforce length limit
    escaped_message = html.escape(message)
    MAX_LENGTH = 4000
    if len(escaped_message) > MAX_LENGTH:
        print("⚠️ Message too long, truncating...")
        escaped_message = escaped_message[:MAX_LENGTH] + "\n\n...truncated"

    for cid in TELEGRAM_CHAT_IDS:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': cid,
            'text': escaped_message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload)
        print(f"📤 Sent alert to {cid} – Status: {response.status_code}")

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
    cutoff_timestamp = time() - (5 * 60)  # Last 5 minutes
    print(f"⏳ Scanning content since: {datetime.fromtimestamp(cutoff_timestamp, tz=sg_timezone)}")

    # 🔥 Scan New Posts
    for submission in subreddit.new(limit=20):
        if submission.created_utc < cutoff_timestamp:
            continue  # Only alert for posts created in last 5 mins

        post_date = datetime.fromtimestamp(submission.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
        post_text = (submission.title + " " + (submission.selftext or "")).strip()
        lower_post_text = post_text.lower()

        if any(k in lower_post_text for k in keywords):
            post_result = sentiment(post_text[:512])[0]
            post_emoji, post_sentiment = label_to_text(post_result["label"])
            post_score = post_result["score"]

            telegram_msg = (
                f"🚨 *5SIR POST Detected!*\n"
                f"📅 Date: {post_date} (SGT)\n"
                f"{post_emoji} Sentiment: {post_sentiment} ({post_score:.3f})\n"
                f"👤 Author: u/{submission.author}\n"
                f"📝 Title: {submission.title}\n"
                f"🔗 https://reddit.com{submission.permalink}\n"
                f"============================="
            )
            send_telegram_alert(telegram_msg)

    # 🔥 Scan New Comments Site-Wide
    for comment in subreddit.comments(limit=50):
        if comment.created_utc < cutoff_timestamp:
            continue  # Only alert for comments created in last 5 mins

        comment_text = comment.body.strip() if comment.body else ""
        lower_comment_text = comment_text.lower()

        if any(k in lower_comment_text for k in keywords):
            comment_result = sentiment(comment_text[:512])[0]
            comment_emoji, comment_sentiment = label_to_text(comment_result["label"])
            comment_score = comment_result["score"]
            comment_date = datetime.fromtimestamp(comment.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
            comment_link = f"https://reddit.com{comment.permalink}"

            telegram_msg = (
                f"🚨 *5SIR COMMENT Detected!*\n"
                f"📅 Date: {comment_date} (SGT)\n"
                f"{comment_emoji} Sentiment: {comment_sentiment} ({comment_score:.3f})\n"
                f"👤 Author: u/{comment.author}\n"
                f"💭 {comment_text[:300]}...\n"
                f"🔗 {comment_link}\n"
                f"============================="
            )
            send_telegram_alert(telegram_msg)

    gc.collect()
