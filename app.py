from fastapi import FastAPI
import os
import praw
import requests
from transformers import pipeline
from datetime import datetime, timezone, timedelta
import gc
import torch

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
    for cid in TELEGRAM_CHAT_IDS:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': cid, 'text': message, 'parse_mode': 'Markdown'}
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
    print("🤖 Scanning posts & comments for 5SIR mentions...")

    for submission in subreddit.new(limit=20):  # Adjust limit if needed
        post_date = datetime.fromtimestamp(submission.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
        post_text = (submission.title + " " + (submission.selftext or "")).strip()
        lower_post_text = post_text.lower()

        post_mentions_5sir = any(k in lower_post_text for k in keywords)
        post_result = sentiment(post_text[:512])[0]
        post_emoji, post_sentiment = label_to_text(post_result["label"])
        post_score = post_result["score"]

        # Start building Telegram message
        telegram_msg = (
            f"🚨 *5SIR POST/COMMENT Activity Detected!*\n"
            f"📅 Date: {post_date} (SGT)\n"
            f"{post_emoji} Post Sentiment: {post_sentiment} ({post_score:.3f})\n"
            f"👤 Author: u/{submission.author}\n"
            f"📍 Subreddit: r/{submission.subreddit.display_name}\n"
            f"📝 Title: {submission.title}\n"
            f"🔗 https://reddit.com{submission.permalink}\n\n"
            f"💬 All Comments Under This Post:\n"
            f"---------------------------------------\n"
        )

        found_5sir_in_comments = False

        # ✅ Scan comments under this post
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            comment_text = comment.body.strip() if comment.body else ""
            lower_comment_text = comment_text.lower()

            if any(k in lower_comment_text for k in keywords):
                found_5sir_in_comments = True
                comment_result = sentiment(comment_text[:512])[0]
                comment_emoji, comment_sentiment = label_to_text(comment_result["label"])
                comment_score = comment_result["score"]
                comment_date = datetime.fromtimestamp(comment.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
                comment_link = f"https://reddit.com{comment.permalink}"

                telegram_msg += (
                    f"{comment_emoji} 📅 {comment_date} (SGT)\n"
                    f"👤 u/{comment.author}\n"
                    f"💭 {comment_text[:300]}...\n"
                    f"💬 Sentiment: {comment_sentiment} ({comment_score:.3f})\n"
                    f"🔗 {comment_link}\n"
                    f"---------------------------------------\n"
                )

        # 🔥 Send alert if post OR any comment mentions 5sir
        if post_mentions_5sir or found_5sir_in_comments:
            telegram_msg += "============================="
            send_telegram_alert(telegram_msg)

    gc.collect()
