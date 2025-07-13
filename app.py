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
        print("🧠 Lazy-loading DistilBERT sentiment model...")
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",
            device=-1
        )
        print("✅ Model loaded.")
    return classifier

# ========== Telegram Alert ==========
def send_telegram_alert(message):
    for cid in TELEGRAM_CHAT_IDS:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': cid, 'text': message}
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

    for submission in subreddit.new(limit=20):  # Scan latest 20 posts
        post_date = datetime.fromtimestamp(submission.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
        post_text = (submission.title + " " + (submission.selftext or "")).strip()
        lower_post_text = post_text.lower()

        print(f"📝 Checking post: {submission.title}")
        post_mentions_5sir = any(k in lower_post_text for k in keywords)

        if post_mentions_5sir:
            print("🎯 Keyword matched in POST!")
            post_result = sentiment(post_text[:512])[0]
            post_label = post_result["label"]
            post_score = post_result["score"]
            send_telegram_alert(f"🚨 5SIR Post Detected: {post_label} ({post_score:.3f})\n{post_text[:300]}...")

        # ✅ Scan comments under this post
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            comment_text = comment.body.strip() if comment.body else ""
            lower_comment_text = comment_text.lower()

            print(f"💬 Checking comment by u/{comment.author}: {comment_text[:80]}...")
            comment_mentions_5sir = any(k in lower_comment_text for k in keywords)

            if comment_mentions_5sir:
                print("🎯 Keyword matched in COMMENT!")
                comment_result = sentiment(comment_text[:512])[0]
                comment_label = comment_result["label"]
                comment_score = comment_result["score"]
                comment_date = datetime.fromtimestamp(comment.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
                send_telegram_alert(
                    f"🚨 5SIR Comment Detected: {comment_label} ({comment_score:.3f})\n"
                    f"📅 {comment_date}\n"
                    f"👤 u/{comment.author}\n"
                    f"💭 {comment_text[:300]}..."
                )

    gc.collect()
