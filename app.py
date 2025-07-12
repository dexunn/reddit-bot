import praw
import requests
from transformers import pipeline
from datetime import datetime, timezone, timedelta
import os

# ========== SETUP TELEGRAM ==========
bot_token = os.environ['TELEGRAM_BOT_TOKEN']
chat_ids = os.environ['TELEGRAM_CHAT_IDS'].split(',')

def send_telegram_alert(message):
    for cid in chat_ids:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': cid, 'text': message}
        requests.post(url, data=payload)

# ========== SETUP REDDIT ==========
reddit = praw.Reddit(
    client_id=os.environ['REDDIT_CLIENT_ID'],
    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
    user_agent=os.environ['REDDIT_USER_AGENT'],
    username=os.environ['REDDIT_USERNAME'],
    password=os.environ['REDDIT_PASSWORD']
)

# ========== LOAD BERT ==========
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    framework="pt"
)

# ========== TIMEZONE FOR SINGAPORE ==========
sg_timezone = timezone(timedelta(hours=8))

# ========== Helper Functions ==========
def truncate_text(text, max_length=512):
    return text[:max_length] if len(text) > max_length else text

def label_to_text(label):
    mapping = {
        "LABEL_0": ("🔴", "Negative"),
        "LABEL_1": ("⚪", "Neutral"),
        "LABEL_2": ("🟢", "Positive")
    }
    return mapping.get(label, ("❓", "Unknown"))

# ========== SETTINGS ==========
subreddit = reddit.subreddit("NationalServiceSG")
keywords = ["5sir", "5 sir"]
print("🤖 Bot polling Reddit for new activity...")

# ========== SCAN RECENT POSTS ==========
for submission in subreddit.new(limit=50):  # Scan latest 50 posts
    post_date = datetime.fromtimestamp(submission.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
    post_text = (submission.title + " " + (submission.selftext or "")).strip().lower()

    post_mentions_5sir = any(keyword in post_text for keyword in keywords)

    if post_mentions_5sir:
        post_result = classifier(truncate_text(post_text))[0]
        post_label, post_name = label_to_text(post_result["label"])
        post_score = post_result["score"]

        telegram_msg = (
            f"🚨 *5SIR POST Detected!*\n"
            f"📅 Date: {post_date} (SGT)\n"
            f"{post_label} Sentiment: {post_name} ({post_score:.3f})\n"
            f"👤 Author: u/{submission.author}\n"
            f"📝 Title: {submission.title}\n"
            f"🔗 https://reddit.com{submission.permalink}"
        )
        send_telegram_alert(telegram_msg)

    # ✅ SCAN COMMENTS UNDER THIS POST
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        comment_text = comment.body.strip().lower()
        if any(keyword in comment_text for keyword in keywords):
            comment_result = classifier(truncate_text(comment_text))[0]
            comment_label, comment_name = label_to_text(comment_result["label"])
            comment_score = comment_result["score"]
            comment_date = datetime.fromtimestamp(comment.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
            comment_link = f"https://reddit.com{comment.permalink}"

            telegram_msg = (
                f"🚨 *5SIR COMMENT Detected!*\n"
                f"📅 Date: {comment_date} (SGT)\n"
                f"{comment_label} Sentiment: {comment_name} ({comment_score:.3f})\n"
                f"👤 u/{comment.author}\n"
                f"💭 {comment.body[:300]}...\n"
                f"🔗 {comment_link}"
            )
            send_telegram_alert(telegram_msg)

print("✅ Polling complete. Exiting...")
