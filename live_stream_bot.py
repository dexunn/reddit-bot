import praw
import requests
from transformers import pipeline
from datetime import datetime, timezone, timedelta
import time

# ========== SETUP TELEGRAM ==========
bot_token = '8078149602:AAHhW7ZYprkDyTAMCLEYJwFlAP7Q3u2Lqwc'
chat_ids = ['429377238', '121026884']

def send_telegram_alert(message):
    for cid in chat_ids:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': cid, 'text': message}
        requests.post(url, data=payload)

# ========== SETUP REDDIT ==========
reddit = praw.Reddit(
    client_id='4utxbFsrfTEy9KS-ZhAkUQ',
    client_secret='CvAaTfEZ3Cb8BxJfOOtcY_gJSG-hww',
    user_agent='cacheways',
    username='Motor_Ratio7092',
    password='dexunhan2311'
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

# ========== Helper: Truncate Long Text ==========
def truncate_text(text, max_length=512):
    if len(text) > max_length:
        print("⚠️ Text truncated for sentiment analysis.")
    return text[:max_length]

# ========== Helper: Convert Label to Emoji and Name ==========
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
print("🤖 BERT-powered bot is LIVE and monitoring new Reddit activity...")

# ========== MONITOR NEW POSTS ==========
def process_submission(submission):
    post_date = datetime.fromtimestamp(submission.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
    post_text = (submission.title + " " + (submission.selftext or "")).strip()
    lower_post_text = post_text.lower()

    # Check if post mentions 5sir
    post_mentions_5sir = any(keyword in lower_post_text for keyword in keywords)

    # Analyze post sentiment
    post_result = classifier(truncate_text(post_text))[0]
    post_label, post_name = label_to_text(post_result["label"])
    post_score = post_result["score"]

    telegram_msg = (
        f"🚨 *5SIR POST/COMMENT Activity Detected!*\n"
        f"📅 Date: {post_date} (SGT)\n"
        f"{post_label} Post Sentiment: {post_name} ({post_score:.3f})\n"
        f"👤 Author: u/{submission.author}\n"
        f"📍 Subreddit: r/{submission.subreddit.display_name}\n"
        f"📝 Title: {submission.title}\n"
        f"🔗 https://reddit.com{submission.permalink}\n\n"
        f"💬 All Comments Under This Post:\n"
        f"---------------------------------------\n"
    )

    found_5sir_in_comments = False

    # ✅ SCAN COMMENTS UNDER THIS POST
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comment_text = comment.body.strip()
        lower_comment_text = comment_text.lower()

        if any(keyword in lower_comment_text for keyword in keywords):
            found_5sir_in_comments = True
            comment_result = classifier(truncate_text(comment_text))[0]
            comment_label, comment_name = label_to_text(comment_result["label"])
            comment_score = comment_result["score"]
            comment_date = datetime.fromtimestamp(comment.created_utc, tz=sg_timezone).strftime('%Y-%m-%d %H:%M:%S')
            comment_link = f"https://reddit.com{comment.permalink}"

            print(f"   {comment_label} COMMENT MENTIONS 5SIR ({comment_name} {comment_score:.3f}) — u/{comment.author}")

            telegram_msg += (
                f"{comment_label} 📅 {comment_date} (SGT)\n"
                f"👤 u/{comment.author}\n"
                f"💭 {comment_text[:300]}...\n"
                f"💬 Sentiment: {comment_name} ({comment_score:.3f})\n"
                f"🔗 {comment_link}\n"
                f"---------------------------------------\n"
            )

    # 🔥 Send alert if post mentions 5sir OR any comment mentions 5sir
    if post_mentions_5sir or found_5sir_in_comments:
        telegram_msg += "============================="
        send_telegram_alert(telegram_msg)

# ========== STREAM NEW SUBMISSIONS ==========
while True:
    try:
        for submission in subreddit.stream.submissions(skip_existing=True):
            process_submission(submission)
    except Exception as e:
        print(f"⚠️ Error occurred: {e}. Retrying in 10 seconds...")
        time.sleep(10)
