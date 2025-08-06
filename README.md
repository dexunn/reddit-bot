# Reddit Toxicity Monitoring Bot 🔍

This is a lightweight NLP-powered API that scans Reddit comments and flags toxic content using a BERT-based model. Built with `FastAPI`, this project is designed for fast deployment and integration into larger monitoring or alerting systems.

## Features

- Fetches live Reddit comments using `praw`
- Classifies toxicity using a transformer-based model (`transformers`, `torch`)
- Exposes a RESTful API via `FastAPI`
- Ready for deployment with Docker

## Tech Stack

- **Python**
- **FastAPI** – for the REST API
- **HuggingFace Transformers** – for NLP model
- **PyTorch** – for model inference
- **PRAW** – to access Reddit’s API
- **Uvicorn** – ASGI server
- **Docker** – containerized for easy deployment
