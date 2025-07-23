# NPC: Detecting and Explaining LLM-Generated Source Code

This repository contains the source code, datasets, evaluation results, and related resources for our system.

## 📁 Folder Structure

- **evaluation_results**  
  The evaluation results after letting participants use our system.

- **npc_web**  
  The main web application for running the system (see setup guide below).

- **templates**  
  HTML files for the frontend web pages.

- **static**  
  CSS files for styling the web pages.

- **training_dataset**  
  The training dataset used for finding neighborhood samples.

- **unixcoder-base**  
  The UnixCoder model and files used for computing code embeddings.

---

## ⚡ Installation & Usage

1. **Clone this repository**
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo/npc_web
    ```

2. **Create and activate a virtual environment**
    ```bash
    python3.11 -m venv myenv
    source myenv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install Flask
    ```

3. **Start required services in separate terminals:**

    - **Terminal 1: Start Redis server**
        ```bash
        redis-server
        # To check: redis-cli ping
        # To stop: redis-cli shutdown
        ```
    - **Terminal 2: Start Celery worker**
        ```bash
        python -m celery -A app.celery worker --loglevel=info
        ```
    - **Terminal 3: Start the Flask app**
        ```bash
        python app.py
        ```

