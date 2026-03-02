# NPC: Detecting and Explaining LLM-Generated Source Code

This repository contains the source code, datasets, prompt template, evaluation results, and related resources for our system. 

## 🤖 Model Weights
The trained GraphCodeBERT model weights for this project are hosted on Hugging Face to keep the repository lightweight. 
* **Hugging Face Repository:** [FFFFAHHH/NPC_model](https://huggingface.co/FFFFAHHH/NPC_model)

---

## 📁 Folder Structure
- **prompt.md**
  The prompt template used by NPC to generate the explanation.
  
- **evaluation_results** The evaluation results after letting participants use our system.

- **npc_web** The main web application for running the system (see setup guide below).

- **templates** HTML files for the frontend web pages.

- **static** CSS files for styling the web pages.

- **training_dataset** The training dataset used for finding neighborhood samples.

- **unixcoder-base** The UnixCoder model and files used for computing code embeddings.

---

## ⚡ Installation & Usage

1. **Clone this repository**
    ```bash
    git clone [https://github.com/pachanitha/NPC_Project.git](https://github.com/pachanitha/NPC_Project.git)
    cd NPC_Project/npc_web
    ```

2. **Create and activate a virtual environment**
    ```bash
    python3.11 -m venv myenv
    source myenv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install Flask huggingface_hub
    ```

3. **Download the trained model from Hugging Face**
    Before running the application, download the trained model weights into your project directory using the Hugging Face CLI:
    ```bash
    huggingface-cli download FFFFAHHH/detect_gen_code Detect_AI.pth --local-dir .
    ```
    *(Note: Ensure the `--local-dir` matches the path where your `model.py` or `app.py` expects to find the `.pth` file).*

4. **Start required services in separate terminals:**

    - **Terminal 1: Start Redis server**
        ```bash
        redis-server
        # To check: redis-cli ping
        # To stop: redis-cli shutdown
        ```
    - **Terminal 2: Start Celery worker**
        ```bash
        # Make sure your virtual environment is activated in this terminal too
        python -m celery -A app.celery worker --loglevel=info
        ```
    - **Terminal 3: Start the Flask app**
        ```bash
        # Make sure your virtual environment is activated in this terminal too
        python app.py
        ```
