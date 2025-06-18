# model.py

import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM,AutoModel
from langchain_core.prompts import PromptTemplate
import openai
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import faiss
import logging
import absl.logging
import sys
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import time
from dotenv import load_dotenv
# Add custom path for UnixCoder
sys.path.append('/Users/fah/Downloads/senior_1/')
from unixcoder import UniXcoder

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
absl.logging.set_verbosity(absl.logging.INFO)
logger = logging.getLogger(__name__)

session = requests.Session()
session.timeout = 30

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure APIs
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key or gemini_api_key.strip() == "":
    logger.error("Gemini API key is missing. Please set the GEMINI_API_KEY environment variable.")
    raise ValueError("Missing Gemini API key. Please provide a valid key.")

# Configure the Gemini API with the provided key
genai.configure(api_key=gemini_api_key)

openai.api_key = os.getenv('OPENAI_API_KEY')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/Users/fah/Downloads/senior_1/web_draft/graphcodebert-base"
# Initialize GraphCodeBERT tokenizer and model
graphCodeBERT_tokenizer = AutoTokenizer.from_pretrained(model_path)
graphCodeBERT_model = AutoModel.from_pretrained(model_path, cache_dir="/Users/fah/Downloads/senior_1/web_draft/graphcodebert-base")
graphCodeBERT_model.to(device)

# Paths to precomputed embeddings and filenames for the training dataset
EMBEDDINGS_0_PATH = "/Users/fah/Downloads/senior_1/training_dataset/precomputed_embeddings_0_.npy"
EMBEDDINGS_1_PATH = "/Users/fah/Downloads/senior_1/training_dataset/precomputed_embeddings_1_.npy"
FILENAMES_0_PATH = "/Users/fah/Downloads/senior_1/training_dataset/precomputed_filenames_0_.txt"
FILENAMES_1_PATH = "/Users/fah/Downloads/senior_1/training_dataset/precomputed_filenames_1_.txt"
FAISS_INDEX_PATH = "/Users/fah/Downloads/senior_1/training_dataset/faiss_index.bin"

# Load UniXcoder
UNIXCODER_PATH = "/Users/fah/Downloads/senior_1/web_draft/unixcoder-base"
unixcoder_model = UniXcoder(UNIXCODER_PATH).to(device)

class GraphCodeBERTForClassification(nn.Module):
    def __init__(self, model):
        super(GraphCodeBERTForClassification, self).__init__()
        self.model = model
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token's output
        logits = self.classifier(cls_output)
        return logits


num_labels = 2  # Set to the number of classes in your dataset
graphCodeBERT_classification_model = GraphCodeBERTForClassification(graphCodeBERT_model)
graphCodeBERT_classification_model.to(device)

# Dataset Class
class CodeDataset(Dataset):
    def __init__(self, directories):
        self.samples = []
        self.filenames = []
        for directory in directories:
            for filename in os.listdir(directory):
                label = int(filename.split('_')[0])  # Assumes label in filename
                with open(os.path.join(directory, filename), 'r') as f:
                    code = f.read()
                    self.samples.append((code, label))
                    self.filenames.append(filename)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        code, label = self.samples[index]
        inputs = graphCodeBERT_tokenizer(code, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return {'code': code, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label, dtype=torch.long)}

    def get_filename(self, index):
        return self.filenames[index]

# Warmup the model
def warmup_model(model, tokenizer):
    dummy_input = tokenizer("dummy code", padding='max_length', max_length=512, return_tensors="pt").to(device)
    for _ in range(5):  # Run 5 dummy iterations
        with torch.no_grad():
            model(dummy_input['input_ids'], attention_mask=dummy_input['attention_mask'])

# Call warmup function before actual inference
warmup_model(graphCodeBERT_classification_model, graphCodeBERT_tokenizer)

def read_file(file_path):
    if not os.path.exists(file_path):
        logger.error(f"[ERROR] File does not exist: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as file:
        code = file.read()
    return code

# Function to classify source code
def classify_code(code):
    # Load the trained model weights
    model_path = "/Users/fah/Downloads/senior_1/graphCodeBERT_trained_model_final.pth"

    try:
        state_dict = torch.load(model_path, map_location=device)
        graphCodeBERT_classification_model.load_state_dict(state_dict, strict=True)
    except FileNotFoundError:
        logger.error(f"[ERROR] Model file not found at {model_path}. Please check the path.")
        raise
    
    # Set model to evaluation mode
    graphCodeBERT_classification_model.eval()

    # Tokenization of Input Code
    inputs = graphCodeBERT_tokenizer(
        code,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(device)  # Tokenize the code and move tensors to the device

    # Perform inference
    with torch.no_grad():
        # Forward pass to get the logits using the classification model
        outputs = graphCodeBERT_classification_model(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )
        logits = outputs  # Get the logits directly from the classification model
        probabilities = torch.softmax(logits, dim=1)  # Apply softmax to get probabilities

    # Interpretation: Class 0 = ChatGPT-generated, Class 1 = Human-written
    return probabilities[0][0].item(), probabilities[0][1].item() 

# Function to load precomputed embeddings and filenames
def load_precomputed_data(prefix):
    if prefix == "0_":
        embeddings_path, filenames_path = EMBEDDINGS_0_PATH, FILENAMES_0_PATH
    elif prefix == "1_":
        embeddings_path, filenames_path = EMBEDDINGS_1_PATH, FILENAMES_1_PATH
    else:
        raise ValueError(f"Invalid prefix: {prefix}")

    embeddings = np.load(embeddings_path)
    with open(filenames_path, "r") as f:
        filenames = [line.strip() for line in f.readlines()]
    return embeddings, filenames

# Initialize FAISS index
def initialize_faiss_index(embeddings, index_path):
    dimension = embeddings.shape[1]
    if os.path.exists(index_path):
        logger.info(f"[INFO] Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
    else:
        logger.info(f"[INFO] Creating new FAISS index...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
    return index

def load_faiss_index_for_prefix(prefix, embeddings):
    if prefix == "0_":
        index_path = "/Users/fah/Downloads/senior_1/training_dataset/faiss_index_prefix0.bin"
    elif prefix == "1_":
        index_path = "/Users/fah/Downloads/senior_1/training_dataset/faiss_index_prefix1.bin"
    else:
        raise ValueError(f"Invalid prefix: {prefix}")

    dimension = embeddings.shape[1]
    if os.path.exists(index_path):
        logger.info(f"[INFO] Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
    else:
        logger.info(f"[INFO] Creating new FAISS index for prefix {prefix}...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
    return index


# Compute embedding for a given source code
def compute_code_embedding(source_code):
    tokenized_ids = unixcoder_model.tokenize([source_code], mode="<encoder-only>", padding=True)
    tokenized_ids_tensor = torch.tensor(tokenized_ids).to(device)
    with torch.no_grad():
        _, sentence_embeddings = unixcoder_model(tokenized_ids_tensor)
    return sentence_embeddings.mean(dim=0).cpu().numpy()

def localneighborhood(source_code, prefix, k, ratio):
    try:
        # 1) Load embeddings & filenames
        embeddings, filenames = load_precomputed_data(prefix)

        # select directory
        if prefix == '0_':
            directory = "/Users/fah/Downloads/senior_1/training_dataset/prefix0"
        elif prefix == '1_':
            directory = "/Users/fah/Downloads/senior_1/training_dataset/prefix1"
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

        # 2) Load or build FAISS
        index = load_faiss_index_for_prefix(prefix, embeddings)

        # 3) Compute source embedding
        source_embedding = compute_code_embedding(source_code).astype(np.float32).reshape(1, -1)

        # 4) Query FAISS for all distances
        distances, indices = index.search(source_embedding, len(embeddings))
        results = [
            (filenames[i],
             read_file(os.path.join(directory, filenames[i])),
             float(distances[0][j]))
            for j, i in enumerate(indices[0])
        ]

        # 5) Threshold at 90th percentile
        thr = np.percentile([r[2] for r in results], 90)
        filtered = [r for r in results if r[2] <= thr]

        # if threshold leaves too few, revert to full set
        if len(filtered) < k:
            filtered = results

        # 6) Validate ratio
        if sum(ratio) != 100:
            ratio = [70, 20, 10]

        # 7) Compute initial splits
        num_nearest  = int(ratio[0] * k / 100)
        num_middle   = int(ratio[1] * k / 100)
        num_farthest = int(ratio[2] * k / 100)

        # 8) Ensure total ≤ k by decrementing farthest → middle → nearest
        while (num_nearest + num_middle + num_farthest) > k:
            if num_farthest > 0:
                num_farthest -= 1
            elif num_middle > 0:
                num_middle -= 1
            else:
                num_nearest -= 1

        # 9) Partition
        filtered.sort(key=lambda x: x[2])
        nearest_list  = filtered[:num_nearest]
        middle_list   = filtered[num_nearest:num_nearest + num_middle]
        farthest_list = filtered[num_nearest + num_middle:]

        # 10) Sample each
        nearest_sample  = random.sample(nearest_list, min(len(nearest_list), num_nearest))
        middle_sample   = random.sample(middle_list,  min(len(middle_list),  num_middle))
        farthest_sample = random.sample(farthest_list,min(len(farthest_list),num_farthest))

        sample = nearest_sample + middle_sample + farthest_sample

        # 11) Fill up to k if needed
        if len(sample) < k:
            remaining = k - len(sample)
            # choose from the thresholded set (filtered)
            sample += random.sample(filtered, min(remaining, len(filtered)))

        # 12) Sort & return
        sample.sort(key=lambda x: x[2])
        return sample

    except Exception as e:
        logger.error(f"[ERROR] localneighborhood failed: {e}")
        return []



def gen_explanation(source_code, probability_chatgpt, probability_human, local_first, local_second, model_choice):
    logger.debug(f"Generating explanation for model: {model_choice}")

    logger.debug(f"Neighborhoods - First: {local_first}, Second: {local_second}")
    if not all(len(entry) == 3 for entry in local_first + local_second):
        logger.error("[ERROR] Neighborhood data format is invalid. Expected tuples with three elements.")
        raise ValueError("Invalid neighborhood data format.")
    
    logger.debug(f"Code: {source_code[:20]}..., Probabilities: {probability_chatgpt}, {probability_human}")

    probability_chatgpt = round(probability_chatgpt, 4)
    probability_human = round(probability_human, 4)

    if probability_chatgpt > probability_human:
        more_likely_classification = "ChatGPT-generated"
        higher_probability = probability_chatgpt
        classification_samples = "ChatGPT-generated examples"
    else:
        more_likely_classification = "Human-written"
        higher_probability = probability_human
        classification_samples = "human-written examples"

    explanation_template = """
    **Context**:
    We use a machine learning model to determine whether ChatGPT has generated a piece of source code. The classification engine is based on GraphCodeBERT. The model we used is part of the GPTSniffer (https://github.com/MDEGroup/GPTSniffer).
    The model yields the following probability scores, marked as **Classification Results**, for the target source code, marked as **Target Source Code**, below. Moreover, along with the probability scores for the target source code, you are provided with some samples of the ChatGPT-generated and human-written code (marked as **Examples**) that were classified by the same model. 

    **Examples**:
    - ChatGPT-generated samples:
        {related_examples_chatGPT}

    - Human-written samples:
        {related_examples_human}

    **Target Source Code**:
    {source_code}

    **Classification Results**:
        - Probability of being ChatGPT-generated: {probability_chatgpt}%.
        - Probability of being Human-written: {probability_human}%.

    **Instruction**:
    ญlease provide a comprehensive explanation and reasoning why the model predicts such code as human-written or AI-generated containing the following sections:

    **Answer structure**:
    
    **Overview**
    - Briefly summarize the most influential features or code excerpts that led to the classification as **{more_likely_classification}**. This must not exceed 20 lines.
    
    End the overview with a markdown horizontal line (e.g., `---`).

    **Highlight and Explain Key Code Lines**:
    - Key Code Lines: Extract five code lines or statements from the target source code that demonstrate {more_likely_classification} characteristics.
        - Explain Each Code Line:
            - Describe why the code lines indicates {more_likely_classification} patterns.
            - Detail the features or patterns present in the code that lead to {more_likely_classification}.
            

    **Comparison to Previous Examples**:
    - Analyze how the features in the target source code align with or differ from the given code examples and their classifications.
    - Compare these features with similar patterns found in the dataset examples. For each comparison, provide the corresponding code snippet from the example alongside the target code excerpt. This side-by-side presentation will help readers understand the model's classification better.
    - When referring to a specific example (e.g., "Example 1"), include its relevant code snippet for a direct visual comparison with the target source code.
    - Discuss how these similarities or differences support the classification decision.

    Your final explanation should be detailed, structured and is in a markdown format. 
    """

    related_examples_chatGPT = ""
    for idx, (filename, neighbor_code, distance) in enumerate(local_first):
        related_examples_chatGPT += f"Example {idx + 1} (Filename: {filename}, Distance: {distance}):\n```java\n{neighbor_code}\n```\n\n"

    related_examples_human = ""
    for idx, (filename, neighbor_code, distance) in enumerate(local_second):
        related_examples_human += f"Example {idx + 1} (Filename: {filename}, Distance: {distance}):\n```java\n{neighbor_code}\n```\n\n"

    explanation_prompt = explanation_template.format(
        source_code=source_code,
        probability_chatgpt=f"{probability_chatgpt:.4f}",
        probability_human=f"{probability_human:.4f}",
        more_likely_classification=more_likely_classification,
        higher_probability=f"{higher_probability:.4f}",
        classification_samples=classification_samples,
        related_examples_chatGPT=related_examples_chatGPT,
        related_examples_human=related_examples_human
    )

    # Model API selection logic
    if model_choice == 'ChatGPT':
        retries = 5
        for attempt in range(retries):
            try:
                explanation_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant to help me analyze the code base on the given prompt."},
                        {"role": "user", "content": explanation_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.5,
                )
                explanation = explanation_response.choices[0].message.content
                break  # Exit loop if successful
            except openai.error.RateLimitError:

                logger.warning(f"Rate limit hit. Retrying ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)  # Exponential backoff
            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise e
        else:
            logger.error("Failed to generate explanation after multiple attempts due to rate limiting.")
            raise Exception("Rate limit exceeded. Please try again later.")


    elif model_choice == 'Gemini':
        logger.debug("Calling Gemini API...")
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            explanation_response = model.generate_content(explanation_prompt)
            logger.debug(f"Gemini Response: {explanation_response}")
            explanation = explanation_response.text
        except Exception as e:
            if "API key expired" in str(e):
                logger.error("Gemini API key expired. Please update your GEMINI_API_KEY environment variable with a valid key.")
                raise Exception("Gemini API key expired. Please renew your API key.")
            else:
                logger.error(f"[ERROR] Gemini API failed: {e}")
                raise e
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}. Please choose 'ChatGPT' or 'Gemini'.")

    return explanation

Chat_code2_path = '/Users/fah/Downloads/senior_1/Chat_code2.java'
Chat_code2 = read_file(Chat_code2_path)

probability_chatgpt, probability_human = classify_code(Chat_code2)

ratio = [70,20,10]
local_random_first_prefix_threshold_random_scale = localneighborhood(Chat_code2, '0_', 4, ratio)
for filename, neighbor_code, distance in local_random_first_prefix_threshold_random_scale:
    print(f"Filename: {filename}\nDistance: {distance}\n")
