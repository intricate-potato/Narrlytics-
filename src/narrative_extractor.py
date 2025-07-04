
import requests
import json
from tqdm import tqdm
import os
import hashlib

class NarrativeExtractor:
    ACTANTIAL_PROMPT = """
    According to the Actantial Model by Greimas with the actant label set 
    ["Sender", "Receiver", "Subject", "Object", "Helper", "Opponent"], 
    the actants are defined as follows:
    * Subject: The character who carries out the action and desires the Object.
    * Object: The character or thing that is desired.
    * Sender: The character who initiates the action and communicates the Object.
    * Receiver: The character who receives the action or the Object.
    * Helper: The character who assists the Subject in achieving its goal.
    * Opponent: The character who opposes the Subject in achieving its goal.

    Based on this Actantial Model and the actant label set, please recognize 
    the actants in the given article.

    Article: {article_text}

    Question: What are the main actants in the text? Provide the answer in the 
    following JSON format: {{"Actant Label": ["Actant Name"]}}. If there is no 
    corresponding actant, return the following empty list: {{"Actant Label": []}}.

    Answer:
    """

    def __init__(self, config):
        self.config = config
        self.api_url = config['model']['ollama_api_url']
        self.model_name = config['model']['llm_model_name']
        self.cache_dir = os.path.join(config['data']['cache_path'], 'actants')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, article_id):
        # Use a hash of the article_id to create a safe filename
        return os.path.join(self.cache_dir, f"{hashlib.md5(article_id.encode()).hexdigest()}.json")

    def extract_actants(self, article_text):
        prompt = self.ACTANTIAL_PROMPT.format(article_text=article_text)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # The response from Ollama with format="json" is a single JSON object.
            # We need to parse the 'response' field which is a JSON string.
            ollama_response_content = response.json()['response']
            actants = json.loads(ollama_response_content)
            return self.post_process_actants(actants)

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            # Return a default empty structure on connection/request errors
            return {actant: [] for actant in ["Sender", "Receiver", "Subject", "Object", "Helper", "Opponent"]}
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from Ollama response: {ollama_response_content}")
            # Return a default empty structure on JSON decoding errors
            return {actant: [] for actant in ["Sender", "Receiver", "Subject", "Object", "Helper", "Opponent"]}

    def batch_extract_actants(self, articles):
        results = {}
        for article_id, article_text in tqdm(articles, desc="Extracting Actants with Caching"):
            cache_path = self._get_cache_path(article_id)
            
            if self.config['data']['cache_extracted_actants'] and os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    results[article_id] = json.load(f)
                continue

            extracted_actants = self.extract_actants(article_text)
            results[article_id] = extracted_actants
            
            if self.config['data']['cache_extracted_actants']:
                with open(cache_path, 'w') as f:
                    json.dump(extracted_actants, f)
                    
        return results

    def post_process_actants(self, actants):
        processed_actants = {}
        for actant_label in ["Sender", "Receiver", "Subject", "Object", "Helper", "Opponent"]:
            raw_actant_list = actants.get(actant_label, [])
            
            # Ensure raw_actant_list is actually a list, if not, treat as empty
            if not isinstance(raw_actant_list, list):
                raw_actant_list = []

            # Filter out non-string items and convert to a set to remove duplicates
            # Then convert back to a list
            cleaned_actant_names = []
            for item in raw_actant_list:
                if isinstance(item, str):
                    cleaned_actant_names.append(item)
                # else: Optionally log a warning if an unexpected type is found
                #     print(f"Warning: Unexpected actant type for {actant_label}: {type(item)}")

            processed_actants[actant_label] = list(set(cleaned_actant_names))
        return processed_actants
