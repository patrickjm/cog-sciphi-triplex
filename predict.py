import json
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cog import BasePredictor, Input
import os

class Predictor(BasePredictor):
    def setup(self):
        # Path to local files
        model_path = "./"

        # Load the model from local files
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        ).eval()

        # Load the tokenizer from local files
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

    def predict(
        self,
        text: str = Input(description="Input text to extract triplets from"),
        entity_types: List[str] = Input(description="List of entity types to extract", default=["LOCATION", "POSITION", "DATE", "CITY", "COUNTRY", "NUMBER"]),
        predicates: List[str] = Input(description="List of predicates to extract", default=["POPULATION", "AREA"])
    ) -> str:
        input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """

        message = input_format.format(
            entity_types=json.dumps({"entity_types": entity_types}),
            predicates=json.dumps({"predicates": predicates}),
            text=text
        )

        messages = [{'role': 'user', 'content': message}]
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        output = self.tokenizer.decode(self.model.generate(input_ids=input_ids, max_length=2048)[0], skip_special_tokens=True)
        return output