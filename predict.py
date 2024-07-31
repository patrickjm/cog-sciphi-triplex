import json
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        self.model = AutoModelForCausalLM.from_pretrained("sciphi/triplex", trust_remote_code=True).to('cuda').eval()
        self.tokenizer = AutoTokenizer.from_pretrained("sciphi/triplex", trust_remote_code=True)

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
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        output = self.tokenizer.decode(self.model.generate(input_ids=input_ids, max_length=2048)[0], skip_special_tokens=True)
        return output