"""
Base class for running baselines with LLM queries.
"""

import json
import logging
from typing import Optional, Dict, Any
from models.openrouter_client import OpenRouterClient

class BaselineRunner:
    """
    A simple baseline runner for multiple-choice QA with optional context.
    Uses OpenRouterClient to query the LLM.
    """
    
    def __init__(
        self,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        temperature: float = 0.3,
        max_tokens: int = 100,
        api_key: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize the baseline runner.
        
        Args:
            model: Model name from OpenRouter
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Max response length
            api_key: OpenRouter API key (uses env var if None)
            log_file: Optional file to log results
        """
        self.client = OpenRouterClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(handler)
        
    def _format_prompt(
        self,
        question: str,
        options: Dict[str, str],
        gold_passage,
        context: Optional[str] = None,
    ) -> str:
        """
        Format a prompt for multiple-choice QA.
        
        Args:
            question: The question to ask
            options: Dict of {option_letter: option_text}, e.g. {"A": "Paris", "B": "London"}
            context: Optional context/passage to include
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        prompt_parts.append(f"Question: {question}\n")
        prompt_parts.append(f"This is the gold passage and should help you choose the correct answer: {gold_passage}")

        prompt_parts.append("Options:")
        
        for letter, text in sorted(options.items()):
            prompt_parts.append(f"{letter}. {text}")

        prompt_parts.append(f"\nAnswer with the options: {list(options.keys())}:")
        return "\n".join(prompt_parts)
    
    def query(
        self,
        question: str,
        options: Dict[str, str],
        gold_passage,
        context: Optional[str] = None,
        passage_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query the LLM on a multiple-choice question.
        
        Args:
            question: The question to ask
            options: Dict of {option_letter: option_text}
            context: Optional context/passage
            passage_idx: Optional index for logging/tracking
            
        Returns:
            Dict with:
                - answer: The model's answer (A, B, C, or D)
                - raw_response: Full response from the model
                - passage_idx: The passage index (if provided)
        """
        prompt = self._format_prompt(question, options, gold_passage, context)
        
        try:
            response = self.client.query(prompt)
            # Extract just the letter (A, B, C, or D) from the response
            # Handle cases like "A", "['A']", "[A]", etc.
            answer = response.strip().upper()
            # Extract first uppercase letter that's A-D
            answer = next((c for c in answer if c in "ABCD"), "")
            
            result = {
                "answer": answer,
                "raw_response": response,
                "passage_idx": passage_idx,
            }
            
            self.logger.info(
                f"Query {passage_idx}: Question='{question[:50]}...', "
                f"Answer={answer}, Response='{response[:100]}...'"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            return {
                "answer": None,
                "raw_response": str(e),
                "passage_idx": passage_idx,
                "error": str(e),
            }
    
    def batch_query(
        self,
        records: list,
        question_key: str = "question",
        options_keys: dict = None,
        context_key: Optional[str] = None,
        idx_key: str = "idx",
    ) -> list:
        """
        Query multiple records in batch.
        
        Args:
            records: List of record dicts
            question_key: Key for the question field
            options_keys: Dict mapping {"A": "choice_a", "B": "choice_b", ...}
            context_key: Optional key for context field
            idx_key: Key for the index field
            
        Returns:
            List of result dicts
        """
        if options_keys is None:
            options_keys = {
                "A": "choice_a",
                "B": "choice_b",
                "C": "choice_c",
                "D": "choice_d",
            }
        
        results = []
        
        for i, record in enumerate(records):
            question = record.get(question_key, "")
            options = {
                letter: record.get(key, "")
                for letter, key in options_keys.items()
            }
            context = record.get(context_key) if context_key else None
            passage_idx = record.get(idx_key, i)
            
            result = self.query(
                question=question,
                options=options,
                context=context,
                passage_idx=passage_idx,
            )
            results.append(result)
        
        return results

