import os
import sys
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.openrouter_client import OpenRouterClient


class HypothesisGenerator:
    """
    Generates hypotheses for answering legal questions.
    
    When use_generated_hypotheses=False (default), returns the MCQ options as hypotheses.
    When use_generated_hypotheses=True, generates actual hypotheses using an LLM.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the hypothesis generator.
        
        Args:
            llm_client: OpenRouterClient instance for generating hypotheses.
                       If None, will create one with default settings.
        """
        self.llm_client = OpenRouterClient()
    
    def generate(
        self,
        question: str,
        options: Dict[str, str],
        use_generated_hypotheses: bool = False,
        num_hypotheses: int = None
    ) -> List[str]:
        """
        Generate hypotheses for a given question.
        
        Args:
            question: The legal question to generate hypotheses for.
            options: Dictionary of MCQ options, e.g., {"A": "option text", "B": "option text"}.
            use_generated_hypotheses: If False (default), returns the options as hypotheses.
                                     If True, generates actual hypotheses using LLM.
            num_hypotheses: Number of hypotheses to generate when use_generated_hypotheses=True.
                          If None, defaults to the number of options.
        
        Returns:
            List of hypothesis strings.
        """
        if not use_generated_hypotheses:
            # Return options as hypotheses when options are available
            return [option_text for option_text in options.values()]
        
        # Generate hypotheses using LLM
        if num_hypotheses is None:
            num_hypotheses = len(options)
        
        prompt = self._build_hypothesis_prompt(question, num_hypotheses)
        
        try:
            response = self.llm_client.query(
                message=prompt,
                system_prompt="You are a legal expert generating hypotheses to answer legal questions. Generate hypotheses for the question that can be rooted in truth.",
                temperature=0.7,
                max_tokens=1024
            )
            
            # Parse the response to extract hypotheses
            hypotheses = self._parse_hypotheses(response, num_hypotheses)
            return hypotheses
            
        except Exception as e:
            print(f"Error generating hypotheses: {e}")
    
    def _build_hypothesis_prompt(self, question: str, num_hypotheses: int) -> str:
        """
        Build a prompt for generating hypotheses.
        
        Args:
            question: The legal question.
            options: Dictionary of MCQ options.
            num_hypotheses: Number of hypotheses to generate.
        
        Returns:
            Formatted prompt string.
        """
        prompt = f"""Given the following legal question, generate {num_hypotheses} distinct hypotheses that could help answer this question. Each hypothesis should be a testable statement that, if supported by evidence, would lead to selecting one of the options.

        Question: {question}

        Generate {num_hypotheses} hypotheses, one per line. Each hypothesis should be a clear, testable statement about the legal scenario.

        Hypotheses:
        """
        
        return prompt
    
    def _parse_hypotheses(self, response: str, num_hypotheses: int) -> List[str]:
        """
        Parse hypotheses from LLM response.
        
        Args:
            response: Raw response from LLM.
            num_hypotheses: Expected number of hypotheses.
        
        Returns:
            List of parsed hypothesis strings.
        """
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        hypotheses = []
        for line in lines:
            # Remove numbering (e.g., "1. ", "Hypothesis 1:", etc.)
            cleaned = line
            for prefix in ['-', '*', '•', '1.', '2.', '3.', '4.', 'Hypothesis', 'hypothesis']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            if cleaned:
                hypotheses.append(cleaned)
        
        while len(hypotheses) < num_hypotheses and hypotheses:
            hypotheses.append(hypotheses[-1])
        
        return hypotheses[:num_hypotheses]