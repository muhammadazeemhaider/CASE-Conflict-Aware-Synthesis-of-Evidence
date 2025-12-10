import os
import sys
import json
import random
from collections import Counter
from typing import List, Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
from models.openrouter_client import OpenRouterClient

# --- legal personas ---
# user-defined personas for evaluating evidence
LEGAL_PERSONAS = {
    "The Strict Textualist": (
        "You are a Strict Textualist following the motto that the text is the law. "
        "Analyze provided evidence only. Do not use outside knowledge. "
        "If evidence does not explicitly state the answer, reject the option. "
        "Catch 'hallucinations' where the model invents rules not found in text."
    ),
    "The Devil's Advocate": (
        "You are a Devil's Advocate. Your goal is to find loopholes in the argument. "
        "Look for exceptions, loopholes, or missing conditions in evidence. "
        "Be highly skeptical. If an answer looks too simple, check for missing conditions."
    ),
    "The Equity Advocate": (
        "You are an Equity Advocate. You view law as a tool for fairness. "
        "In housing/tort cases, consider the vulnerable party for e.g., the tenant. "
        "Interpret ambiguities to prevent unjust outcomes for the vulnerable party."
    ),
    "The Legal Realist": (
        "You are a Legal Realist (Pragmatist). You care about practical consequences. "
        "If literal text leads to absurd results, reject it. "
        "Choose the option that represents a workable, sensible application of rules."
    ),
    "The Precedent Loyalist": (
        "You are a Precedent Loyalist. You care about consistency. "
        "Compare facts in 'Question' strictly against facts in 'Evidence' (Case Law). "
        "If facts don't match, the rule does not apply. Prevent false analogies."
    )
}

class ArbiterDecision(dspy.Signature):
    """
    You are a specialized legal agent with a specific persona.
    Evaluate the multiple choice question using only the retrieved evidence.
    """
    persona_description = dspy.InputField(desc="specific legal philosophy to adopt")
    question = dspy.InputField(desc="legal question to answer")
    options = dspy.InputField(desc="possible answers (A, B, C, D)")
    evidence = dspy.InputField(desc="retrieved legal passages for decision")
    
    reasoning = dspy.OutputField(desc="chain-of-thought. why your persona supports this vote")
    vote = dspy.OutputField(desc="best option letter (A, B, C, or D)")


# --- jury manager ---
class Jury:
    def __init__(self, client: OpenRouterClient = None):
        self.personas = LEGAL_PERSONAS
        
        # config dspy if not already done
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key and not dspy.settings.lm:
            lm = dspy.LM(
                model='openai/meta-llama/llama-3.3-70b-instruct:free',
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                max_tokens=2048
            )
            dspy.settings.configure(lm=lm)

    def synthesize(self, question: str, choices: Dict[str, str], evidence_list: List[Any], num_arbiters: int = 3) -> Dict[str, Any]:
        """
        conducts a trial where random agents review evidence and vote.
        accepts evidence as list of dicts (advocate) or list of json strings (dataset).
        """
        # for now, selects 3 members randomly from the jury personas.
        # formatting inputs
        evidence_text = "\n Evidence - "
        for i, ev in enumerate(evidence_list):
            content = ""

            if isinstance(ev, str):
                try:
                    ev_json = json.loads(ev)
                    content = ev_json.get('contents', str(ev))
                except json.JSONDecodeError:
                    content = ev

            elif isinstance(ev, dict):
                content = ev.get('contents', ev.get('text', str(ev)))
            
            evidence_text += f"[Document {i+1}] {content}\n"

        options_text = "\n".join([f"{k}: {v}" for k, v in choices.items()])

        # select diverse jury
        available_keys = list(self.personas.keys())
        k = min(num_arbiters, len(available_keys))
        selected_names = random.sample(available_keys, k)
        
        print(f"Jury selected: {selected_names}")

        votes = []
        logs = []
        
        # use chain of thought for better reasoning
        predictor = dspy.ChainOfThought(ArbiterDecision)

        # deliberation loop
        for name in selected_names:
            desc = self.personas[name]
            try:
                # agent thinks here
                pred = predictor(
                    persona_description=desc,
                    question=question,
                    options=options_text,
                    evidence=evidence_text
                )
                
                # clean output 
                raw_vote = pred.vote.strip().upper()
                clean_vote = None
                
                # heuristic: find first valid option letter
                for char in raw_vote:
                    if char in choices:
                        clean_vote = char
                        break
                
                if clean_vote:
                    votes.append(clean_vote)
                
                logs.append({
                    "persona": name,
                    "vote": clean_vote,
                    "reasoning": pred.reasoning
                })
                
            except Exception as e:
                print(f"Error with juror {name}: {e}")

        # final verdict (majority vote)
        if not votes:
            return {
                "final_verdict": None,
                "confidence": 0.0,
                "vote_breakdown": {},
                "juror_deliberations": logs
            }

        vote_counts = Counter(votes)
        winner, win_count = vote_counts.most_common(1)[0]
        confidence = win_count / len(votes)

        return {
            "final_verdict": winner,
            "confidence": round(confidence, 2),
            "vote_breakdown": dict(vote_counts),
            "juror_deliberations": logs
        }