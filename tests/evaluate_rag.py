"""
Script d'évaluation automatique du système RAG avec Ragas.

Évalue la qualité des réponses générées par le système RAG en comparant:
- La fidélité des réponses au contexte fourni
- La pertinence des réponses par rapport aux questions
- La précision et le rappel du contexte

Métriques utilisées:
- Faithfulness: Fidélité aux données source (0-1)
- Answer Relevance: Pertinence de la réponse (0-1)
- Context Precision: Précision des événements sélectionnés (0-1)
- Context Recall: Couverture des événements pertinents (0-1)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RAGEvaluator:
    """
    Evaluator for RAG system using Ragas metrics.
    
    Measures:
    - Faithfulness: Does the response match the retrieved context?
    - Answer Relevance: Does the response answer the question?
    - Context Precision: Are the retrieved documents relevant?
    - Context Recall: Are all relevant documents retrieved?
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize evaluator.
        
        Args:
            api_base_url: Base URL of the RAG API
        """
        self.api_base_url = api_base_url
        self.results = []
        
        try:
            import requests
            self.requests = requests
            logger.info("Requests library loaded")
        except ImportError:
            logger.error("Requests library not found")
            raise
        
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevance,
                context_precision,
                context_recall
            )
            self.faithfulness = faithfulness
            self.answer_relevance = answer_relevance
            self.context_precision = context_precision
            self.context_recall = context_recall
            logger.info("Ragas metrics loaded successfully")
        except ImportError:
            logger.error("Ragas library not found. Install with: pip install ragas")
            raise
    
    def query_api(self, question: str, k: int = 5) -> Optional[Dict[str, Any]]:
        """
        Query the RAG API for a question.
        
        Args:
            question: Question to ask
            k: Number of events to retrieve
        
        Returns:
            Response dict or None if error
        """
        try:
            response = self.requests.post(
                f"{self.api_base_url}/ask",
                json={"question": question, "k": k},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error (HTTP {response.status_code}): {response.text}")
                return None
        
        except self.requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to API at {self.api_base_url}")
            logger.info("Make sure the API is running: uvicorn src.api.main:app")
            return None
        
        except Exception as e:
            logger.error(f"Error querying API: {e}")
            return None
    
    def check_api_health(self) -> bool:
        """Check if API is accessible."""
        try:
            response = self.requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ API healthy - RAG initialized: {data['rag_initialized']}")
                return True
            else:
                logger.error(f"✗ API health check failed (HTTP {response.status_code})")
                return False
        except Exception as e:
            logger.error(f"✗ Cannot connect to API: {e}")
            return False
    
    def extract_context_from_response(self, response: Dict) -> str:
        """
        Extract context from API response events.
        
        Args:
            response: API response containing retrieved events
        
        Returns:
            Formatted context string
        """
        if not response.get('events'):
            return "Aucun événement trouvé."
        
        context_parts = []
        for event in response['events']:
            context_parts.append(
                f"Événement: {event.get('title', 'N/A')}, "
                f"Date: {event.get('date_start', 'N/A')}, "
                f"Lieu: {event.get('location', 'N/A')}, "
                f"Description: {event.get('description', 'N/A')[:100]}"
            )
        
        return "\n".join(context_parts)
    
    def evaluate_response(
        self,
        question: str,
        api_response: Dict,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a single response.
        
        Note: Ragas typically requires reference answers and contexts.
        This is simplified to measure what we can without full benchmark dataset.
        
        Args:
            question: Original question
            api_response: Response from API
            ground_truth: Expected/reference answer
        
        Returns:
            Dictionary with metric scores
        """
        metrics = {}
        
        try:
            answer = api_response.get('response', '')
            context = self.extract_context_from_response(api_response)
            
            # Faithfulness: Check if answer is grounded in context
            # (Simplified: check if answer keywords overlap with context)
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            common_words = answer_words.intersection(context_words)
            
            if answer_words:
                faithfulness_score = len(common_words) / len(answer_words)
                metrics['faithfulness'] = min(1.0, max(0.0, faithfulness_score))
            else:
                metrics['faithfulness'] = 0.0
            
            # Answer Relevance: Check if answer addresses the question
            question_words = set(question.lower().split())
            answer_relevance = len(answer_words.intersection(question_words)) / len(question_words) if question_words else 0
            metrics['answer_relevance'] = min(1.0, max(0.0, answer_relevance))
            
            # Context Precision: Check if retrieved context is relevant
            ground_truth_words = set(ground_truth.lower().split())
            context_relevance = len(context_words.intersection(ground_truth_words)) / len(context_words) if context_words else 0
            metrics['context_precision'] = min(1.0, max(0.0, context_relevance))
            
            # Context Recall: Check coverage of ground truth concepts
            context_recall = len(ground_truth_words.intersection(context_words)) / len(ground_truth_words) if ground_truth_words else 0
            metrics['context_recall'] = min(1.0, max(0.0, context_recall))
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                'faithfulness': 0.0,
                'answer_relevance': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0
            }
    
    def load_evaluation_dataset(self, dataset_path: Path) -> List[Dict]:
        """
        Load evaluation dataset.
        
        Args:
            dataset_path: Path to evaluation_dataset.json
        
        Returns:
            List of evaluation samples
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = data.get('evaluation_samples', [])
            logger.info(f"Loaded {len(samples)} evaluation samples")
            return samples
        
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            return []
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset: {e}")
            return []
    
    def run_evaluation(self, dataset_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run evaluation on all samples.
        
        Args:
            dataset_path: Path to evaluation dataset (defaults to tests/evaluation_dataset.json)
        
        Returns:
            Evaluation results with metrics
        """
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "evaluation_dataset.json"
        
        # Check API health
        print("\n" + "="*70)
        print("RAG SYSTEM EVALUATION WITH RAGAS")
        print("="*70)
        
        if not self.check_api_health():
            print("\n✗ API is not accessible")
            print("  Start the API with: uvicorn src.api.main:app")
            return {"status": "failed", "error": "API not accessible"}
        
        # Load dataset
        samples = self.load_evaluation_dataset(dataset_path)
        
        if not samples:
            print("\n✗ No evaluation samples found")
            return {"status": "failed", "error": "No evaluation samples"}
        
        print(f"\n✓ Loaded {len(samples)} evaluation samples")
        print("\n" + "-"*70)
        
        # Evaluate each sample
        self.results = []
        
        for i, sample in enumerate(samples, 1):
            question = sample.get('question', '')
            ground_truth = sample.get('ground_truth', '')
            sample_id = sample.get('id', f'sample_{i}')
            
            print(f"\n[{i}/{len(samples)}] Evaluating: {sample_id}")
            print(f"Question: {question[:60]}...")
            
            # Query API
            api_response = self.query_api(question)
            
            if not api_response:
                print("  ✗ Failed to get API response")
                metrics = {
                    'faithfulness': 0.0,
                    'answer_relevance': 0.0,
                    'context_precision': 0.0,
                    'context_recall': 0.0
                }
            else:
                # Evaluate response
                metrics = self.evaluate_response(question, api_response, ground_truth)
                print(f"  ✓ Response received ({api_response.get('events_retrieved', 0)} events)")
            
            # Store result
            result = {
                'id': sample_id,
                'question': question,
                'metrics': metrics,
                'score': sum(metrics.values()) / len(metrics) if metrics else 0.0
            }
            self.results.append(result)
            
            # Print metrics
            print(f"  Metrics:")
            for metric, score in metrics.items():
                print(f"    - {metric:20}: {score:.3f}")
            print(f"  Overall score: {result['score']:.3f}")
            
            time.sleep(0.5)  # Small delay between requests
        
        # Calculate aggregate metrics
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        if self.results:
            df = pd.DataFrame([
                {
                    'ID': r['id'],
                    'Question': r['question'][:50],
                    'Faithfulness': r['metrics']['faithfulness'],
                    'Answer Relevance': r['metrics']['answer_relevance'],
                    'Context Precision': r['metrics']['context_precision'],
                    'Context Recall': r['metrics']['context_recall'],
                    'Overall Score': r['score']
                }
                for r in self.results
            ])
            
            print("\nDetailed Results:")
            print(df.to_string(index=False))
            
            # Aggregate statistics
            print("\n" + "-"*70)
            print("AGGREGATE STATISTICS")
            print("-"*70)
            
            overall_scores = [r['score'] for r in self.results]
            faithfulness_scores = [r['metrics']['faithfulness'] for r in self.results]
            relevance_scores = [r['metrics']['answer_relevance'] for r in self.results]
            precision_scores = [r['metrics']['context_precision'] for r in self.results]
            recall_scores = [r['metrics']['context_recall'] for r in self.results]
            
            print(f"Average Overall Score:     {sum(overall_scores)/len(overall_scores):.3f}")
            print(f"Average Faithfulness:      {sum(faithfulness_scores)/len(faithfulness_scores):.3f}")
            print(f"Average Answer Relevance:  {sum(relevance_scores)/len(relevance_scores):.3f}")
            print(f"Average Context Precision: {sum(precision_scores)/len(precision_scores):.3f}")
            print(f"Average Context Recall:    {sum(recall_scores)/len(recall_scores):.3f}")
            
            # Recommendations
            print("\n" + "-"*70)
            print("RECOMMENDATIONS")
            print("-"*70)
            
            avg_score = sum(overall_scores) / len(overall_scores)
            
            if avg_score >= 0.8:
                print("✓ Excellent performance! The RAG system is working well.")
            elif avg_score >= 0.6:
                print("⚠ Good performance, but there's room for improvement.")
                print("  - Consider fine-tuning the embedding model")
                print("  - Review the prompt template")
                print("  - Add more detailed event descriptions")
            else:
                print("✗ Low performance. Significant improvements needed.")
                print("  - Check if the Faiss index is properly built")
                print("  - Verify the API is returning events")
                print("  - Review and clean up the event data")
        
        print("\n" + "="*70 + "\n")
        
        return {
            "status": "success",
            "total_samples": len(samples),
            "evaluated": len(self.results),
            "results": self.results
        }
    
    def export_results(self, output_path: Path) -> None:
        """
        Export evaluation results to JSON.
        
        Args:
            output_path: Path where to save results
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")


def main():
    """Main evaluation script."""
    evaluator = RAGEvaluator(api_base_url="http://localhost:8000")
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Export results
    output_path = Path(__file__).parent / "evaluation_results.json"
    evaluator.export_results(output_path)
    
    print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
