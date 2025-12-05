"""Explainability system for decision reasoning."""

import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from decision_engine.schemas import (
    DecisionRecommendation,
    DecisionOutput,
    DecisionContext,
    DecisionReasoning,
    DecisionAction,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionExplainer:
    """Generate human-readable explanations for decisions."""
    
    @staticmethod
    def generate_reasoning(
        context: DecisionContext,
        action: DecisionAction,
        factors: Dict[str, Any],
        rules: List[str],
        models: Dict[str, float]
    ) -> DecisionReasoning:
        """
        Generate decision reasoning.
        
        Args:
            context: Decision context
            action: Recommended action
            factors: Factor dict
            rules: Rule triggers
            models: Model contributions
            
        Returns:
            DecisionReasoning
        """
        # Build primary factors
        primary_factors = []
        for key, value in list(factors.items())[:5]:
            if isinstance(value, float):
                primary_factors.append(f"{key}: {value:.2f}")
            else:
                primary_factors.append(f"{key}: {value}")
        
        # Generate risk assessment
        risk_assessment = DecisionExplainer._generate_risk_assessment(context, action)
        
        # Generate opportunity assessment
        opportunity_assessment = DecisionExplainer._generate_opportunity_assessment(context, action)
        
        return DecisionReasoning(
            primary_factors=primary_factors,
            rule_triggers=rules,
            model_contributions=models,
            risk_assessment=risk_assessment,
            opportunity_assessment=opportunity_assessment,
        )
    
    @staticmethod
    def _generate_risk_assessment(context: DecisionContext, action: DecisionAction) -> str:
        """Generate risk assessment text."""
        risks = []
        
        if context.current_position <= 3:
            risks.append("Podium position at stake")
        
        if context.safety_car_active:
            risks.append("SC active (unpredictable)")
        
        if context.tire_age > 25:
            risks.append("High tire age")
        
        if action == DecisionAction.PIT_NOW:
            risks.append("Pit stop time loss")
        
        if not risks:
            return "Low risk: Favorable conditions"
        elif len(risks) <= 2:
            return f"Medium risk: {', '.join(risks)}"
        else:
            return f"High risk: {', '.join(risks)}"
    
    @staticmethod
    def _generate_opportunity_assessment(context: DecisionContext, action: DecisionAction) -> str:
        """Generate opportunity assessment text."""
        opportunities = []
        
        if action == DecisionAction.UNDERCUT_NOW:
            opportunities.append("Undercut potential")
        
        if action == DecisionAction.PIT_UNDER_SC:
            opportunities.append("Cheap pit stop under SC")
        
        if context.tire_age < 10:
            opportunities.append("Fresh tires advantage")
        
        if not opportunities:
            return "Limited opportunity"
        else:
            return f"Opportunity: {', '.join(opportunities)}"
    
    @staticmethod
    def generate_explanation_text(recommendation: DecisionRecommendation) -> str:
        """
        Generate human-readable explanation.
        
        Args:
            recommendation: Decision recommendation
            
        Returns:
            Explanation text
        """
        action_text = recommendation.action.value.replace('_', ' ').title()
        confidence_pct = int(recommendation.confidence_score * 100)
        traffic_light = recommendation.traffic_light.value.upper()
        
        explanation = (
            f"Recommend: {action_text} "
            f"(Confidence: {confidence_pct}%, {traffic_light}). "
        )
        
        if recommendation.reasoning.primary_factors:
            factors_text = "; ".join(recommendation.reasoning.primary_factors[:3])
            explanation += f"Factors: {factors_text}. "
        
        explanation += (
            f"Expected gain: {recommendation.expected_gain_seconds:.1f}s. "
            f"{recommendation.reasoning.risk_assessment}"
        )
        
        if recommendation.alternatives:
            alt = recommendation.alternatives[0]
            explanation += (
                f" Alternative: {alt.action.value.replace('_', ' ').title()} "
                f"({int(alt.confidence * 100)}%) - {alt.expected_outcome}"
            )
        
        return explanation
    
    @staticmethod
    def generate_comparison_table(recommendations: List[DecisionRecommendation]) -> str:
        """
        Generate markdown comparison table.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Markdown table string
        """
        if not recommendations:
            return "No recommendations"
        
        lines = [
            "| Action | Confidence | Traffic Light | Expected Gain | Risk | Priority |",
            "|--------|-----------|---------------|---------------|------|----------|",
        ]
        
        for rec in recommendations:
            action = rec.action.value.replace('_', ' ').title()
            conf = f"{int(rec.confidence_score * 100)}%"
            light = rec.traffic_light.value.upper()
            gain = f"{rec.expected_gain_seconds:+.1f}s"
            risk = f"{int(rec.risk_score * 100)}%"
            priority = str(rec.priority)
            
            lines.append(f"| {action} | {conf} | {light} | {gain} | {risk} | {priority} |")
        
        return "\n".join(lines)


class DecisionLogger:
    """Log decisions for audit and analysis."""
    
    def log_decision(self, decision_output: DecisionOutput, context: DecisionContext):
        """
        Log decision with structured fields.
        
        Args:
            decision_output: Decision output
            context: Decision context
        """
        log_data = {
            'session_id': decision_output.session_id,
            'lap_number': decision_output.lap_number,
            'driver_number': context.driver_number,
            'timestamp': decision_output.timestamp.isoformat(),
            'computation_time_ms': decision_output.computation_time_ms,
            'recommendation_count': len(decision_output.recommendations),
        }
        
        if decision_output.recommendations:
            top_rec = decision_output.recommendations[0]
            log_data['top_action'] = top_rec.action.value
            log_data['top_confidence'] = top_rec.confidence_score
            log_data['top_traffic_light'] = top_rec.traffic_light.value
            
            # Check for low-confidence critical decisions
            if top_rec.priority >= 9 and top_rec.confidence_score < 0.6:
                logger.warning(
                    f"Low-confidence critical decision: {top_rec.action.value} "
                    f"(confidence={top_rec.confidence_score:.2f})",
                    extra=log_data
                )
            else:
                logger.info(
                    f"Decision: {top_rec.action.value} "
                    f"(confidence={top_rec.confidence_score:.2f}, "
                    f"traffic_light={top_rec.traffic_light.value})",
                    extra=log_data
                )
        else:
            logger.info("No recommendations generated", extra=log_data)
    
    def log_decision_trace(
        self,
        module_name: str,
        recommendation: Any,
        latency_ms: float,
        error: Any = None
    ):
        """
        Log module execution trace.
        
        Args:
            module_name: Module name
            recommendation: Recommendation (or None)
            latency_ms: Execution time
            error: Error (if any)
        """
        log_data = {
            'module': module_name,
            'success': error is None,
            'latency_ms': latency_ms,
        }
        
        if recommendation:
            log_data['action'] = recommendation.action.value
            log_data['confidence'] = recommendation.confidence_score
        
        if error:
            log_data['error'] = str(error)
            logger.error(f"Module {module_name} failed", extra=log_data)
        else:
            logger.debug(f"Module {module_name} executed", extra=log_data)
    
    def export_decision_history(self, session_id: str, output_path: str):
        """
        Export decision history to JSON.
        
        Args:
            session_id: Session ID
            output_path: Output file path
        """
        # In production, would query from database
        # For now, placeholder implementation
        history = {
            'session_id': session_id,
            'exported_at': datetime.utcnow().isoformat(),
            'decisions': [],
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Exported decision history to {output_path}")


class DecisionAuditor:
    """Audit decision accuracy post-race."""
    
    @staticmethod
    def audit_decision(
        decision_output: DecisionOutput,
        actual_outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Audit decision against actual outcome.
        
        Args:
            decision_output: Decision output
            actual_outcome: Actual race outcome
            
        Returns:
            Audit report
        """
        if not decision_output.recommendations:
            return {
                'status': 'no_recommendation',
                'accuracy': 0.0,
            }
        
        top_rec = decision_output.recommendations[0]
        
        # Compare expected vs actual
        expected_gain = top_rec.expected_gain_seconds
        actual_gain = actual_outcome.get('time_gain', 0.0)
        
        # Calculate accuracy
        if expected_gain > 0:
            accuracy = min(1.0, actual_gain / expected_gain) if actual_gain > 0 else 0.0
        else:
            accuracy = 0.5
        
        # Determine outcome
        if actual_gain > expected_gain:
            outcome = "better_than_expected"
        elif actual_gain >= expected_gain * 0.8:
            outcome = "as_expected"
        else:
            outcome = "worse_than_expected"
        
        return {
            'recommendation': top_rec.action.value,
            'expected_gain': expected_gain,
            'actual_gain': actual_gain,
            'accuracy': accuracy,
            'outcome': outcome,
            'confidence': top_rec.confidence_score,
        }
