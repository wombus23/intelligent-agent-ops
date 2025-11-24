"""
Research Agent - Specialized in data gathering and research tasks
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """Agent specialized in research and data gathering"""
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for research agent"""
        return """You are a Research Agent specialized in gathering, analyzing, and synthesizing information.

Your responsibilities:
1. Conduct thorough research on given topics
2. Gather relevant data from multiple sources
3. Identify key facts, statistics, and insights
4. Organize information in a structured format
5. Cite sources and maintain accuracy

Guidelines:
- Be thorough and comprehensive in your research
- Focus on factual, verifiable information
- Provide clear citations and sources
- Organize findings logically
- Highlight key insights and patterns
- Flag any uncertainties or conflicting information

Output Format:
Structure your research findings with:
- Executive Summary
- Key Findings
- Detailed Analysis
- Sources and References
- Recommendations for further investigation"""
    
    async def _execute_core_logic(self, task_input: Dict[str, Any]) -> Any:
        """
        Execute research task
        
        Expected task_input format:
        {
            "query": "Research question or topic",
            "scope": "Scope of research",
            "depth": "shallow|medium|deep",
            "focus_areas": ["area1", "area2"]
        }
        """
        query = task_input.get('query', '')
        scope = task_input.get('scope', 'general')
        depth = task_input.get('depth', 'medium')
        focus_areas = task_input.get('focus_areas', [])
        
        # Build research prompt
        user_message = self._build_research_prompt(query, scope, depth, focus_areas)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Call LLM
        response_text, token_usage, cost = await self._call_llm(messages)
        
        # Track cost if tracker is available
        if self.cost_tracker:
            self.cost_tracker.record_usage(
                agent_name=self.config.name,
                agent_type=self.config.agent_type.value,
                model=self.config.model,
                task_id=task_input.get('task_id', 'unknown'),
                prompt_version=self.config.system_prompt_version,
                input_tokens=token_usage['input_tokens'],
                output_tokens=token_usage['output_tokens'],
                input_cost=cost * 0.3,  # Approximate split
                output_cost=cost * 0.7,
                latency_ms=0,  # Will be set by execute method
                success=True
            )
        
        return {
            'research_findings': response_text,
            'query': query,
            'scope': scope,
            'token_usage': token_usage,
            'cost': cost
        }
    
    def _build_research_prompt(
        self,
        query: str,
        scope: str,
        depth: str,
        focus_areas: list
    ) -> str:
        """Build detailed research prompt"""
        prompt = f"""Conduct research on the following topic:

QUERY: {query}

RESEARCH SCOPE: {scope}
DEPTH LEVEL: {depth}
"""
        
        if focus_areas:
            prompt += f"\nFOCUS AREAS:\n"
            for area in focus_areas:
                prompt += f"- {area}\n"
        
        prompt += """
Please provide:
1. Executive Summary: Brief overview of key findings
2. Detailed Findings: Comprehensive research results organized by topic
3. Key Insights: Important patterns, trends, or discoveries
4. Data and Statistics: Relevant quantitative information
5. Sources: References to information sources
6. Recommendations: Suggestions for further research or action

Ensure all information is accurate, well-organized, and properly cited.
"""
        
        return prompt
    
    async def research_topic(
        self,
        topic: str,
        focus_areas: list = None
    ) -> Dict[str, Any]:
        """
        Convenience method for researching a topic
        
        Args:
            topic: Topic to research
            focus_areas: Optional list of specific focus areas
        
        Returns:
            Research findings
        """
        task_input = {
            'query': topic,
            'scope': 'comprehensive',
            'depth': 'deep',
            'focus_areas': focus_areas or []
        }
        
        response = await self.execute(task_input, task_id=f"research_{topic[:20]}")
        return response.output if response.success else None
