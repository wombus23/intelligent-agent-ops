"""
Base agent class for all specialized agents
"""
import asyncio
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import openai
from groq import AsyncGroq
from core.config import AgentConfig
from core.cost_tracker import CostTracker
from core.prompt_manager import PromptVersionManager


@dataclass
class AgentResponse:
    """Response from an agent execution"""
    agent_name: str
    agent_type: str
    output: Any
    success: bool
    error: Optional[str]
    latency_ms: float
    token_usage: Dict[str, int]
    cost: float
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(
        self,
        config: AgentConfig,
        cost_tracker: Optional[CostTracker] = None,
        prompt_manager: Optional[PromptVersionManager] = None,
        api_keys: Optional[Dict[str, str]] = None
    ):
        self.config = config
        self.cost_tracker = cost_tracker
        self.prompt_manager = prompt_manager
        
        # Initialize client based on provider
        self.provider = config.provider.lower()
        api_keys = api_keys or {}
        
        if self.provider == "groq":
            groq_api_key = api_keys.get("groq") or os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is required for Groq provider")
            self.client = AsyncGroq(api_key=groq_api_key)
        elif self.provider == "openai":
            openai_api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            self.client = openai.AsyncOpenAI(api_key=openai_api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from prompt manager or use default"""
        if self.prompt_manager:
            prompt_version = self.prompt_manager.get_version(
                f"{self.config.name}_system",
                self.config.system_prompt_version
            )
            if prompt_version:
                return prompt_version.content
        
        # Default system prompt
        return self._get_default_system_prompt()
    
    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for this agent type"""
        pass
    
    @abstractmethod
    async def _execute_core_logic(self, task_input: Dict[str, Any]) -> Any:
        """
        Core execution logic for the agent
        Must be implemented by subclasses
        """
        pass
    
    async def execute(
        self,
        task_input: Dict[str, Any],
        task_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Execute the agent's task with monitoring and error handling
        
        Args:
            task_input: Input data for the task
            task_id: Unique identifier for the task
            context: Additional context information
        
        Returns:
            AgentResponse with results and metadata
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Execute core logic
            output = await self._execute_core_logic(task_input)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Create successful response
            response = AgentResponse(
                agent_name=self.config.name,
                agent_type=self.config.agent_type.value,
                output=output,
                success=True,
                error=None,
                latency_ms=latency_ms,
                token_usage=context.get('token_usage', {}),
                cost=context.get('cost', 0.0),
                metadata={
                    'task_id': task_id,
                    'model': self.config.model,
                    'temperature': self.config.temperature,
                    **context
                }
            )
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Create error response
            response = AgentResponse(
                agent_name=self.config.name,
                agent_type=self.config.agent_type.value,
                output=None,
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                token_usage={},
                cost=0.0,
                metadata={
                    'task_id': task_id,
                    'error_type': type(e).__name__,
                    **context
                }
            )
            
            return response
    
    async def _call_llm(
        self,
        messages: list[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> tuple[str, Dict[str, int], float]:
        """
        Call LLM with retry logic and cost tracking
        
        Returns:
            tuple: (response_text, token_usage, cost)
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        retry_count = 0
        max_retries = self.config.retry_strategy.max_retries
        current_model = self.config.model
        
        while retry_count <= max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.timeout
                )
                
                # Extract response
                response_text = response.choices[0].message.content
                
                # Extract token usage
                token_usage = {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                
                # Calculate cost
                cost = self._calculate_cost(current_model, token_usage)
                
                return response_text, token_usage, cost
                
            except openai.RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise
                
                # Exponential backoff
                if self.config.retry_strategy.exponential_backoff:
                    delay = min(
                        self.config.retry_strategy.base_delay * (2 ** retry_count),
                        self.config.retry_strategy.max_delay
                    )
                    await asyncio.sleep(delay)
                
            except openai.APIError as e:
                retry_count += 1
                if retry_count > max_retries:
                    # Try fallback model
                    if current_model != self.config.retry_strategy.fallback_model:
                        current_model = self.config.retry_strategy.fallback_model
                        retry_count = 0  # Reset retry count for fallback model
                    else:
                        raise
                
                await asyncio.sleep(self.config.retry_strategy.base_delay)
    
    def _calculate_cost(
        self,
        model: str,
        token_usage: Dict[str, int]
    ) -> float:
        """Calculate cost for token usage"""
        # Cost per 1K tokens (Current as of Nov 2024)
        cost_per_1k_input = {
            # OpenAI
            "gpt-4o": 0.0025,
            "gpt-4o-mini": 0.00015,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.0005,
            # Groq (Current models)
            "llama-3.3-70b-versatile": 0.00059,
            "llama-3.1-70b-versatile": 0.00059,
            "llama-3.1-8b-instant": 0.00005,
            "mixtral-8x7b-32768": 0.00024,
            "gemma2-9b-it": 0.00020,
        }
        
        cost_per_1k_output = {
            # OpenAI
            "gpt-4o": 0.01,
            "gpt-4o-mini": 0.0006,
            "gpt-4-turbo": 0.03,
            "gpt-3.5-turbo": 0.0015,
            # Groq (Current models)
            "llama-3.3-70b-versatile": 0.00079,
            "llama-3.1-70b-versatile": 0.00079,
            "llama-3.1-8b-instant": 0.00008,
            "mixtral-8x7b-32768": 0.00024,
            "gemma2-9b-it": 0.00020,
        }
        
        input_cost = (token_usage['input_tokens'] / 1000) * cost_per_1k_input.get(model, 0.001)
        output_cost = (token_usage['output_tokens'] / 1000) * cost_per_1k_output.get(model, 0.002)
        
        return input_cost + output_cost
    
    async def validate_output(self, output: Any) -> bool:
        """
        Validate agent output
        Can be overridden by subclasses
        """
        return output is not None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, model={self.config.model})"
