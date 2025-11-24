"""
Configuration management for Intelligent Agent Ops
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


class ModelType(Enum):
    """Supported model types"""
    # OpenAI Models
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4_TURBO = "gpt-4-turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    
    # Groq Models (Current as of Nov 2024)
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_3_1_70B = "llama-3.1-70b-versatile"
    LLAMA_3_1_8B = "llama-3.1-8b-instant"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA2_9B = "gemma2-9b-it"
    
    # Claude Models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"


class LLMProvider(Enum):
    """LLM providers"""
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"


class AgentType(Enum):
    """Agent types in the system"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    CUSTOM = "custom"


@dataclass
class RetryStrategy:
    """Configuration for retry logic"""
    max_retries: int = 3
    fallback_model: str = "gpt-3.5-turbo"
    exponential_backoff: bool = True
    base_delay: float = 1.0
    max_delay: float = 60.0


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    agent_type: AgentType
    model: str = "llama-3.1-8b-instant"  # Default to Groq Llama 3.1 8B
    provider: str = "groq"  # Default to Groq
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    retry_strategy: Optional[RetryStrategy] = None
    system_prompt_version: str = "v1.0.0"
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.retry_strategy is None:
            self.retry_strategy = RetryStrategy()


@dataclass
class OrchestrationConfig:
    """Configuration for agent orchestration"""
    max_concurrent_agents: int = 3
    task_timeout: int = 120
    enable_parallel_execution: bool = True
    result_aggregation_strategy: str = "weighted_voting"
    enable_intermediate_results: bool = True
    workflow_cache_ttl: int = 3600  # seconds


@dataclass
class CostTrackingConfig:
    """Configuration for cost tracking"""
    enabled: bool = True
    cost_per_1k_input_tokens: Dict[str, float] = field(default_factory=lambda: {
        # OpenAI Models
        "gpt-4o": 0.0025,
        "gpt-4o-mini": 0.00015,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.0005,
        # Groq Models (Current pricing - Nov 2024)
        "llama-3.3-70b-versatile": 0.00059,
        "llama-3.1-70b-versatile": 0.00059,
        "llama-3.1-8b-instant": 0.00005,
        "mixtral-8x7b-32768": 0.00024,
        "gemma2-9b-it": 0.00020,
        # Claude Models
        "claude-3-5-sonnet-20241022": 0.003,
        "claude-3-opus-20240229": 0.015,
    })
    cost_per_1k_output_tokens: Dict[str, float] = field(default_factory=lambda: {
        # OpenAI Models
        "gpt-4o": 0.01,
        "gpt-4o-mini": 0.0006,
        "gpt-4-turbo": 0.03,
        "gpt-3.5-turbo": 0.0015,
        # Groq Models (Current pricing - Nov 2024)
        "llama-3.3-70b-versatile": 0.00079,
        "llama-3.1-70b-versatile": 0.00079,
        "llama-3.1-8b-instant": 0.00008,
        "mixtral-8x7b-32768": 0.00024,
        "gemma2-9b-it": 0.00020,
        # Claude Models
        "claude-3-5-sonnet-20241022": 0.015,
        "claude-3-opus-20240229": 0.075,
    })
    alert_threshold_daily: float = 100.0
    alert_threshold_monthly: float = 1000.0
    export_interval_hours: int = 24


@dataclass
class PromptVersioningConfig:
    """Configuration for prompt versioning"""
    enabled: bool = True
    storage_path: str = "prompts/versions"
    enable_ab_testing: bool = True
    auto_rollback_on_error: bool = True
    version_retention_days: int = 90


@dataclass
class ObservabilityConfig:
    """Configuration for observability and tracing"""
    langsmith_enabled: bool = True
    langsmith_project: str = "intelligent-agent-ops"
    log_level: str = "INFO"
    trace_sample_rate: float = 1.0
    enable_performance_metrics: bool = True
    metrics_export_interval: int = 60  # seconds


@dataclass
class IAOpsConfig:
    """Main configuration class for Intelligent Agent Ops"""
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    default_model: str = "llama3-8b-8192"
    default_provider: str = "groq"
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    cost_tracking: CostTrackingConfig = field(default_factory=CostTrackingConfig)
    prompt_versioning: PromptVersioningConfig = field(default_factory=PromptVersioningConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    agent_configs: Dict[str, AgentConfig] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "IAOpsConfig":
        """Create configuration from environment variables"""
        config = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
            default_model=os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant"),
            default_provider=os.getenv("DEFAULT_PROVIDER", "groq"),
        )
        
        # Orchestration config
        config.orchestration = OrchestrationConfig(
            max_concurrent_agents=int(os.getenv("MAX_CONCURRENT_AGENTS", "3")),
            task_timeout=int(os.getenv("TASK_TIMEOUT", "120")),
            enable_parallel_execution=os.getenv("ENABLE_PARALLEL_EXECUTION", "true").lower() == "true",
        )
        
        # Cost tracking config
        config.cost_tracking = CostTrackingConfig(
            enabled=os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true",
            alert_threshold_daily=float(os.getenv("COST_ALERT_DAILY", "100.0")),
            alert_threshold_monthly=float(os.getenv("COST_ALERT_MONTHLY", "1000.0")),
        )
        
        # Prompt versioning config
        config.prompt_versioning = PromptVersioningConfig(
            enabled=os.getenv("ENABLE_PROMPT_VERSIONING", "true").lower() == "true",
            storage_path=os.getenv("PROMPT_STORAGE_PATH", "prompts/versions"),
        )
        
        # Observability config
        config.observability = ObservabilityConfig(
            langsmith_enabled=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            langsmith_project=os.getenv("LANGCHAIN_PROJECT", "intelligent-agent-ops"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        
        # Default agent configs
        config.agent_configs = {
            "research": AgentConfig(
                name="research_agent",
                agent_type=AgentType.RESEARCH,
                model=config.default_model,
                provider=config.default_provider,
                temperature=0.3,
                max_tokens=2000,
            ),
            "analysis": AgentConfig(
                name="analysis_agent",
                agent_type=AgentType.ANALYSIS,
                model=config.default_model,
                provider=config.default_provider,
                temperature=0.5,
                max_tokens=1500,
            ),
            "synthesis": AgentConfig(
                name="synthesis_agent",
                agent_type=AgentType.SYNTHESIS,
                model=config.default_model,
                provider=config.default_provider,
                temperature=0.7,
                max_tokens=1000,
            ),
        }
        
        return config
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.agent_configs.get(agent_name)
    
    def add_agent_config(self, agent_name: str, config: AgentConfig):
        """Add or update agent configuration"""
        self.agent_configs[agent_name] = config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check that at least one API key is provided
        if not self.openai_api_key and not self.groq_api_key and not self.anthropic_api_key:
            errors.append("At least one API key is required (OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY)")
        
        if self.observability.langsmith_enabled and not self.langchain_api_key:
            errors.append("LangChain API key required when LangSmith is enabled")
        
        if self.cost_tracking.alert_threshold_daily > self.cost_tracking.alert_threshold_monthly:
            errors.append("Daily cost threshold cannot exceed monthly threshold")
        
        return errors


# Default configuration instance
default_config = IAOpsConfig.from_env()
