# Intelligent Agent Ops (IAOps)

**A comprehensive LLMOps platform for multi-agent orchestration, prompt versioning, and intelligent monitoring with cost optimization.**

IAOps is an end-to-end framework designed for production LLM applications that require multi-agent collaboration, sophisticated prompt management, and detailed observability. Unlike traditional RAG evaluation frameworks, IAOps focuses on agent orchestration, prompt lifecycle management, and real-time cost tracking.

## 🎯 Key Features

- 🤖 **Multi-Agent Orchestration**: Coordinate multiple specialized AI agents with role-based task distribution
- 📝 **Prompt Version Control**: Git-like versioning system for prompts with A/B testing capabilities
- 💰 **Cost Tracking & Optimization**: Real-time token usage monitoring and cost analysis per agent/prompt
- 🔍 **LangSmith Integration**: Advanced tracing and debugging for complex agent workflows
- 📊 **Performance Analytics**: Latency tracking, success rates, and quality metrics dashboard
- 🔄 **Fallback Strategies**: Automatic retry logic with model degradation (GPT-4 → GPT-3.5)
- ⚡ **Async Processing**: High-throughput agent coordination with concurrent execution
- 🎨 **Interactive UI**: Streamlit dashboard for monitoring and control

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Request                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Agent Orchestrator (Main Controller)            │
│  - Task decomposition                                        │
│  - Agent selection & routing                                 │
│  - Workflow management                                       │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │
    ┌────────▼────────┐        ┌───────▼────────┐
    │  Research Agent │        │  Analysis Agent │
    │  (Data Gather)  │        │  (Processing)   │
    └────────┬────────┘        └───────┬─────────┘
             │                          │
             └──────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Synthesis Agent│
              │  (Final Output) │
              └────────┬─────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Monitoring Layer                           │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │
│  │  LangSmith   │  │ Cost Tracker  │  │ Prompt Versioner │ │
│  │   Tracing    │  │   Analytics   │  │   Management     │ │
│  └──────────────┘  └───────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- LangSmith API key (optional but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/intelligent-agent-ops.git
cd intelligent-agent-ops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=intelligent-agent-ops
ENABLE_COST_TRACKING=true
ENABLE_PROMPT_VERSIONING=true
MAX_RETRIES=3
DEFAULT_MODEL=gpt-4o-mini
FALLBACK_MODEL=gpt-3.5-turbo
EOF
```

## 📁 Project Structure

```
intelligent-agent-ops/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── research_agent.py      # Research & data gathering
│   ├── analysis_agent.py      # Data analysis & processing
│   ├── synthesis_agent.py     # Final output generation
│   └── orchestrator.py        # Main orchestration logic
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── cost_tracker.py        # Token & cost monitoring
│   ├── prompt_manager.py      # Prompt versioning system
│   └── observability.py       # LangSmith integration
├── utils/
│   ├── __init__.py
│   ├── retry_logic.py         # Fallback strategies
│   ├── metrics.py             # Performance metrics
│   └── validators.py          # Output validation
├── ui/
│   ├── dashboard.py           # Streamlit dashboard
│   └── components/            # UI components
├── tests/
│   ├── test_agents.py
│   ├── test_orchestrator.py
│   ├── test_cost_tracker.py
│   └── test_prompt_manager.py
├── examples/
│   ├── basic_usage.py
│   ├── multi_agent_workflow.py
│   └── prompt_versioning_demo.py
├── prompts/
│   └── versions/              # Versioned prompt storage
├── data/
│   └── analytics/             # Analytics data storage
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
└── README.md
```

## 💡 Usage Examples

### Basic Multi-Agent Workflow

```python
from agents.orchestrator import AgentOrchestrator
from core.config import IAOpsConfig

# Initialize configuration
config = IAOpsConfig.from_env()

# Create orchestrator
orchestrator = AgentOrchestrator(config)

# Execute multi-agent task
result = await orchestrator.execute_task(
    task="Analyze the sentiment of tech company earnings calls from Q4 2024",
    agents=["research", "analysis", "synthesis"],
    context={"industry": "technology", "quarter": "Q4 2024"}
)

print(f"Final Result: {result.output}")
print(f"Total Cost: ${result.total_cost:.4f}")
print(f"Execution Time: {result.execution_time:.2f}s")
```

### Prompt Version Management

```python
from core.prompt_manager import PromptVersionManager

# Initialize prompt manager
pm = PromptVersionManager()

# Create new prompt version
pm.create_version(
    prompt_name="research_agent_system",
    content="You are a research assistant specializing in...",
    version="v1.2.0",
    description="Improved specificity in research instructions",
    tags=["production", "research"]
)

# A/B test prompts
results = await pm.ab_test(
    prompt_name="research_agent_system",
    versions=["v1.1.0", "v1.2.0"],
    test_cases=test_dataset,
    metrics=["accuracy", "latency", "cost"]
)

# Promote best version
pm.promote_to_production(
    prompt_name="research_agent_system",
    version="v1.2.0"
)
```

### Cost Tracking & Optimization

```python
from core.cost_tracker import CostTracker

# Initialize cost tracker
tracker = CostTracker()

# Get cost breakdown by agent
cost_report = tracker.get_agent_costs(
    start_date="2024-01-01",
    end_date="2024-01-31",
    group_by="agent"
)

# Get most expensive prompts
expensive_prompts = tracker.get_top_expensive_prompts(limit=10)

# Set cost alerts
tracker.set_alert(
    threshold=100.0,  # $100
    period="daily",
    notification_channel="email"
)
```

### Monitoring with LangSmith

```python
from core.observability import ObservabilityManager

# Initialize observability
obs = ObservabilityManager()

# Get execution traces
traces = obs.get_traces(
    agent="analysis_agent",
    start_time="2024-01-01T00:00:00Z",
    filters={"status": "error"}
)

# Analyze performance metrics
metrics = obs.analyze_performance(
    agent="synthesis_agent",
    metric="latency",
    percentile=95
)
```

## 🎛️ Configuration

### Agent Configuration

```python
from core.config import AgentConfig

research_config = AgentConfig(
    name="research_agent",
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=2000,
    timeout=30,
    retry_strategy={
        "max_retries": 3,
        "fallback_model": "gpt-3.5-turbo",
        "exponential_backoff": True
    }
)
```

### Orchestration Configuration

```python
from core.config import OrchestrationConfig

orchestration_config = OrchestrationConfig(
    max_concurrent_agents=3,
    task_timeout=120,
    enable_parallel_execution=True,
    result_aggregation_strategy="weighted_voting"
)
```

## 📊 Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run ui/dashboard.py
```

Features:
- Real-time agent activity monitoring
- Cost analytics and trends
- Prompt version comparison
- Execution trace visualization
- Performance metrics
- Alert management

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test suite
pytest tests/test_orchestrator.py -v

# Run integration tests
pytest tests/integration/ -v
```

## 🐳 Docker Deployment

```bash
# Build image
docker-compose build

# Run services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📈 Metrics & Analytics

### Agent Performance Metrics

- **Success Rate**: Percentage of successful task completions
- **Average Latency**: Mean execution time per agent
- **Token Efficiency**: Tokens per task / output quality ratio
- **Error Rate**: Frequency of failures and retries

### Cost Metrics

- **Cost per Agent**: Total spend broken down by agent type
- **Cost per Task**: Average cost for different task categories
- **Token Usage**: Input/output token distribution
- **Model Distribution**: Usage patterns across different models

### Prompt Metrics

- **Version Performance**: Quality metrics per prompt version
- **A/B Test Results**: Statistical significance of version differences
- **Adoption Rate**: Speed of production rollout
- **Rollback Frequency**: Stability indicator

## 🔧 Advanced Features

### Custom Agent Creation

```python
from agents.base_agent import BaseAgent

class CustomValidationAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.agent_type = "validation"
    
    async def execute(self, task_input):
        # Custom validation logic
        validated_output = await self.validate_content(task_input)
        return validated_output
    
    async def validate_content(self, content):
        # Implementation
        pass
```

### Workflow Templates

```python
from agents.orchestrator import WorkflowTemplate

# Define custom workflow
custom_workflow = WorkflowTemplate(
    name="content_generation_workflow",
    stages=[
        {"agent": "research", "parallel": False},
        {"agent": "analysis", "parallel": False},
        {"agent": "synthesis", "parallel": False},
        {"agent": "validation", "parallel": False}
    ],
    error_handling="continue",  # or "stop"
    result_caching=True
)

# Execute workflow
result = await orchestrator.execute_workflow(
    workflow=custom_workflow,
    input_data=task_data
)
```

## 🔐 Security & Best Practices

- Store API keys in environment variables
- Use role-based access control for prompt management
- Implement rate limiting for cost control
- Enable audit logging for compliance
- Validate all agent outputs before use
- Implement content filtering for safety

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [LangSmith](https://smith.langchain.com/) - Observability platform
- [OpenAI](https://openai.com/) - LLM provider
- [Streamlit](https://streamlit.io/) - Dashboard framework



## 🗺️ Roadmap

- [ ] Support for additional LLM providers (Anthropic, Cohere, Mistral)
- [ ] Advanced agent memory systems
- [ ] Graph-based workflow orchestration
- [ ] Real-time streaming support
- [ ] Custom evaluation metrics framework
- [ ] Enterprise SSO integration
- [ ] Multi-tenancy support
- [ ] Vector store integration for context management

