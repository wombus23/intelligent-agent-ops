"""
Basic usage example for Intelligent Agent Ops
Demonstrates multi-agent workflow with cost tracking and prompt versioning
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from core.config import IAOpsConfig, AgentConfig, AgentType
from core.cost_tracker import CostTracker
from core.prompt_manager import PromptVersionManager, PromptStatus
from agents.research_agent import ResearchAgent


async def main():
    """Main example execution"""
    
    print("=" * 80)
    print("Intelligent Agent Ops - Basic Usage Example")
    print("=" * 80)
    print()
    
    # 1. Initialize Configuration
    print("📋 Step 1: Loading Configuration...")
    config = IAOpsConfig.from_env()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return
    
    print("✅ Configuration loaded successfully")
    print(f"   - Default Model: {config.default_model}")
    print(f"   - Cost Tracking: {'Enabled' if config.cost_tracking.enabled else 'Disabled'}")
    print(f"   - Prompt Versioning: {'Enabled' if config.prompt_versioning.enabled else 'Disabled'}")
    print()
    
    # 2. Initialize Cost Tracker
    print("💰 Step 2: Initializing Cost Tracker...")
    cost_tracker = CostTracker()
    print("✅ Cost Tracker initialized")
    print()
    
    # 3. Initialize Prompt Manager
    print("📝 Step 3: Initializing Prompt Version Manager...")
    prompt_manager = PromptVersionManager()
    
    # Create a custom research prompt version
    try:
        custom_prompt = prompt_manager.create_version(
            prompt_name="research_agent_system",
            content="""You are an Advanced Research Agent with expertise in technology analysis.

Your mission is to provide comprehensive, data-driven research with a focus on:
- Emerging technologies and trends
- Market analysis and competitive landscape
- Technical feasibility and challenges
- Business implications and opportunities

Always structure your findings with clear headings, bullet points for key insights,
and quantitative data where available.""",
            version="v1.0.0",
            description="Initial version with technology focus",
            tags=["production", "research", "technology"],
            status=PromptStatus.PRODUCTION
        )
        
        # Promote to production
        prompt_manager.promote_to_production("research_agent_system", "v1.0.0")
        print("✅ Prompt version created and promoted to production")
        print(f"   - Version: {custom_prompt.version}")
        print(f"   - Checksum: {custom_prompt.checksum}")
    except ValueError as e:
        print(f"⚠️  Prompt already exists: {e}")
    print()
    
    # 4. Create Research Agent
    print("🤖 Step 4: Creating Research Agent...")
    agent_config = AgentConfig(
        name="tech_research_agent",
        agent_type=AgentType.RESEARCH,
        model=config.default_model,
        temperature=0.3,
        max_tokens=2000,
        system_prompt_version="v1.0.0"
    )
    
    research_agent = ResearchAgent(
        config=agent_config,
        cost_tracker=cost_tracker,
        prompt_manager=prompt_manager
    )
    print("✅ Research Agent created")
    print(f"   - Name: {agent_config.name}")
    print(f"   - Model: {agent_config.model}")
    print(f"   - Temperature: {agent_config.temperature}")
    print()
    
    # 5. Execute Research Task
    print("🔍 Step 5: Executing Research Task...")
    print("-" * 80)
    
    task_input = {
        'query': 'What are the latest developments in multimodal AI models?',
        'scope': 'technology industry',
        'depth': 'medium',
        'focus_areas': [
            'Recent model releases',
            'Key capabilities and benchmarks',
            'Applications and use cases',
            'Industry adoption trends'
        ],
        'task_id': 'demo_research_001'
    }
    
    print(f"Query: {task_input['query']}")
    print(f"Scope: {task_input['scope']}")
    print(f"Focus Areas: {', '.join(task_input['focus_areas'])}")
    print()
    
    # Execute the task
    response = await research_agent.execute(
        task_input=task_input,
        task_id='demo_research_001'
    )
    
    # 6. Display Results
    print("📊 Step 6: Results")
    print("-" * 80)
    
    if response.success:
        print("✅ Task completed successfully!")
        print()
        print("Research Findings:")
        print("-" * 80)
        print(response.output['research_findings'])
        print("-" * 80)
        print()
        
        # Display metrics
        print("📈 Performance Metrics:")
        print(f"   - Latency: {response.latency_ms:.2f}ms")
        print(f"   - Input Tokens: {response.token_usage.get('input_tokens', 0)}")
        print(f"   - Output Tokens: {response.token_usage.get('output_tokens', 0)}")
        print(f"   - Total Tokens: {response.token_usage.get('total_tokens', 0)}")
        print(f"   - Cost: ${response.cost:.6f}")
        print()
    else:
        print(f"❌ Task failed: {response.error}")
        print()
    
    # 7. Display Cost Summary
    print("💵 Step 7: Cost Summary")
    print("-" * 80)
    
    session_summary = cost_tracker.get_session_summary()
    print(f"Session Costs:")
    print(f"   - Total Cost: ${session_summary['total_cost']:.6f}")
    print(f"   - Total Requests: {session_summary['total_requests']}")
    print(f"   - Successful: {session_summary['successful_requests']}")
    print(f"   - Failed: {session_summary['failed_requests']}")
    print(f"   - Total Tokens: {session_summary['total_tokens']:,}")
    
    if session_summary['total_requests'] > 0:
        print(f"   - Avg Cost/Request: ${session_summary['avg_cost_per_request']:.6f}")
    print()
    
    # 8. Prompt Version Information
    print("📝 Step 8: Prompt Version Information")
    print("-" * 80)
    
    versions = prompt_manager.list_versions("research_agent_system")
    print(f"Available Versions: {len(versions)}")
    
    for version in versions:
        print(f"\n   Version: {version.version}")
        print(f"   Status: {version.status.value}")
        print(f"   Created: {version.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Description: {version.description}")
        print(f"   Tags: {', '.join(version.tags)}")
    print()
    
    # 9. Export Cost Data
    print("💾 Step 9: Exporting Cost Data...")
    
    try:
        cost_tracker.export_to_csv(
            "data/analytics/demo_cost_export.csv"
        )
        print("✅ Cost data exported to: data/analytics/demo_cost_export.csv")
    except Exception as e:
        print(f"⚠️  Export failed: {e}")
    print()
    
    print("=" * 80)
    print("Demo completed! 🎉")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Check the LangSmith dashboard for detailed traces")
    print("2. Run 'streamlit run ui/dashboard.py' to view the monitoring dashboard")
    print("3. Explore prompt A/B testing with prompt_manager.ab_test()")
    print("4. Create custom agents by extending BaseAgent")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
