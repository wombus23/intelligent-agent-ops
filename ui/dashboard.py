"""
Intelligent Agent Ops - Streamlit Dashboard
Real-time monitoring and management interface
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import IAOpsConfig
from core.cost_tracker import CostTracker
from core.prompt_manager import PromptVersionManager, PromptStatus
from agents.research_agent import ResearchAgent


# Page configuration
st.set_page_config(
    page_title="Intelligent Agent Ops Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .error-text {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-text {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration"""
    return IAOpsConfig.from_env()


@st.cache_resource
def load_cost_tracker():
    """Load cost tracker"""
    return CostTracker()


@st.cache_resource
def load_prompt_manager():
    """Load prompt manager"""
    return PromptVersionManager()


def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<p class="main-header">🤖 Intelligent Agent Ops Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=IAOps", use_container_width=True)
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Overview", "💰 Cost Analytics", "📝 Prompt Manager", "🤖 Agent Executor", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # Load data
        cost_tracker = load_cost_tracker()
        summary = cost_tracker.get_cost_summary("today")
        
        st.metric("Today's Cost", f"${summary['total_cost']:.4f}")
        st.metric("Total Requests", summary['total_requests'])
        st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
        
        st.markdown("---")
        st.markdown("**Version:** 0.1.0")
        st.markdown("**Status:** 🟢 Active")
    
    # Main content based on selected page
    if page == "🏠 Overview":
        show_overview()
    elif page == "💰 Cost Analytics":
        show_cost_analytics()
    elif page == "📝 Prompt Manager":
        show_prompt_manager()
    elif page == "🤖 Agent Executor":
        show_agent_executor()
    elif page == "⚙️ Settings":
        show_settings()


def show_overview():
    """Overview dashboard"""
    st.header("📊 System Overview")
    
    cost_tracker = load_cost_tracker()
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        period = st.selectbox("Time Period", ["today", "week", "month"])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    with col3:
        if st.button("🔄 Refresh"):
            st.cache_resource.clear()
            st.rerun()
    
    if auto_refresh:
        st.markdown("*Dashboard will refresh in 30 seconds*")
    
    # Get summary data
    summary = cost_tracker.get_cost_summary(period)
    
    # Metrics row
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Cost",
            f"${summary['total_cost']:.4f}",
            delta=None,
            help="Total spending for selected period"
        )
    
    with col2:
        st.metric(
            "Total Requests",
            f"{summary['total_requests']:,}",
            help="Number of LLM API calls"
        )
    
    with col3:
        st.metric(
            "Avg Cost/Request",
            f"${summary['avg_cost_per_request']:.6f}" if summary['total_requests'] > 0 else "$0.00",
            help="Average cost per request"
        )
    
    with col4:
        success_rate = summary['success_rate']
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta=f"{success_rate - 95:.1f}%" if success_rate < 95 else "✓",
            help="Percentage of successful requests"
        )
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Cost by Agent")
        if summary['by_agent']:
            df_agents = pd.DataFrame(list(summary['by_agent'].items()), columns=['Agent', 'Cost'])
            df_agents = df_agents.sort_values('Cost', ascending=False)
            
            fig = px.bar(
                df_agents,
                x='Agent',
                y='Cost',
                title='Cost Distribution by Agent',
                color='Cost',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent cost data available yet")
    
    with col2:
        st.markdown("### 🤖 Cost by Model")
        if summary['by_model']:
            df_models = pd.DataFrame(list(summary['by_model'].items()), columns=['Model', 'Cost'])
            df_models = df_models.sort_values('Cost', ascending=False)
            
            fig = px.pie(
                df_models,
                values='Cost',
                names='Model',
                title='Cost Distribution by Model',
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model cost data available yet")
    
    # Recent activity
    st.markdown("---")
    st.markdown("### 📋 Recent Activity")
    
    df = cost_tracker.get_costs_by_period()
    if not df.empty:
        df_recent = df.tail(10).sort_values('timestamp', ascending=False)
        
        # Format for display
        display_df = df_recent[['timestamp', 'agent_name', 'model', 'total_cost', 'total_tokens', 'success']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['total_cost'] = display_df['total_cost'].apply(lambda x: f"${x:.6f}")
        display_df['success'] = display_df['success'].apply(lambda x: "✅" if x else "❌")
        
        display_df.columns = ['Timestamp', 'Agent', 'Model', 'Cost', 'Tokens', 'Status']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity to display")


def show_cost_analytics():
    """Cost analytics page"""
    st.header("💰 Cost Analytics")
    
    cost_tracker = load_cost_tracker()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Convert to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Get data
    df = cost_tracker.get_costs_by_period(start_datetime, end_datetime)
    
    if df.empty:
        st.warning("No cost data available for selected period")
        return
    
    # Summary metrics
    st.markdown("### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${df['total_cost'].sum():.4f}")
    with col2:
        st.metric("Total Tokens", f"{df['total_tokens'].sum():,.0f}")
    with col3:
        st.metric("Avg Latency", f"{df['latency_ms'].mean():.0f}ms")
    with col4:
        st.metric("Total Requests", len(df))
    
    st.markdown("---")
    
    # Time series chart
    st.markdown("### 📈 Cost Over Time")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_costs = df.groupby('date')['total_cost'].sum().reset_index()
    
    fig = px.line(
        daily_costs,
        x='date',
        y='total_cost',
        title='Daily Cost Trend',
        markers=True
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cost ($)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top expensive prompts
    st.markdown("---")
    st.markdown("### 💸 Most Expensive Prompts")
    
    expensive_prompts = cost_tracker.get_top_expensive_prompts(limit=10)
    if expensive_prompts:
        df_expensive = pd.DataFrame(expensive_prompts)
        df_expensive['total_cost'] = df_expensive['total_cost'].apply(lambda x: f"${x:.6f}")
        df_expensive['avg_latency_ms'] = df_expensive['avg_latency_ms'].apply(lambda x: f"{x:.0f}ms")
        
        st.dataframe(
            df_expensive[['agent_name', 'prompt_version', 'total_cost', 'execution_count', 'avg_latency_ms']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No prompt data available")
    
    # Export data
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export to CSV"):
            csv_path = f"cost_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            cost_tracker.export_to_csv(csv_path, start_datetime, end_datetime)
            st.success(f"Data exported to {csv_path}")
    
    with col2:
        # Create download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"cost_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def show_prompt_manager():
    """Prompt manager page"""
    st.header("📝 Prompt Version Manager")
    
    pm = load_prompt_manager()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📚 Browse Versions", "➕ Create Version", "🧪 A/B Testing"])
    
    with tab1:
        st.markdown("### Prompt Versions")
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search prompts", placeholder="Enter prompt name or tag...")
        with col2:
            status_filter = st.selectbox("Status", ["All", "production", "testing", "draft", "deprecated"])
        
        # Get all prompts
        status = None if status_filter == "All" else PromptStatus(status_filter)
        prompts = pm.search_prompts(query=search_query if search_query else None, status=status)
        
        if prompts:
            for prompt in prompts:
                with st.expander(f"{prompt.name} - {prompt.version} [{prompt.status.value}]"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {prompt.description}")
                        st.markdown(f"**Tags:** {', '.join(prompt.tags)}")
                        st.markdown(f"**Created:** {prompt.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown(f"**Updated:** {prompt.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        if prompt.performance_metrics:
                            st.markdown("**Performance Metrics:**")
                            st.json(prompt.performance_metrics)
                    
                    with col2:
                        if prompt.status != PromptStatus.PRODUCTION:
                            if st.button(f"Promote to Production", key=f"promote_{prompt.name}_{prompt.version}"):
                                pm.promote_to_production(prompt.name, prompt.version)
                                st.success("Promoted to production!")
                                st.rerun()
                    
                    st.markdown("**Content:**")
                    st.code(prompt.content, language="text")
        else:
            st.info("No prompts found")
    
    with tab2:
        st.markdown("### Create New Prompt Version")
        
        with st.form("create_prompt"):
            prompt_name = st.text_input("Prompt Name*", placeholder="e.g., research_agent_system")
            version = st.text_input("Version*", placeholder="e.g., v1.0.0")
            description = st.text_area("Description", placeholder="What's new in this version?")
            content = st.text_area("Prompt Content*", height=300, placeholder="Enter your prompt here...")
            tags = st.text_input("Tags (comma-separated)", placeholder="e.g., production, research")
            status = st.selectbox("Initial Status", ["draft", "testing"])
            
            submitted = st.form_submit_button("Create Version")
            
            if submitted:
                if not prompt_name or not version or not content:
                    st.error("Prompt name, version, and content are required")
                else:
                    try:
                        tag_list = [t.strip() for t in tags.split(",")] if tags else []
                        pm.create_version(
                            prompt_name=prompt_name,
                            content=content,
                            version=version,
                            description=description,
                            tags=tag_list,
                            status=PromptStatus(status)
                        )
                        st.success(f"Version {version} created successfully!")
                        st.cache_resource.clear()
                        st.rerun()
                    except ValueError as e:
                        st.error(f"Error: {e}")
    
    with tab3:
        st.markdown("### A/B Testing")
        st.info("A/B testing functionality requires running evaluations. Use the API or examples to run A/B tests programmatically.")
        
        # Show past test results
        test_results = pm.get_test_results()
        if test_results:
            st.markdown("### Previous Test Results")
            for result in test_results[-5:]:  # Show last 5
                with st.expander(f"{result['prompt_name']} - {result['test_id']}"):
                    st.markdown(f"**Versions Tested:** {', '.join(result['versions'])}")
                    st.markdown(f"**Test Cases:** {result['test_cases']}")
                    st.markdown(f"**Winner:** {result['winner']}")
                    st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                    st.json(result['results'])
        else:
            st.info("No A/B test results available yet")


def show_agent_executor():
    """Agent executor page"""
    st.header("🤖 Agent Executor")
    
    st.markdown("Execute tasks with AI agents in real-time")
    
    config = load_config()
    cost_tracker = load_cost_tracker()
    prompt_manager = load_prompt_manager()
    
    # Agent configuration
    col1, col2 = st.columns(2)
    with col1:
        agent_type = st.selectbox("Select Agent", ["Research Agent", "Analysis Agent", "Synthesis Agent"])
    with col2:
        model = st.selectbox("Model", [
            # Groq Models (Recommended - Current Nov 2024)
            "llama-3.1-8b-instant (Groq - Fastest)",
            "llama-3.3-70b-versatile (Groq - Latest)",
            "llama-3.1-70b-versatile (Groq - High Quality)",
            "mixtral-8x7b-32768 (Groq - Long Context)",
            "gemma2-9b-it (Groq - Efficient)",
            # OpenAI Models
            "gpt-4o-mini (OpenAI)",
            "gpt-4o (OpenAI)",
            "gpt-3.5-turbo (OpenAI)",
        ])
        
        # Extract actual model name
        model_name = model.split(" (")[0]
    
    st.markdown("---")
    
    # Task input
    st.markdown("### Task Configuration")
    
    if agent_type == "Research Agent":
        query = st.text_area("Research Query*", height=100, placeholder="What would you like to research?")
        scope = st.text_input("Scope", value="general", placeholder="e.g., technology, healthcare")
        depth = st.select_slider("Research Depth", options=["shallow", "medium", "deep"], value="medium")
        focus_areas = st.text_area("Focus Areas (one per line)", height=100, 
                                   placeholder="Key area 1\nKey area 2\nKey area 3")
        
        if st.button("🚀 Execute Research Task", type="primary", use_container_width=True):
            if not query:
                st.error("Please enter a research query")
            else:
                with st.spinner("🔍 Researching... This may take a moment..."):
                    # Create agent
                    from core.config import AgentConfig, AgentType
                    
                    # Detect provider from model name
                    provider = "groq" if any(x in model_name for x in ["llama", "mixtral", "gemma"]) else "openai"
                    
                    agent_config = AgentConfig(
                        name="dashboard_research_agent",
                        agent_type=AgentType.RESEARCH,
                        model=model_name,
                        provider=provider,
                        temperature=0.3
                    )
                    
                    # Get API keys from config
                    main_config = load_config()
                    api_keys = {}
                    if provider == "groq" and main_config.groq_api_key:
                        api_keys["groq"] = main_config.groq_api_key
                    elif provider == "openai" and main_config.openai_api_key:
                        api_keys["openai"] = main_config.openai_api_key
                    
                    agent = ResearchAgent(
                        config=agent_config,
                        cost_tracker=cost_tracker,
                        prompt_manager=prompt_manager,
                        api_keys=api_keys
                    )
                    
                    # Prepare task input
                    focus_list = [f.strip() for f in focus_areas.split("\n") if f.strip()]
                    task_input = {
                        'query': query,
                        'scope': scope,
                        'depth': depth,
                        'focus_areas': focus_list,
                        'task_id': f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    }
                    
                    # Execute
                    try:
                        response = asyncio.run(agent.execute(task_input, task_input['task_id']))
                        
                        if response.success:
                            st.success("✅ Task completed successfully!")
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("### 📊 Results")
                            st.markdown(response.output['research_findings'])
                            
                            # Display metrics
                            st.markdown("---")
                            st.markdown("### 📈 Performance Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Latency", f"{response.latency_ms:.0f}ms")
                            with col2:
                                st.metric("Input Tokens", response.token_usage.get('input_tokens', 0))
                            with col3:
                                st.metric("Output Tokens", response.token_usage.get('output_tokens', 0))
                            with col4:
                                st.metric("Cost", f"${response.cost:.6f}")
                        else:
                            st.error(f"❌ Task failed: {response.error}")
                    
                    except Exception as e:
                        st.error(f"Error executing task: {str(e)}")
    else:
        st.info(f"{agent_type} is not yet implemented. Use Research Agent for now.")


def show_settings():
    """Settings page"""
    st.header("⚙️ Settings")
    
    config = load_config()
    
    # Configuration display
    st.markdown("### Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**General Settings**")
        st.text(f"Default Model: {config.default_model}")
        st.text(f"Max Concurrent Agents: {config.orchestration.max_concurrent_agents}")
        st.text(f"Task Timeout: {config.orchestration.task_timeout}s")
    
    with col2:
        st.markdown("**Feature Flags**")
        st.text(f"Cost Tracking: {'✅' if config.cost_tracking.enabled else '❌'}")
        st.text(f"Prompt Versioning: {'✅' if config.prompt_versioning.enabled else '❌'}")
        st.text(f"LangSmith Tracing: {'✅' if config.observability.langsmith_enabled else '❌'}")
    
    st.markdown("---")
    
    # Cost alerts
    st.markdown("### 💰 Cost Alerts")
    
    with st.form("cost_alerts"):
        daily_threshold = st.number_input(
            "Daily Cost Alert Threshold ($)",
            value=config.cost_tracking.alert_threshold_daily,
            min_value=0.0
        )
        
        monthly_threshold = st.number_input(
            "Monthly Cost Alert Threshold ($)",
            value=config.cost_tracking.alert_threshold_monthly,
            min_value=0.0
        )
        
        if st.form_submit_button("Update Thresholds"):
            st.info("Alert thresholds would be updated (requires backend implementation)")
    
    st.markdown("---")
    
    # System info
    st.markdown("### 📊 System Information")
    
    cost_tracker = load_cost_tracker()
    prompt_manager = load_prompt_manager()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Prompts", len(prompt_manager.index))
    
    with col2:
        session_summary = cost_tracker.get_session_summary()
        st.metric("Session Requests", session_summary['total_requests'])
    
    with col3:
        st.metric("Session Cost", f"${session_summary['total_cost']:.6f}")
    
    st.markdown("---")
    
    # Danger zone
    with st.expander("⚠️ Danger Zone"):
        st.warning("Caution: These actions cannot be undone")
        
        if st.button("Clear Cost History", type="secondary"):
            st.error("This would delete all cost tracking data (not implemented in demo)")
        
        if st.button("Reset Prompt Versions", type="secondary"):
            st.error("This would delete all prompt versions (not implemented in demo)")


if __name__ == "__main__":
    main()
