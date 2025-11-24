"""
Unit tests for CostTracker
"""
import pytest
from datetime import datetime, timedelta
from core.cost_tracker import CostTracker, TokenUsage, CostRecord


class TestCostTracker:
    """Test suite for CostTracker"""
    
    @pytest.fixture
    def cost_tracker(self, tmp_path):
        """Create a temporary cost tracker"""
        db_path = tmp_path / "test_cost_tracking.db"
        return CostTracker(str(db_path))
    
    def test_initialization(self, cost_tracker):
        """Test cost tracker initialization"""
        assert cost_tracker is not None
        assert cost_tracker.db_path.exists()
        assert cost_tracker.total_session_cost == 0.0
        assert len(cost_tracker.session_costs) == 0
    
    def test_record_usage(self, cost_tracker):
        """Test recording usage"""
        record = cost_tracker.record_usage(
            agent_name="test_agent",
            agent_type="research",
            model="gpt-4o-mini",
            task_id="test_001",
            prompt_version="v1.0.0",
            input_tokens=100,
            output_tokens=200,
            input_cost=0.001,
            output_cost=0.002,
            latency_ms=150.5,
            success=True,
            metadata={"test": "data"}
        )
        
        assert record is not None
        assert record.agent_name == "test_agent"
        assert record.total_cost == 0.003
        assert record.token_usage.total_tokens == 300
        assert len(cost_tracker.session_costs) == 1
        assert cost_tracker.total_session_cost == 0.003
    
    def test_get_session_summary(self, cost_tracker):
        """Test getting session summary"""
        # Record some usage
        for i in range(5):
            cost_tracker.record_usage(
                agent_name=f"agent_{i}",
                agent_type="research",
                model="gpt-4o-mini",
                task_id=f"task_{i}",
                prompt_version="v1.0.0",
                input_tokens=100,
                output_tokens=200,
                input_cost=0.001,
                output_cost=0.002,
                latency_ms=150.5,
                success=i % 2 == 0  # Alternate success/failure
            )
        
        summary = cost_tracker.get_session_summary()
        
        assert summary['total_requests'] == 5
        assert summary['successful_requests'] == 3
        assert summary['failed_requests'] == 2
        assert summary['total_cost'] == 0.015
        assert summary['total_tokens'] == 1500
    
    def test_token_efficiency_ratio(self):
        """Test token efficiency ratio calculation"""
        token_usage = TokenUsage(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300
        )
        
        assert token_usage.token_efficiency_ratio == 2.0
    
    def test_calculate_cost(self, cost_tracker):
        """Test cost calculation"""
        cost_config = {
            "gpt-4o-mini_input": 0.00015,
            "gpt-4o-mini_output": 0.0006,
        }
        
        input_cost, output_cost, total_cost = cost_tracker.calculate_cost(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=2000,
            cost_config=cost_config
        )
        
        assert input_cost == 0.00015
        assert output_cost == 0.0012
        assert total_cost == 0.00135
    
    def test_get_costs_by_period(self, cost_tracker):
        """Test getting costs by period"""
        # Record usage
        cost_tracker.record_usage(
            agent_name="test_agent",
            agent_type="research",
            model="gpt-4o-mini",
            task_id="test_001",
            prompt_version="v1.0.0",
            input_tokens=100,
            output_tokens=200,
            input_cost=0.001,
            output_cost=0.002,
            latency_ms=150.5,
            success=True
        )
        
        df = cost_tracker.get_costs_by_period()
        
        assert not df.empty
        assert 'agent_name' in df.columns
        assert 'total_cost' in df.columns
        assert len(df) == 1
    
    def test_check_alert_threshold(self, cost_tracker):
        """Test alert threshold checking"""
        # Record usage below threshold
        cost_tracker.record_usage(
            agent_name="test_agent",
            agent_type="research",
            model="gpt-4o-mini",
            task_id="test_001",
            prompt_version="v1.0.0",
            input_tokens=100,
            output_tokens=200,
            input_cost=0.001,
            output_cost=0.002,
            latency_ms=150.5,
            success=True
        )
        
        exceeded, current_cost = cost_tracker.check_alert_threshold(
            period="today",
            threshold=10.0
        )
        
        assert not exceeded
        assert current_cost < 10.0


@pytest.mark.asyncio
class TestCostTrackerAsync:
    """Async tests for cost tracker"""
    
    @pytest.fixture
    def cost_tracker(self, tmp_path):
        """Create a temporary cost tracker"""
        db_path = tmp_path / "test_cost_tracking_async.db"
        return CostTracker(str(db_path))
    
    async def test_concurrent_recording(self, cost_tracker):
        """Test concurrent cost recording"""
        import asyncio
        
        async def record_cost(i):
            cost_tracker.record_usage(
                agent_name=f"agent_{i}",
                agent_type="research",
                model="gpt-4o-mini",
                task_id=f"task_{i}",
                prompt_version="v1.0.0",
                input_tokens=100,
                output_tokens=200,
                input_cost=0.001,
                output_cost=0.002,
                latency_ms=150.5,
                success=True
            )
        
        # Record 10 costs concurrently
        await asyncio.gather(*[record_cost(i) for i in range(10)])
        
        summary = cost_tracker.get_session_summary()
        assert summary['total_requests'] == 10
        assert summary['total_cost'] == 0.03
