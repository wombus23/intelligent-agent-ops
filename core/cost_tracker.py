"""
Cost tracking and monitoring for LLM operations
"""
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from collections import defaultdict


@dataclass
class TokenUsage:
    """Token usage information"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    @property
    def token_efficiency_ratio(self) -> float:
        """Calculate output/input token ratio"""
        return self.output_tokens / self.input_tokens if self.input_tokens > 0 else 0


@dataclass
class CostRecord:
    """Record of a single LLM operation cost"""
    timestamp: datetime
    agent_name: str
    agent_type: str
    model: str
    task_id: str
    prompt_version: str
    token_usage: TokenUsage
    input_cost: float
    output_cost: float
    total_cost: float
    latency_ms: float
    success: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['token_usage'] = asdict(self.token_usage)
        d['metadata'] = json.dumps(self.metadata)
        return d


class CostTracker:
    """Track and analyze costs for LLM operations"""
    
    def __init__(self, db_path: str = "data/analytics/cost_tracking.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # In-memory cache for current session
        self.session_costs: List[CostRecord] = []
        self.total_session_cost = 0.0
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                model TEXT NOT NULL,
                task_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                input_cost REAL NOT NULL,
                output_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                latency_ms REAL NOT NULL,
                success BOOLEAN NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON cost_records(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_name 
            ON cost_records(agent_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model 
            ON cost_records(model)
        """)
        
        conn.commit()
        conn.close()
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_config: Dict[str, float]
    ) -> tuple[float, float, float]:
        """
        Calculate cost for token usage
        
        Returns:
            tuple: (input_cost, output_cost, total_cost)
        """
        input_cost_per_1k = cost_config.get(f"{model}_input", 0.0)
        output_cost_per_1k = cost_config.get(f"{model}_output", 0.0)
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    def record_usage(
        self,
        agent_name: str,
        agent_type: str,
        model: str,
        task_id: str,
        prompt_version: str,
        input_tokens: int,
        output_tokens: int,
        input_cost: float,
        output_cost: float,
        latency_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostRecord:
        """Record a cost entry"""
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        
        record = CostRecord(
            timestamp=datetime.now(),
            agent_name=agent_name,
            agent_type=agent_type,
            model=model,
            task_id=task_id,
            prompt_version=prompt_version,
            token_usage=token_usage,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            latency_ms=latency_ms,
            success=success,
            metadata=metadata or {}
        )
        
        # Save to database
        self._save_record(record)
        
        # Update session cache
        self.session_costs.append(record)
        self.total_session_cost += record.total_cost
        
        return record
    
    def _save_record(self, record: CostRecord):
        """Save record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cost_records (
                timestamp, agent_name, agent_type, model, task_id,
                prompt_version, input_tokens, output_tokens, total_tokens,
                input_cost, output_cost, total_cost, latency_ms, success, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.timestamp.isoformat(),
            record.agent_name,
            record.agent_type,
            record.model,
            record.task_id,
            record.prompt_version,
            record.token_usage.input_tokens,
            record.token_usage.output_tokens,
            record.token_usage.total_tokens,
            record.input_cost,
            record.output_cost,
            record.total_cost,
            record.latency_ms,
            record.success,
            json.dumps(record.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def get_costs_by_period(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        group_by: str = "day"
    ) -> pd.DataFrame:
        """Get cost breakdown by time period"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM cost_records"
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by specified period
        if group_by == "day":
            df['period'] = df['timestamp'].dt.date
        elif group_by == "hour":
            df['period'] = df['timestamp'].dt.floor('H')
        elif group_by == "month":
            df['period'] = df['timestamp'].dt.to_period('M')
        
        return df
    
    def get_agent_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get total costs grouped by agent"""
        df = self.get_costs_by_period(start_date, end_date)
        
        if df.empty:
            return {}
        
        return df.groupby('agent_name')['total_cost'].sum().to_dict()
    
    def get_model_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get total costs grouped by model"""
        df = self.get_costs_by_period(start_date, end_date)
        
        if df.empty:
            return {}
        
        return df.groupby('model')['total_cost'].sum().to_dict()
    
    def get_top_expensive_prompts(
        self,
        limit: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get most expensive prompts by total cost"""
        df = self.get_costs_by_period(start_date, end_date)
        
        if df.empty:
            return []
        
        grouped = df.groupby(['agent_name', 'prompt_version']).agg({
            'total_cost': 'sum',
            'total_tokens': 'sum',
            'latency_ms': 'mean',
            'task_id': 'count'
        }).reset_index()
        
        grouped.columns = ['agent_name', 'prompt_version', 'total_cost', 
                          'total_tokens', 'avg_latency_ms', 'execution_count']
        
        top_prompts = grouped.nlargest(limit, 'total_cost')
        
        return top_prompts.to_dict('records')
    
    def get_cost_summary(
        self,
        period: str = "today"
    ) -> Dict[str, Any]:
        """Get cost summary for a period"""
        if period == "today":
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()
        elif period == "week":
            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()
        elif period == "month":
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
        else:
            start_date = None
            end_date = None
        
        df = self.get_costs_by_period(start_date, end_date)
        
        if df.empty:
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_requests": 0,
                "avg_cost_per_request": 0.0,
                "success_rate": 0.0
            }
        
        return {
            "total_cost": float(df['total_cost'].sum()),
            "total_tokens": int(df['total_tokens'].sum()),
            "total_requests": len(df),
            "avg_cost_per_request": float(df['total_cost'].mean()),
            "success_rate": float(df['success'].mean() * 100),
            "by_agent": df.groupby('agent_name')['total_cost'].sum().to_dict(),
            "by_model": df.groupby('model')['total_cost'].sum().to_dict()
        }
    
    def check_alert_threshold(
        self,
        period: str = "today",
        threshold: float = 100.0
    ) -> tuple[bool, float]:
        """
        Check if cost exceeds threshold
        
        Returns:
            tuple: (threshold_exceeded, current_cost)
        """
        summary = self.get_cost_summary(period)
        current_cost = summary["total_cost"]
        
        return current_cost > threshold, current_cost
    
    def export_to_csv(
        self,
        filepath: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """Export cost records to CSV"""
        df = self.get_costs_by_period(start_date, end_date)
        df.to_csv(filepath, index=False)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session costs"""
        if not self.session_costs:
            return {
                "total_cost": 0.0,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0
            }
        
        successful = sum(1 for r in self.session_costs if r.success)
        total_tokens = sum(r.token_usage.total_tokens for r in self.session_costs)
        
        return {
            "total_cost": self.total_session_cost,
            "total_requests": len(self.session_costs),
            "successful_requests": successful,
            "failed_requests": len(self.session_costs) - successful,
            "total_tokens": total_tokens,
            "avg_cost_per_request": self.total_session_cost / len(self.session_costs)
        }
