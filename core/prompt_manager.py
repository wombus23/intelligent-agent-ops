"""
Prompt version management system with A/B testing capabilities
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
from enum import Enum


class PromptStatus(Enum):
    """Status of a prompt version"""
    DRAFT = "draft"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class PromptVersion:
    """Represents a specific version of a prompt"""
    name: str
    version: str
    content: str
    description: str
    created_at: datetime
    updated_at: datetime
    status: PromptStatus
    tags: List[str]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        d['status'] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['status'] = PromptStatus(data['status'])
        return cls(**data)


@dataclass
class ABTestResult:
    """Results from A/B testing prompt versions"""
    test_id: str
    prompt_name: str
    versions: List[str]
    test_cases: int
    results: Dict[str, Dict[str, float]]
    winner: Optional[str]
    confidence: float
    started_at: datetime
    completed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        d['completed_at'] = self.completed_at.isoformat()
        return d


class PromptVersionManager:
    """Manage prompt versions with Git-like version control"""
    
    def __init__(self, storage_path: str = "prompts/versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self.test_results_file = self.storage_path / "ab_test_results.json"
        
        self.index: Dict[str, List[PromptVersion]] = {}
        self.production_versions: Dict[str, str] = {}
        self._load_index()
    
    def _load_index(self):
        """Load prompt index from disk"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                data = json.load(f)
                
                # Load prompt versions
                for name, versions in data.get('prompts', {}).items():
                    self.index[name] = [
                        PromptVersion.from_dict(v) for v in versions
                    ]
                
                # Load production versions
                self.production_versions = data.get('production', {})
    
    def _save_index(self):
        """Save prompt index to disk"""
        data = {
            'prompts': {
                name: [v.to_dict() for v in versions]
                for name, versions in self.index.items()
            },
            'production': self.production_versions,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA-256 checksum of prompt content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_prompt_file(self, name: str, version: str) -> Path:
        """Get file path for a prompt version"""
        safe_name = name.replace('/', '_').replace('\\', '_')
        return self.storage_path / f"{safe_name}_{version}.txt"
    
    def create_version(
        self,
        prompt_name: str,
        content: str,
        version: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: PromptStatus = PromptStatus.DRAFT
    ) -> PromptVersion:
        """Create a new prompt version"""
        # Check if version already exists
        if prompt_name in self.index:
            existing_versions = [v.version for v in self.index[prompt_name]]
            if version in existing_versions:
                raise ValueError(f"Version {version} already exists for {prompt_name}")
        
        # Create prompt version
        checksum = self._calculate_checksum(content)
        now = datetime.now()
        
        prompt_version = PromptVersion(
            name=prompt_name,
            version=version,
            content=content,
            description=description,
            created_at=now,
            updated_at=now,
            status=status,
            tags=tags or [],
            metadata=metadata or {},
            performance_metrics={},
            checksum=checksum
        )
        
        # Save prompt content to file
        prompt_file = self._get_prompt_file(prompt_name, version)
        with open(prompt_file, 'w') as f:
            f.write(content)
        
        # Add to index
        if prompt_name not in self.index:
            self.index[prompt_name] = []
        
        self.index[prompt_name].append(prompt_version)
        self._save_index()
        
        return prompt_version
    
    def get_version(
        self,
        prompt_name: str,
        version: Optional[str] = None
    ) -> Optional[PromptVersion]:
        """
        Get a specific prompt version
        If version is None, returns production version
        """
        if prompt_name not in self.index:
            return None
        
        if version is None:
            # Get production version
            version = self.production_versions.get(prompt_name)
            if not version:
                return None
        
        # Find version
        for v in self.index[prompt_name]:
            if v.version == version:
                # Load content from file
                prompt_file = self._get_prompt_file(prompt_name, version)
                if prompt_file.exists():
                    with open(prompt_file, 'r') as f:
                        v.content = f.read()
                return v
        
        return None
    
    def list_versions(
        self,
        prompt_name: str,
        status: Optional[PromptStatus] = None
    ) -> List[PromptVersion]:
        """List all versions of a prompt"""
        if prompt_name not in self.index:
            return []
        
        versions = self.index[prompt_name]
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def update_version(
        self,
        prompt_name: str,
        version: str,
        **updates
    ) -> Optional[PromptVersion]:
        """Update metadata for a prompt version"""
        prompt_version = self.get_version(prompt_name, version)
        
        if not prompt_version:
            return None
        
        # Update allowed fields
        allowed_updates = ['description', 'tags', 'metadata', 'status', 'performance_metrics']
        
        for key, value in updates.items():
            if key in allowed_updates:
                setattr(prompt_version, key, value)
        
        prompt_version.updated_at = datetime.now()
        self._save_index()
        
        return prompt_version
    
    def promote_to_production(
        self,
        prompt_name: str,
        version: str
    ) -> bool:
        """Promote a version to production"""
        prompt_version = self.get_version(prompt_name, version)
        
        if not prompt_version:
            return False
        
        # Update old production version status
        old_prod_version = self.production_versions.get(prompt_name)
        if old_prod_version:
            old_prompt = self.get_version(prompt_name, old_prod_version)
            if old_prompt:
                old_prompt.status = PromptStatus.DEPRECATED
        
        # Set new production version
        prompt_version.status = PromptStatus.PRODUCTION
        self.production_versions[prompt_name] = version
        
        self._save_index()
        return True
    
    def rollback_version(
        self,
        prompt_name: str,
        target_version: str
    ) -> bool:
        """Rollback to a previous version"""
        return self.promote_to_production(prompt_name, target_version)
    
    def delete_version(
        self,
        prompt_name: str,
        version: str
    ) -> bool:
        """Delete a prompt version"""
        if prompt_name not in self.index:
            return False
        
        # Cannot delete production version
        if self.production_versions.get(prompt_name) == version:
            raise ValueError("Cannot delete production version. Promote another version first.")
        
        # Find and remove version
        versions = self.index[prompt_name]
        self.index[prompt_name] = [v for v in versions if v.version != version]
        
        # Delete prompt file
        prompt_file = self._get_prompt_file(prompt_name, version)
        if prompt_file.exists():
            prompt_file.unlink()
        
        self._save_index()
        return True
    
    def compare_versions(
        self,
        prompt_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two prompt versions"""
        v1 = self.get_version(prompt_name, version1)
        v2 = self.get_version(prompt_name, version2)
        
        if not v1 or not v2:
            return {}
        
        # Simple diff
        import difflib
        diff = list(difflib.unified_diff(
            v1.content.splitlines(),
            v2.content.splitlines(),
            lineterm='',
            fromfile=f'version {version1}',
            tofile=f'version {version2}'
        ))
        
        return {
            'version1': version1,
            'version2': version2,
            'diff': '\n'.join(diff),
            'checksum1': v1.checksum,
            'checksum2': v2.checksum,
            'performance_comparison': {
                'version1': v1.performance_metrics,
                'version2': v2.performance_metrics
            }
        }
    
    def search_prompts(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[PromptStatus] = None
    ) -> List[PromptVersion]:
        """Search prompts by query, tags, or status"""
        results = []
        
        for prompt_name, versions in self.index.items():
            for version in versions:
                # Filter by status
                if status and version.status != status:
                    continue
                
                # Filter by tags
                if tags and not any(tag in version.tags for tag in tags):
                    continue
                
                # Filter by query (search in name, description, content)
                if query:
                    query_lower = query.lower()
                    searchable = f"{version.name} {version.description} {version.content}".lower()
                    if query_lower not in searchable:
                        continue
                
                results.append(version)
        
        return results
    
    async def ab_test(
        self,
        prompt_name: str,
        versions: List[str],
        test_cases: List[Dict[str, Any]],
        metrics: List[str],
        evaluator_func
    ) -> ABTestResult:
        """
        Run A/B test on multiple prompt versions
        
        Args:
            prompt_name: Name of the prompt
            versions: List of version strings to test
            test_cases: List of test inputs
            metrics: List of metrics to evaluate
            evaluator_func: Async function to evaluate each version
        """
        test_id = f"{prompt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()
        
        results = {}
        
        # Test each version
        for version in versions:
            prompt_version = self.get_version(prompt_name, version)
            if not prompt_version:
                continue
            
            version_results = {metric: [] for metric in metrics}
            
            # Run test cases
            for test_case in test_cases:
                result = await evaluator_func(prompt_version, test_case)
                
                for metric in metrics:
                    if metric in result:
                        version_results[metric].append(result[metric])
            
            # Calculate average metrics
            results[version] = {
                metric: sum(values) / len(values) if values else 0.0
                for metric, values in version_results.items()
            }
        
        # Determine winner (highest average across all metrics)
        if results:
            avg_scores = {
                version: sum(metrics.values()) / len(metrics)
                for version, metrics in results.items()
            }
            winner = max(avg_scores, key=avg_scores.get)
            confidence = (avg_scores[winner] - min(avg_scores.values())) / max(avg_scores.values()) if max(avg_scores.values()) > 0 else 0.0
        else:
            winner = None
            confidence = 0.0
        
        completed_at = datetime.now()
        
        ab_result = ABTestResult(
            test_id=test_id,
            prompt_name=prompt_name,
            versions=versions,
            test_cases=len(test_cases),
            results=results,
            winner=winner,
            confidence=confidence,
            started_at=started_at,
            completed_at=completed_at
        )
        
        # Save test results
        self._save_test_results(ab_result)
        
        return ab_result
    
    def _save_test_results(self, result: ABTestResult):
        """Save A/B test results"""
        results = []
        
        if self.test_results_file.exists():
            with open(self.test_results_file, 'r') as f:
                results = json.load(f)
        
        results.append(result.to_dict())
        
        with open(self.test_results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_test_results(
        self,
        prompt_name: Optional[str] = None
    ) -> List[ABTestResult]:
        """Get A/B test results"""
        if not self.test_results_file.exists():
            return []
        
        with open(self.test_results_file, 'r') as f:
            results = json.load(f)
        
        if prompt_name:
            results = [r for r in results if r['prompt_name'] == prompt_name]
        
        return results
    
    def export_version(
        self,
        prompt_name: str,
        version: str,
        export_path: str
    ):
        """Export a prompt version to a file"""
        prompt_version = self.get_version(prompt_name, version)
        
        if not prompt_version:
            raise ValueError(f"Version {version} not found for {prompt_name}")
        
        export_data = prompt_version.to_dict()
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_version(self, import_path: str) -> PromptVersion:
        """Import a prompt version from a file"""
        with open(import_path, 'r') as f:
            data = json.load(f)
        
        prompt_version = PromptVersion.from_dict(data)
        
        # Add to index
        if prompt_version.name not in self.index:
            self.index[prompt_version.name] = []
        
        self.index[prompt_version.name].append(prompt_version)
        
        # Save content
        prompt_file = self._get_prompt_file(prompt_version.name, prompt_version.version)
        with open(prompt_file, 'w') as f:
            f.write(prompt_version.content)
        
        self._save_index()
        
        return prompt_version
