from pydantic import BaseModel
from typing import List, Dict, Any
from llmbrain.llm.base import BaseLLMProvider
from llmbrain.services.project_service import ProjectService
import logging

logger = logging.getLogger(__name__)

class DriftReport(BaseModel):
    wiki_id: str
    wiki_title: str
    is_drifting: bool
    rationale: str
    risk_level: str

class DriftDetector:
    def __init__(self, project_id: str, service: ProjectService, llm: BaseLLMProvider):
        self.project_id = project_id
        self.service = service
        self.llm = llm
        self.store = service._store_for_project(project_id)
        
    async def analyze_drift(self) -> List[DriftReport]:
        if not self.store:
            return []
            
        pages = self.store.get_wiki_pages(self.project_id)
        reports = []
        
        chunks = self.store.get_chunks(self.project_id) or []
        
        for page in pages:
            sources = page.get("sources", [])
            if not sources:
                continue
                
            doc_content = page.get("markdown_content", "")
            if not doc_content:
                continue
            
            code_snippets = []
            source_paths = {s.get("path") for s in sources if s.get("path")}
            
            for c in chunks:
                if c.get("path") in source_paths:
                    code_snippets.append(c.get("content", ""))
                    
            if not code_snippets:
                continue
                
            # Limit context size
            code_context = "\n---\n".join(code_snippets[:10]) 
            
            prompt = f"""
Compare the following documentation with the referenced code.
Detect if there is any semantic drift (i.e. the documentation describes something that is no longer true in the code, or misses critical new features).

Documentation:
{doc_content[:2000]}

Referenced Code:
{code_context[:4000]}
"""
            schema = {
                "type": "object",
                "properties": {
                    "is_drifting": {"type": "boolean"},
                    "rationale": {"type": "string"},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["is_drifting", "rationale", "risk_level"]
            }
            
            try:
                res = await self.llm.generate_structured(prompt, schema)
                val = res.structured_data
                if val:
                    reports.append(DriftReport(
                        wiki_id=page["id"],
                        wiki_title=page["title"],
                        is_drifting=val.get("is_drifting", False),
                        rationale=val.get("rationale", ""),
                        risk_level=val.get("risk_level", "low")
                    ))
            except Exception as e:
                logger.error(f"Failed to analyze drift for {page['title']}: {e}")
                
        return reports
