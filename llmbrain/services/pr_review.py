from pydantic import BaseModel
from typing import List, Optional
from llmbrain.llm.base import BaseLLMProvider
from llmbrain.services.git_diff import analyze_git_diff
from llmbrain.services.project_service import ProjectService
import logging

logger = logging.getLogger(__name__)

class PRComment(BaseModel):
    file_path: str
    line_number: Optional[int]
    comment: str
    severity: str

class PRReviewer:
    def __init__(self, project_id: str, service: ProjectService, llm: BaseLLMProvider):
        self.project_id = project_id
        self.service = service
        self.llm = llm
        
    async def generate_review(self, path: str, base_ref: str) -> List[PRComment]:
        changed_files = analyze_git_diff(path, base_ref)
        comments = []
        
        for cf in changed_files:
            if not cf.diff_content or not cf.diff_content.strip():
                continue
                
            prompt = f"""
Review the following code changes for {cf.path}.
Provide constructive feedback, identify potential bugs, or suggest improvements.
Only provide comments if there is a real issue. Do not nitpick.

Diff:
{cf.diff_content[:5000]}
"""
            schema = {
                "type": "object",
                "properties": {
                    "comments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line_number": {"type": ["integer", "null"]},
                                "comment": {"type": "string"},
                                "severity": {"type": "string", "enum": ["info", "warning", "critical"]}
                            },
                            "required": ["comment", "severity"]
                        }
                    }
                },
                "required": ["comments"]
            }
            
            try:
                res = await self.llm.generate_structured(prompt, schema)
                val = res.structured_data
                for c in val.get("comments", []):
                    comments.append(PRComment(
                        file_path=cf.path,
                        line_number=c.get("line_number"),
                        comment=c.get("comment"),
                        severity=c.get("severity", "info")
                    ))
            except Exception as e:
                logger.error(f"Failed to review {cf.path}: {e}")
                
        return comments
