# Prompts package — edit these files to tune agent behaviour without touching agent logic.
from prompts.knowledge_agent_prompts import (
    get_base_context_block,
    get_process_prompt,
    get_physics_prompt,
    get_equipment_prompt,
    get_oem_prompt,
    get_critique_injection,
    get_critique_prompt,
)
from prompts.analyst_prompts import get_analyst_system_prompt
from prompts.report_prompts import get_report_prompt
from prompts.planner_prompts import get_planner_prompt

__all__ = [
    "get_base_context_block",
    "get_process_prompt",
    "get_physics_prompt",
    "get_equipment_prompt",
    "get_oem_prompt",
    "get_critique_injection",
    "get_critique_prompt",
    "get_analyst_system_prompt",
    "get_report_prompt",
    "get_planner_prompt",
]
