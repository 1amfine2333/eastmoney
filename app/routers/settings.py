"""
Settings endpoints.
"""
import os
from fastapi import APIRouter

from app.models.settings import SettingsUpdate
from app.core.utils import load_env_file, save_env_file, mask_api_key

router = APIRouter(prefix="/api/settings", tags=["Settings"])


@router.get("")
async def get_settings():
    """Get current settings (with masked API keys)."""
    env = load_env_file()

    return {
        "llm_provider": env.get("LLM_PROVIDER", "gemini"),
        "gemini_api_key_masked": mask_api_key(env.get("GEMINI_API_KEY", "")),
        "openai_api_key_masked": mask_api_key(env.get("OPENAI_API_KEY", "")),
        "openai_base_url": env.get("OPENAI_BASE_URL", ""),
        "openai_model": env.get("OPENAI_MODEL", ""),
        "tavily_api_key_masked": mask_api_key(env.get("TAVILY_API_KEY", ""))
    }


@router.post("")
async def update_settings(settings: SettingsUpdate):
    """Update application settings."""
    updates = {}
    if settings.llm_provider:
        updates["LLM_PROVIDER"] = settings.llm_provider
    if settings.gemini_api_key:
        updates["GEMINI_API_KEY"] = settings.gemini_api_key
    if settings.openai_api_key:
        updates["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.openai_base_url is not None:
        updates["OPENAI_BASE_URL"] = settings.openai_base_url
    if settings.openai_model is not None:
        updates["OPENAI_MODEL"] = settings.openai_model
    if settings.tavily_api_key:
        updates["TAVILY_API_KEY"] = settings.tavily_api_key

    save_env_file(updates)

    # Update runtime env
    for k, v in updates.items():
        if v is not None:
            os.environ[k] = v

    return {"status": "success"}
