from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MODEL_NOT_CONFIGURED = 2001300
MODEL_NOT_ALLOWED = 2001301


def _models(*names: str) -> dict[str, str]:
    return {name.casefold(): name for name in names}


def _provider_key(value: str) -> str:
    return re.sub(r'[\s_.-]+', '', value.casefold())


@dataclass(frozen=True)
class RepairProvider:
    label: str
    aliases: tuple[str, ...]
    opencode_provider: str
    npm: str
    rewrites: Mapping[str, str]
    models: Mapping[str, str]


# This registry mirrors every `llm` and `vlm` model in
# backend/core/config/model_catalog.yaml. Keep the repair executor restricted
# to text-capable models even when the catalog also contains media models.
REPAIR_MODEL_PROVIDERS: dict[str, RepairProvider] = {
    'claude': RepairProvider(
        label="Claude", aliases=("claude", "anthropic"), opencode_provider="anthropic", npm="@ai-sdk/anthropic", rewrites={},
        models=_models(
            'claude-haiku-4-5',
            'claude-opus-4-7',
            'claude-sonnet-4-6',
        ),
    ),
    'deepseek': RepairProvider(
        label="DeepSeek", aliases=("deepseek",), opencode_provider="deepseek", npm="@ai-sdk/openai-compatible", rewrites={"https://api.deepseek.com/v1": "https://api.deepseek.com"},
        models=_models(
            'deepseek-v4-flash',
            'deepseek-v4-pro',
        ),
    ),
    'doubao': RepairProvider(
        label="Doubao", aliases=("doubao", "volcengine", "ark"), opencode_provider="doubao", npm="@ai-sdk/openai-compatible", rewrites={},
        models=_models(
            'deepseek-r1-250528',
            'deepseek-v3-1-terminus',
            'deepseek-v3-2-251201',
            'deepseek-v3-250324',
            'doubao-1-5-lite-32k-250115',
            'doubao-1-5-pro-32k-250115',
            'doubao-1-5-pro-32k-character-250228',
            'doubao-1-5-pro-32k-character-250715',
            'doubao-1-5-vision-pro-32k-250115',
            'doubao-seed-1-6-250615',
            'doubao-seed-1-6-251015',
            'doubao-seed-1-6-flash-250828',
            'doubao-seed-1-6-vision-250815',
            'doubao-seed-1-8-251228',
            'doubao-seed-2-0-code-preview-260215',
            'doubao-seed-2-0-lite-260215',
            'doubao-seed-2-0-mini-260215',
            'doubao-seed-2-0-pro-260215',
            'doubao-seed-code-preview-251028',
            'glm-4-7-251222',
        ),
    ),
    'glm': RepairProvider(
        label="GLM", aliases=("glm", "zhipu", "zhipuai"), opencode_provider="zhipuai", npm="@ai-sdk/openai-compatible", rewrites={},
        models=_models(
            'AutoGLM-Phone',
            'CharGLM-4',
            'CodeGeeX-4',
            'GLM-4-Flash-250414',
            'GLM-4-FlashX-250414',
            'GLM-4-Long',
            'GLM-4.1V-Thinking-Flash',
            'GLM-4.1V-Thinking-FlashX',
            'GLM-4.5-Air',
            'GLM-4.5-AirX',
            'GLM-4.5-Flash',
            'GLM-4.6',
            'GLM-4.6V',
            'GLM-4.6V-Flash',
            'GLM-4.7',
            'GLM-4.7-Flash',
            'GLM-4.7-FlashX',
            'GLM-4V-Flash',
            'GLM-5',
            'GLM-5-Turbo',
            'GLM-5.1',
            'GLM-5V-Turbo',
        ),
    ),
    'kimi': RepairProvider(
        label="Kimi", aliases=("kimi", "moonshot", "moonshotai"), opencode_provider="moonshotai-cn", npm="@ai-sdk/openai-compatible", rewrites={"https://api.moonshot.cn": "https://api.moonshot.cn/v1"},
        models=_models(
            'kimi-k2-0711-preview',
            'kimi-k2-0905-preview',
            'kimi-k2-thinking',
            'kimi-k2-thinking-turbo',
            'kimi-k2-turbo-preview',
            'kimi-k2.5',
            'kimi-k2.6',
            'moonshot-v1-128k',
            'moonshot-v1-128k-vision-preview',
            'moonshot-v1-32k',
            'moonshot-v1-32k-vision-preview',
            'moonshot-v1-8k',
            'moonshot-v1-8k-vision-preview',
        ),
    ),
    'minimax': RepairProvider(
        label="MiniMax", aliases=("minimax", "minimaxcn"), opencode_provider="minimax-cn", npm="@ai-sdk/anthropic", rewrites={"https://api.minimaxi.com/v1": "https://api.minimaxi.com/anthropic/v1"},
        models=_models(
            'M2-her',
            'MiniMax-M2.5',
            'MiniMax-M2.5-highspeed',
            'MiniMax-M2.7',
            'MiniMax-M2.7-highspeed',
        ),
    ),
    'openai': RepairProvider(
        label="OpenAI", aliases=("openai",), opencode_provider="openai", npm="@ai-sdk/openai", rewrites={"https://api.openai.com": "https://api.openai.com/v1"},
        models=_models(
            'gpt-4.1',
            'gpt-4.1-mini',
            'gpt-4o-mini',
            'gpt-5',
            'gpt-5-mini',
            'gpt-5-nano',
            'gpt-5-pro',
            'gpt-5.1',
            'gpt-5.2',
            'gpt-5.2-pro',
            'gpt-5.4',
            'gpt-5.4-mini',
            'gpt-5.4-nano',
            'gpt-5.4-pro',
            'gpt-5.5',
            'gpt-5.5-pro',
            'o3',
        ),
    ),
    'qwen': RepairProvider(
        label="Qwen", aliases=("qwen", "alibaba", "alibabacn", "dashscope"), opencode_provider="alibaba-cn", npm="@ai-sdk/openai-compatible", rewrites={"https://dashscope.aliyuncs.com": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        models=_models(
            'qvq-72b-preview',
            'qvq-max',
            'qvq-plus',
            'qwen-flash',
            'qwen-flash-2025-07-28',
            'qwen-long',
            'qwen-max',
            'qwen-plus',
            'qwen-plus-2025-12-01',
            'qwen-turbo',
            'qwen-vl-max',
            'qwen-vl-ocr',
            'qwen-vl-plus',
            'qwen2.5-0.5b-instruct',
            'qwen2.5-1.5b-instruct',
            'qwen2.5-14b-instruct',
            'qwen2.5-14b-instruct-1m',
            'qwen2.5-32b-instruct',
            'qwen2.5-3b-instruct',
            'qwen2.5-72b-instruct',
            'qwen2.5-7b-instruct',
            'qwen2.5-7b-instruct-1m',
            'qwen3-0.6b',
            'qwen3-1.7b',
            'qwen3-14b',
            'qwen3-235b-a22b',
            'qwen3-235b-a22b-instruct-2507',
            'qwen3-235b-a22b-thinking-2507',
            'qwen3-30b-a3b',
            'qwen3-30b-a3b-instruct-2507',
            'qwen3-30b-a3b-thinking-2507',
            'qwen3-32b',
            'qwen3-4b',
            'qwen3-8b',
            'qwen3-next-80b-a3b-instruct',
            'qwen3-next-80b-a3b-thinking',
            'qwen3-vl-flash',
            'qwen3-vl-flash-us',
            'qwen3-vl-plus',
            'qwen3.5-122b-a10b',
            'qwen3.5-27b',
            'qwen3.5-35b-a3b',
            'qwen3.5-397b-a17b',
            'qwen3.5-4b',
            'qwen3.5-9b',
            'qwen3.5-flash',
            'qwen3.5-plus',
            'qwen3.6-27b',
            'qwen3.6-35b-a3b',
            'qwen3.6-flash',
            'qwen3.6-max-preview',
            'qwen3.6-plus',
            'qwq-32b',
            'qwq-32b-preview',
            'qwq-plus',
        ),
    ),
    'sensenova': RepairProvider(
        label="SenseNova", aliases=("sensenova", "sensetime", "sensecore"), opencode_provider="sensenova", npm="@ai-sdk/openai-compatible", rewrites={},
        models=_models(
            'DeepSeek V4 Flash',
            'DeepSeek-R1',
            'DeepSeek-R1-Distill-Qwen-14B',
            'DeepSeek-V3',
            'DeepSeek-V3-1',
            'Qwen2-5-Coder',
            'Qwen3-235B',
            'Qwen3-32B',
            'Qwen3-Coder',
            'SenseChat',
            'SenseChat-128K',
            'SenseChat-5',
            'SenseChat-5-1202',
            'SenseChat-5-Cantonese',
            'SenseChat-Turbo',
            'SenseChat-Turbo-1202',
            'SenseChat-Vision',
            'SenseNova 6.7 Flash-Lite',
            'SenseNova U1 Fast',
            'SenseNova-V6-5-Pro',
            'SenseNova-V6-5-Turbo',
            'SenseNova-V6-Pro',
            'SenseNova-V6-Reasoner',
            'SenseNova-V6-Turbo',
            'deepseek-v4-flash',
            'glm-5.2',
            'sensenova-6.7-flash-lite',
        ),
    ),
    'siliconflow': RepairProvider(
        label="SiliconFlow", aliases=("siliconflow",), opencode_provider="siliconflow-cn", npm="@ai-sdk/openai-compatible", rewrites={},
        models=_models(
            'deepseek-ai/DeepSeek-V4-Flash',
            'Pro/moonshotai/Kimi-K2.6',
            'Pro/zai-org/GLM-5.1',
            'MiniMaxAI/MiniMax-M2.5',
            'Pro/MiniMaxAI/MiniMax-M2.5',
            'Pro/zai-org/GLM-5',
            'Pro/deepseek-ai/DeepSeek-V3.2',
            'Pro/moonshotai/Kimi-K2.5',
            'Pro/zai-org/GLM-4.7',
            'deepseek-ai/DeepSeek-V3.2',
            'Pro/deepseek-ai/DeepSeek-V3.1-Terminus',
            'deepseek-ai/DeepSeek-V3.1-Terminus',
            'Qwen/Qwen3.6-35B-A3B',
            'Qwen/Qwen3.6-27B',
            'Qwen/Qwen3.5-397B-A17B',
            'Qwen/Qwen3.5-122B-A10B',
            'Qwen/Qwen3.5-35B-A3B',
            'Qwen/Qwen3.5-27B',
            'Qwen/Qwen3.5-9B',
            'Qwen/Qwen3.5-4B',
            'deepseek-ai/DeepSeek-R1',
            'Pro/deepseek-ai/DeepSeek-R1',
            'deepseek-ai/DeepSeek-V3',
            'Pro/deepseek-ai/DeepSeek-V3',
            'stepfun-ai/Step-3.5-Flash',
            'zai-org/GLM-4.6V',
            'moonshotai/Kimi-K2-Thinking',
            'Pro/moonshotai/Kimi-K2-Thinking',
            'zai-org/GLM-4.6',
            'Qwen/Qwen3-VL-32B-Instruct',
            'Qwen/Qwen3-VL-32B-Thinking',
            'Qwen/Qwen3-VL-8B-Instruct',
            'Qwen/Qwen3-VL-8B-Thinking',
            'Qwen/Qwen3-VL-30B-A3B-Instruct',
            'Qwen/Qwen3-VL-30B-A3B-Thinking',
            'moonshotai/Kimi-K2-Instruct-0905',
            'Pro/moonshotai/Kimi-K2-Instruct-0905',
            'inclusionAI/Ling-flash-2.0',
            'Qwen/Qwen3-Coder-30B-A3B-Instruct',
            'Qwen/Qwen3-30B-A3B-Thinking-2507',
            'Qwen/Qwen3-30B-A3B-Instruct-2507',
            'Qwen/Qwen3-235B-A22B-Instruct-2507',
            'THUDM/GLM-4.1V-9B-Thinking',
            'tencent/Hunyuan-A13B-Instruct',
            'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
            'Qwen/Qwen3-32B',
            'Qwen/Qwen3-14B',
            'Qwen/Qwen3-8B',
            'ByteDance-Seed/Seed-OSS-36B-Instruct',
            'zai-org/GLM-4.5V',
            'zai-org/GLM-4.5-Air',
            'Qwen/Qwen2.5-72B-Instruct-128K',
            'Qwen/Qwen2.5-72B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
            'Qwen/Qwen2.5-14B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'Pro/Qwen/Qwen2.5-7B-Instruct',
        ),
    ),

}


_PROVIDER_ALIASES = {
    _provider_key(alias): provider
    for provider, spec in REPAIR_MODEL_PROVIDERS.items()
    for alias in spec.aliases
}


class EvoModelConfigError(ValueError):
    def __init__(self, code: int, reason: str, *, provider: str = '', model: str = '',
                 missing_fields: tuple[str, ...] = ()) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.provider = provider
        self.model = model
        self.missing_fields = missing_fields

    def detail(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            'reason': self.reason,
            'model_role': 'evo_llm',
        }
        if self.provider:
            data['provider'] = self.provider
        if self.model:
            data['model'] = self.model
        if self.missing_fields:
            data['missing_fields'] = list(self.missing_fields)
        return {
            'code': self.code,
            'message': (
                '请先完成 evo_llm 模型配置'
                if self.code == MODEL_NOT_CONFIGURED
                else '当前配置的自进化模型不支持自进化'
            ),
            'data': data,
        }


def build_repair_opencode_settings(role: object) -> dict[str, str]:
    if not isinstance(role, Mapping):
        raise EvoModelConfigError(MODEL_NOT_CONFIGURED, 'model_not_configured')

    raw_provider = _text(role.get('provider')) or _text(role.get('source'))
    raw_model = _text(role.get('model'))
    base_url = _text(role.get('base_url'))
    api_key = _text(role.get('api_key'))
    missing = tuple(
        field
        for field, value in (
            ('source', raw_provider),
            ('model', raw_model),
            ('base_url', base_url),
            ('api_key', api_key or ('skip_auth' if role.get('skip_auth') is True else '')),
        )
        if not value
    )
    if missing:
        raise EvoModelConfigError(
            MODEL_NOT_CONFIGURED,
            'model_config_incomplete',
            provider=raw_provider,
            model=raw_model,
            missing_fields=missing,
        )

    provider = _PROVIDER_ALIASES.get(_provider_key(raw_provider), '')
    spec = REPAIR_MODEL_PROVIDERS.get(provider)
    model = spec.models.get(raw_model.casefold(), '') if spec else ''
    if not spec or not model:
        raise EvoModelConfigError(
            MODEL_NOT_ALLOWED,
            'evo_llm_not_allowed',
            provider=raw_provider,
            model=raw_model,
        )

    normalized_url = spec.rewrites.get(base_url.rstrip('/'), base_url.rstrip('/'))
    return {
        'model': f'{spec.opencode_provider}/{model}',
        'provider': spec.opencode_provider,
        'provider_model': model,
        'provider_label': spec.label,
        'npm': spec.npm,
        'base_url': normalized_url,
        'api_key': api_key,
        'skip_auth': 'true' if role.get('skip_auth') is True else '',
    }


def _text(value: object) -> str:
    return str(value or '').strip()
