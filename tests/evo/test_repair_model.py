from pathlib import Path

import pytest

from evo.repair_model import (
    MODEL_NOT_ALLOWED,
    MODEL_NOT_CONFIGURED,
    REPAIR_MODEL_PROVIDERS,
    EvoModelConfigError,
    build_repair_opencode_settings,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'backend/core/config/model_catalog.yaml'
CATALOG_PROVIDER_KEYS = {
    'Claude': 'claude',
    'DeepSeek': 'deepseek',
    'Doubao': 'doubao',
    'GLM': 'glm',
    'Kimi': 'kimi',
    'Minimax': 'minimax',
    'OpenAI': 'openai',
    'Qwen': 'qwen',
    'SenseNova': 'sensenova',
    'SiliconFlow': 'siliconflow',
}
BASE_URLS = {
    'claude': 'https://api.anthropic.com/v1',
    'deepseek': 'https://api.deepseek.com/v1',
    'doubao': 'https://ark.cn-beijing.volces.com/api/v3',
    'glm': 'https://open.bigmodel.cn/api/paas/v4',
    'kimi': 'https://api.moonshot.cn',
    'minimax': 'https://api.minimaxi.com/v1',
    'openai': 'https://api.openai.com',
    'qwen': 'https://dashscope.aliyuncs.com',
    'sensenova': 'https://api.sensenova.cn/compatible-mode/v1',
    'siliconflow': 'https://api.siliconflow.cn/v1',
}


def _catalog_text_models() -> dict[str, set[str]]:
    """Read the catalog's simple supplier/model/type structure without a YAML dependency."""
    models = {key: set() for key in CATALOG_PROVIDER_KEYS.values()}
    provider = ''
    model = ''
    for line in CATALOG.read_text(encoding='utf-8').splitlines():
        if line.startswith('    - name: '):
            provider = CATALOG_PROVIDER_KEYS.get(line.removeprefix('    - name: '), '')
            model = ''
        elif line.startswith('        - name: '):
            model = line.removeprefix('        - name: ')
        elif line in ('          type: llm', '          type: vlm'):
            if provider and model:
                models[provider].add(model)
            model = ''
        elif line.startswith('          type: '):
            model = ''
    return models


def _role(provider: str, model: str, *, base_url: str = '', **extra: object) -> dict[str, object]:
    return {
        'source': provider,
        'model': model,
        'base_url': base_url or BASE_URLS.get(provider, 'https://example.test/v1'),
        'api_key': 'test-key',
        **extra,
    }


def test_registry_matches_catalog_text_models():
    catalog_models = _catalog_text_models()
    registry_models = {
        provider: set(spec.models.values())
        for provider, spec in REPAIR_MODEL_PROVIDERS.items()
    }

    assert registry_models == catalog_models
    assert sum(map(len, registry_models.values())) == 221


@pytest.mark.parametrize(
    ('provider', 'model'),
    [
        (provider, model)
        for provider, spec in REPAIR_MODEL_PROVIDERS.items()
        for model in spec.models.values()
    ],
)
def test_every_catalog_model_builds_opencode_settings(provider, model):
    settings = build_repair_opencode_settings(_role(provider, model))
    spec = REPAIR_MODEL_PROVIDERS[provider]

    assert settings['provider'] == spec.opencode_provider
    assert settings['provider_label'] == spec.label
    assert settings['provider_model'] == model
    assert settings['model'] == f'{spec.opencode_provider}/{model}'
    assert settings['npm'] == spec.npm


@pytest.mark.parametrize(
    ('provider', 'model', 'expected_provider'),
    [
        ('Anthropic', 'CLAUDE-SONNET-4-6', 'anthropic'),
        ('volcengine', 'DOUBAO-SEED-2-0-PRO-260215', 'doubao'),
        ('zhipu.ai', 'glm-5-turbo', 'zhipuai'),
        ('Moonshot AI', 'KIMI-K2.6', 'moonshotai-cn'),
        ('Alibaba CN', 'QWEN-PLUS', 'alibaba-cn'),
        ('SenseTime', 'SENSECHAT-5', 'sensenova'),
    ],
)
def test_aliases_and_model_case_are_normalized(provider, model, expected_provider):
    settings = build_repair_opencode_settings({
        'provider': provider,
        'source': 'not-used',
        'model': model,
        'base_url': 'https://custom.example/v1',
        'api_key': 'test-key',
    })
    assert settings['provider'] == expected_provider


@pytest.mark.parametrize(
    ('provider', 'model', 'base_url', 'expected_url'),
    [
        ('deepseek', 'deepseek-v4-flash', 'https://api.deepseek.com/v1/', 'https://api.deepseek.com'),
        ('kimi', 'kimi-k2.6', 'https://api.moonshot.cn/', 'https://api.moonshot.cn/v1'),
        ('minimax', 'MiniMax-M2.7', 'https://api.minimaxi.com/v1/', 'https://api.minimaxi.com/anthropic/v1'),
        ('openai', 'gpt-5.5', 'https://api.openai.com/', 'https://api.openai.com/v1'),
        ('qwen', 'qwen-plus', 'https://dashscope.aliyuncs.com/', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
    ],
)
def test_known_provider_urls_are_rewritten(provider, model, base_url, expected_url):
    settings = build_repair_opencode_settings(_role(provider, model, base_url=base_url))
    assert settings['base_url'] == expected_url


def test_custom_url_and_skip_auth_are_preserved():
    settings = build_repair_opencode_settings({
        'source': 'SenseNova',
        'model': 'SenseChat-5',
        'base_url': 'https://gateway.example/v1/',
        'skip_auth': True,
    })

    assert settings['base_url'] == 'https://gateway.example/v1'
    assert settings['api_key'] == ''
    assert settings['skip_auth'] == 'true'


@pytest.mark.parametrize(
    ('role', 'code', 'reason', 'missing'),
    [
        (None, MODEL_NOT_CONFIGURED, 'model_not_configured', ()),
        ({}, MODEL_NOT_CONFIGURED, 'model_config_incomplete', ('source', 'model', 'base_url', 'api_key')),
        (_role('openai', 'not-a-model'), MODEL_NOT_ALLOWED, 'evo_llm_not_allowed', ()),
        (_role('unknown', 'anything'), MODEL_NOT_ALLOWED, 'evo_llm_not_allowed', ()),
    ],
)
def test_invalid_roles_return_stable_errors(role, code, reason, missing):
    with pytest.raises(EvoModelConfigError) as raised:
        build_repair_opencode_settings(role)

    error = raised.value
    assert error.code == code
    assert error.reason == reason
    assert error.missing_fields == missing
