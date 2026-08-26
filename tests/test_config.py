"""配置层测试：默认回落、用户覆盖、往返一致、初始化判定。"""
from flowmind.config import (
    InventoryConfig, OrchestratorConfig, FlowmindConfig, load_config, save_config, is_initialized,
)


def test_orchestrator_config_default():
    cfg = OrchestratorConfig()
    assert cfg.llm_key_env == "LONGCAT_API_KEY"
    assert cfg.max_plan_steps == 5
    assert cfg.max_retries_per_step == 1
    assert cfg.enable_streaming is True


def test_flowmind_config_includes_orchestrator():
    cfg = FlowmindConfig()
    assert cfg.orchestrator.llm_model == "LongCat-2.0"


def test_defaults_when_no_file(tmp_path):
    path = tmp_path / "flowmind.config.toml"
    cfg = load_config(path)
    assert cfg.inventory.dsi_healthy_max == 60.0
    assert cfg.inventory.currency == "USD"

def test_is_initialized_reflects_file(tmp_path):
    path = tmp_path / "flowmind.config.toml"
    assert is_initialized(path) is False
    save_config(FlowmindConfig(), path)
    assert is_initialized(path) is True

def test_user_values_override_defaults(tmp_path):
    path = tmp_path / "flowmind.config.toml"
    cfg = FlowmindConfig(inventory=InventoryConfig(dsi_healthy_max=30.0, currency="CNY"))
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.inventory.dsi_healthy_max == 30.0
    assert loaded.inventory.currency == "CNY"
    # 未指定的字段仍回落默认
    assert loaded.inventory.dsi_warn_max == 120.0

def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "flowmind.config.toml"
    cfg = FlowmindConfig(inventory=InventoryConfig(capital_high=50000.0))
    save_config(cfg, path)
    assert load_config(path).inventory.capital_high == 50000.0
