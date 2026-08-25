"""localize_video 技能:注册 / 入参校验 / 错误分类路径。

不跑真 VLE (paddleocr/torch 太重),只验:
- @skill 注册成功,input_model 正确
- video_path 为空字符串被 validator 拒绝
- 不存在的本地文件 → degraded SkillOutput (failure_category='video')
- 输入校验失败 (空字符串) → invoke() 包 VALIDATION 错误
- _classify_exception 对未知异常 → 'unknown'
"""
from __future__ import annotations

import pytest

from flowmind.errors import _classify_exception, is_retriable, FailureCategory
from flowmind.skill import registry, invoke
from flowmind.skills.localize_video import (
    LocalizeVideoInput,
    LocalizeVideoReport,
    localize_video,
)


def test_registered():
    spec = registry().get("localize_video")
    assert spec is not None
    assert spec.id == "localize_video"
    assert spec.input_model is LocalizeVideoInput
    assert spec.version == "0.1.0"


def test_video_path_blank_rejected():
    """空白路径会被 Pydantic validator 拒绝。"""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        LocalizeVideoInput(video_path="   ")


def test_video_path_empty_rejected():
    """空字符串也会被拒绝 (min_length=1 + validator 都拒)。"""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        LocalizeVideoInput(video_path="")


def test_nonexistent_video_returns_degraded(tmp_path):
    """本地文件不存在 → degraded SkillOutput,failure_category=video。"""
    fake = tmp_path / "nope.mp4"
    out = localize_video(LocalizeVideoInput(video_path=str(fake)))
    assert isinstance(out.data, LocalizeVideoReport)
    assert out.data.degraded is True
    assert out.data.failure_category == FailureCategory.VIDEO.value
    assert out.data.output_path is None
    assert out.degraded is True
    assert out.degradation_reason == FailureCategory.VIDEO.value


def test_invoke_wraps_validation_error():
    """invoke 包装层应把 Pydantic ValidationError 转成 NOT_FOUND/VALIDATION。"""
    # 先确认一个错的 (无效 enum,例如把 target_lang 写成数字) 能进 invoke 路径
    # LocalizeVideoInput.target_lang 是 str | None;用空字符串会被 validator 拒绝
    res = invoke("localize_video", {"video_path": ""})
    assert res.ok is False
    assert res.error.code == "VALIDATION"


def test_classify_unknown_exception():
    class WeirdError(Exception):
        pass

    assert _classify_exception(WeirdError("nothing special")) == FailureCategory.UNKNOWN.value


def test_classify_timeout_in_message():
    assert _classify_exception(Exception("request timeout after 30s")) == FailureCategory.ENVIRONMENT.value


def test_classify_video_file_not_found():
    assert _classify_exception(Exception("Video file not found")) == FailureCategory.VIDEO.value


def test_is_retriable():
    assert is_retriable(FailureCategory.TRANSIENT.value) is True
    assert is_retriable(FailureCategory.ENVIRONMENT.value) is False
    assert is_retriable(FailureCategory.VIDEO.value) is False
    assert is_retriable(FailureCategory.UNKNOWN.value) is False


def test_localize_video_input_optional_fields():
    """target_lang=None 等可选字段都能正常实例化。"""
    inp = LocalizeVideoInput(video_path="/tmp/anything.mp4")
    assert inp.target_lang is None
    assert inp.enable_tts is None
    assert inp.mask_dilation_y is None