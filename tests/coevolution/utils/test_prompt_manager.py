import pytest
import os
from pathlib import Path
from coevolution.utils.prompt_manager import PromptManager

def test_prompt_manager_single_dir(tmp_path: Path) -> None:
    """Verify loading from a single custom directory."""
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "test.j2").write_text("Hello {{ name }}!")
    
    pm = PromptManager(template_dirs=str(d))
    assert pm.render_prompt("test.j2", name="World") == "Hello World!"

def test_prompt_manager_multi_dir_priority(tmp_path: Path) -> None:
    """Verify prioritized loading and fallback across multiple directories."""
    # Dir 1: Higher priority
    d1 = tmp_path / "pop_prompts"
    d1.mkdir()
    (d1 / "test.j2").write_text("Pop: {{ name }}")
    
    # Dir 2: Lower priority (fallback)
    d2 = tmp_path / "global_prompts"
    d2.mkdir()
    (d2 / "test.j2").write_text("Global: {{ name }}")
    (d2 / "other.j2").write_text("Other: {{ name }}")
    
    # Priority to d1
    pm = PromptManager(template_dirs=[str(d1), str(d2)])
    
    # Should load 'test.j2' from d1 (shadowing d2)
    assert pm.render_prompt("test.j2", name="X") == "Pop: X"
    
    # Should load 'other.j2' from d2 (fallback)
    assert pm.render_prompt("other.j2", name="Y") == "Other: Y"

def test_prompt_manager_language_injection(tmp_path: Path) -> None:
    """Verify that language context is correctly injected or overridden."""
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "test.j2").write_text("Lang: {{ language }}")
    
    # Default injection
    pm = PromptManager(template_dirs=str(d), language="java")
    assert pm.render_prompt("test.j2") == "Lang: java"
    
    # Explicit override in render call
    assert pm.render_prompt("test.j2", language="c++") == "Lang: c++"

def test_prompt_manager_default_dir() -> None:
    """Verify that the default initialization still works."""
    pm = PromptManager()
    # Check that loader is initialized (path will be internal to the project)
    assert pm.env.loader is not None
    assert hasattr(pm.env.loader, "searchpath")
    assert len(pm.env.loader.searchpath) > 0

def test_prompt_manager_error_handling(tmp_path: Path) -> None:
    """Verify that rendering errors are caught and wrapped."""
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "bad.j2").write_text("{{ 1 / 0 }}")
    
    pm = PromptManager(template_dirs=str(d))
    with pytest.raises(RuntimeError, match="Error rendering template 'bad.j2'"):
        pm.render_prompt("bad.j2")
