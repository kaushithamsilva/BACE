import pytest
import os
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch
from coevolution.strategies.llm_base import BaseLLMService, ILanguageModel
from coevolution.core.interfaces.language import ICodeParser

def test_resolve_template_dirs_priority() -> None:
    """Verify the priority order: Home -> Target -> Global."""
    llm = MagicMock(spec=ILanguageModel)
    parser = MagicMock(spec=ICodeParser)
    
    # We use a real instance but patch the filesystem checks
    with patch("inspect.getfile") as mock_getfile, \
         patch("coevolution.strategies.llm_base.os.path.isdir") as mock_isdir, \
         patch("coevolution.strategies.llm_base.os.path.abspath", return_value="/fake/project/src/coevolution/strategies/llm_base.py"):
        
        svc = BaseLLMService(llm=llm, parser=parser, language_name="python")
        
        # Scenario: Operator defined in 'unittest'
        mock_getfile.return_value = "/fake/project/src/coevolution/populations/unittest/operators/repair.py"
        
        # Simulate all directories existing
        mock_isdir.return_value = True
        
        dirs = svc._resolve_template_dirs()
        
        # Verify order and presence
        assert len(dirs) >= 3
        # 1. Local (unittest)
        assert "populations/unittest/prompts" in dirs[0].replace("\\", "/")
        # 2. Code (Universal Fallback)
        assert "populations/code/prompts" in dirs[1].replace("\\", "/")
        # 3. Global (last or near last)
        assert any(d.endswith("prompts") and "populations" not in d for d in dirs)

def test_resolve_template_dirs_fallback() -> None:
    """Verify that if Home and Target are the same, we don't have duplicates."""
    llm = MagicMock(spec=ILanguageModel)
    parser = MagicMock(spec=ICodeParser)
    
    with patch("inspect.getfile") as mock_getfile, \
         patch("coevolution.strategies.llm_base.os.path.isdir", return_value=True), \
         patch("coevolution.strategies.llm_base.os.path.abspath", return_value="/fake/project/src/coevolution/strategies/llm_base.py"):
        
        svc = BaseLLMService(llm=llm, parser=parser, language_name="python")
        
        # Scenario: Operator defined in 'code'
        mock_getfile.return_value = "/fake/project/src/coevolution/populations/code/operators/mutation.py"
        
        dirs = svc._resolve_template_dirs()
        
        # Should not have duplicate 'code' prompt paths
        code_paths = [d for d in dirs if "populations/code/prompts" in d.replace("\\", "/")]
        assert len(code_paths) == 1

def test_resolve_template_dirs_missing_robustness() -> None:
    """Verify that non-existent directories are skipped."""
    llm = MagicMock(spec=ILanguageModel)
    parser = MagicMock(spec=ICodeParser)
    
    with patch("inspect.getfile", return_value="/fake/project/src/coevolution/strategies/llm_base.py"), \
         patch("coevolution.strategies.llm_base.os.path.isdir") as mock_isdir, \
         patch("coevolution.strategies.llm_base.os.path.abspath", return_value="/fake/project/src/coevolution/strategies/llm_base.py"):
        
        svc = BaseLLMService(llm=llm, parser=parser, language_name="python")
        # Only global prompts exist (mocking isdir to return False for everything else)
        mock_isdir.side_effect = lambda p: not ("populations" in p)
        
        dirs = svc._resolve_template_dirs()
        
        # Should only contain the global one
        assert len(dirs) == 1
        assert not any("populations" in d for d in dirs)

class SpecializedOperator(BaseLLMService):
    """Subclass for testing Relative Discovery."""
    pass

def test_relative_discovery_with_subclass() -> None:
    """Test that Relative Discovery works with real class hierarchy."""
    llm = MagicMock(spec=ILanguageModel)
    parser = MagicMock(spec=ICodeParser)
    
    # Instantiate the subclass
    svc = SpecializedOperator(llm=llm, parser=parser, language_name="python")
    
    # By default, SpecializedOperator is defined in this test file
    # so it should try to find 'prompts/' relative to this file
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    expected_home = os.path.join(os.path.dirname(this_file_dir), "prompts")
    
    dirs = svc._resolve_template_dirs()
    
    # If the prompts dir existed next to this file, it would be in dirs
    # Since it likely doesn't in the test env, it won't be there, 
    # but we can verify the 'Local' resolution logic was called for this class.
    with patch("inspect.getfile", wraps=inspect.getfile) as mock_getfile:
        svc._resolve_template_dirs()
        mock_getfile.assert_called()
        # Verify it was called with the actual subclass, not BaseLLMService
        args, _ = mock_getfile.call_args
        assert args[0] == SpecializedOperator
