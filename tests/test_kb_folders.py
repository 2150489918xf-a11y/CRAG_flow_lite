"""
知识库文件夹管理测试
- 路径标准化
- 文件夹 CRUD
- 树结构构建
- 文件夹过滤
"""
import json
import os
import tempfile
import pytest


# ──────────────── 路径标准化测试 ────────────────

class TestFolderNormalization:
    """虚拟路径标准化逻辑测试"""

    def test_normalize_root(self):
        from api.routes.kb import _normalize_folder
        assert _normalize_folder("/") == "/"

    def test_normalize_trailing_slash(self):
        from api.routes.kb import _normalize_folder
        assert _normalize_folder("/财务/") == "/财务"

    def test_normalize_no_leading_slash(self):
        from api.routes.kb import _normalize_folder
        assert _normalize_folder("财务") == "/财务"

    def test_normalize_nested(self):
        from api.routes.kb import _normalize_folder
        assert _normalize_folder("/财务/政策法规/") == "/财务/政策法规"

    def test_normalize_backslash(self):
        from api.routes.kb import _normalize_folder
        assert _normalize_folder("\\财务\\法规") == "/财务/法规"

    def test_normalize_spaces(self):
        from api.routes.kb import _normalize_folder
        assert _normalize_folder("  /test  ") == "/test"


# ──────────────── 祖先路径自动创建 ────────────────

class TestFolderAncestors:
    """祖先文件夹自动创建测试"""

    def test_ancestors_created(self):
        from api.routes.kb import _ensure_folder_ancestors
        folders = {"/"}
        _ensure_folder_ancestors("/a/b/c", folders)
        assert "/a" in folders
        assert "/a/b" in folders
        assert "/a/b/c" in folders

    def test_ancestors_root_noop(self):
        from api.routes.kb import _ensure_folder_ancestors
        folders = {"/"}
        _ensure_folder_ancestors("/", folders)
        # 根目录路径为空，不应额外创建
        assert folders == {"/"}


# ──────────────── 文件夹持久化测试 ────────────────

class TestFolderPersistence:
    """文件夹 JSON 持久化测试"""

    def test_save_and_load(self):
        from api.routes.kb import _save_folders, _load_folders, _FOLDERS_FILE
        import api.routes.kb as kb_module

        tmp = tempfile.mktemp(suffix=".json")
        original = kb_module._FOLDERS_FILE
        try:
            kb_module._FOLDERS_FILE = tmp
            _save_folders({"/", "/test", "/test/sub"})

            loaded = _load_folders()
            assert "/" in loaded
            assert "/test" in loaded
            assert "/test/sub" in loaded
        finally:
            kb_module._FOLDERS_FILE = original
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_load_missing_file(self):
        import api.routes.kb as kb_module
        original = kb_module._FOLDERS_FILE
        try:
            kb_module._FOLDERS_FILE = "/nonexistent/path.json"
            from api.routes.kb import _load_folders
            folders = _load_folders()
            assert "/" in folders  # 至少有根目录
        finally:
            kb_module._FOLDERS_FILE = original


# ──────────────── 端点测试 (TestClient) ────────────────

class TestFolderEndpoints:
    """文件夹管理端点集成测试"""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from api.app import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_create_folder_root_fails(self):
        """创建根目录不应返回成功"""
        resp = self.client.post("/api/knowledgebase/folder", json={
            "path": "/",
        })
        assert resp.status_code != 200

    def test_delete_root_fails(self):
        """删除根目录不应返回成功"""
        resp = self.client.delete("/api/knowledgebase/folder", params={
            "path": "/",
        })
        assert resp.status_code != 200

    def test_move_missing_kb(self):
        """移动不存在的 KB 应返回 404"""
        resp = self.client.post("/api/knowledgebase/move", json={
            "kb_id": "nonexistent_kb_12345",
            "target_folder": "/test",
        })
        assert resp.status_code == 404
