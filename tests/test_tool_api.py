"""
Agent Tool API 测试
- ToolRetrieveRequest / ToolRetrieveResponse 模型验证
- TOOL_SCHEMA 结构验证
- /api/tool/schema 端点
- /api/tool/list_kbs 端点
- /api/tool/retrieve 输入验证
"""
import pytest
from fastapi.testclient import TestClient


# ──────────────── 模型测试 ────────────────

class TestToolModels:
    """Tool API Pydantic 模型测试"""

    def test_retrieve_request_defaults(self):
        from api.models import ToolRetrieveRequest
        req = ToolRetrieveRequest(query="测试问题")
        assert req.query == "测试问题"
        assert req.kb_ids == []
        assert req.top_k == 5
        assert req.mode == "hybrid"
        assert req.folder == ""

    def test_retrieve_request_all_params(self):
        from api.models import ToolRetrieveRequest
        req = ToolRetrieveRequest(
            query="微软投资多少",
            kb_ids=["finance_kb"],
            top_k=10,
            mode="deep",
            folder="/财务",
        )
        assert req.mode == "deep"
        assert req.folder == "/财务"
        assert req.top_k == 10

    def test_tool_source_model(self):
        from api.models import ToolSource
        src = ToolSource(id="chunk_1", content="内容", doc_name="报告.pdf",
                         source_type="local", relevance_score=0.85)
        assert src.source_type == "local"
        assert src.relevance_score == 0.85

    def test_tool_source_defaults(self):
        from api.models import ToolSource
        src = ToolSource(id="c1", content="text")
        assert src.doc_name == ""
        assert src.source_type == "local"
        assert src.relevance_score == 0.0

    def test_tool_metadata(self):
        from api.models import ToolMetadata
        meta = ToolMetadata(mode="fast", total_hits=100, source_count=5,
                            latency_ms=42)
        assert meta.mode == "fast"
        assert meta.total_hits == 100
        assert meta.crag_score == ""

    def test_tool_retrieve_response(self):
        from api.models import ToolRetrieveResponse, ToolSource, ToolMetadata
        resp = ToolRetrieveResponse(
            answer_context="这是上下文",
            sources=[ToolSource(id="1", content="chunk1")],
            metadata=ToolMetadata(mode="hybrid"),
        )
        assert len(resp.sources) == 1
        assert resp.answer_context == "这是上下文"

    def test_retrieve_request_invalid_empty_query(self):
        """空 query 应通过模型验证（端点侧校验）"""
        from api.models import ToolRetrieveRequest
        req = ToolRetrieveRequest(query="")
        assert req.query == ""


# ──────────────── Schema 测试 ────────────────

class TestToolSchema:
    """OpenAI Function Calling Schema 结构测试"""

    def test_schema_structure(self):
        from api.routes.tool import TOOL_SCHEMA
        assert TOOL_SCHEMA["type"] == "function"
        func = TOOL_SCHEMA["function"]
        assert func["name"] == "rag_retrieve"
        assert "description" in func
        assert "parameters" in func

    def test_schema_parameters(self):
        from api.routes.tool import TOOL_SCHEMA
        params = TOOL_SCHEMA["function"]["parameters"]
        assert params["type"] == "object"
        props = params["properties"]
        assert "query" in props
        assert "kb_ids" in props
        assert "top_k" in props
        assert "mode" in props
        assert "folder" in props

    def test_schema_required_fields(self):
        from api.routes.tool import TOOL_SCHEMA
        required = TOOL_SCHEMA["function"]["parameters"]["required"]
        assert "query" in required

    def test_schema_mode_enum(self):
        from api.routes.tool import TOOL_SCHEMA
        mode_prop = TOOL_SCHEMA["function"]["parameters"]["properties"]["mode"]
        assert set(mode_prop["enum"]) == {"fast", "hybrid", "deep"}


# ──────────────── 端点测试 (TestClient) ────────────────

class TestToolEndpoints:
    """Tool API 端点集成测试 (使用 FastAPI TestClient，不需要 ES)"""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from api.app import app
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_schema_endpoint(self):
        resp = self.client.get("/api/tool/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        schema = data["data"]
        assert schema["function"]["name"] == "rag_retrieve"

    def test_retrieve_empty_query(self):
        """空 query 不应返回成功"""
        resp = self.client.post("/api/tool/retrieve", json={
            "query": "   ",
        })
        assert resp.status_code != 200

    def test_retrieve_invalid_mode(self):
        """无效 mode 不应返回成功"""
        resp = self.client.post("/api/tool/retrieve", json={
            "query": "测试",
            "mode": "invalid_mode",
        })
        assert resp.status_code != 200

    def test_retrieve_missing_query(self):
        """缺少必需字段 query 应返回 422"""
        resp = self.client.post("/api/tool/retrieve", json={
            "kb_ids": ["test"],
        })
        assert resp.status_code == 422
