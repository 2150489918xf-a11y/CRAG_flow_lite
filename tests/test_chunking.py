"""
分块引擎测试：9 种策略 + Parent-Child 机制 + PPT 智能路由
"""
import pytest


class TestChunkingBasic:
    """基础分块流程测试"""

    def test_txt_chunking(self, tmp_txt_file):
        from rag.app.chunking import chunk
        chunks = chunk(tmp_txt_file, lang="Chinese")
        assert len(chunks) > 0
        for ck in chunks:
            assert "id" in ck
            assert "docnm_kwd" in ck
            assert "content_ltks" in ck
            assert "content_with_weight" in ck

    def test_empty_file(self, tmp_file_factory):
        from rag.app.chunking import chunk
        path = tmp_file_factory("", suffix=".txt")
        chunks = chunk(path, lang="Chinese")
        assert len(chunks) == 0


class TestChunkingStrategies:
    """9 种分块策略逐一验证"""

    @pytest.mark.parametrize("parser_id", [
        "naive", "qa", "laws", "one",
        "book", "paper", "presentation", "table", "manual",
    ])
    def test_strategy_runs_without_error(self, tmp_file_factory, parser_id):
        """每种策略都应能处理文本文件且不抛异常"""
        content = "这是一段测试文本。" * 50 + "\n\n第二段文字内容。" * 50
        path = tmp_file_factory(content, suffix=".txt")
        from rag.app.chunking import chunk
        chunks = chunk(path, lang="Chinese", parser_config={"parser_id": parser_id})
        assert isinstance(chunks, list), f"策略 {parser_id} 应返回 list"

    def test_qa_strategy(self, tmp_file_factory):
        """QA 策略应能处理 Q/A 格式文本"""
        qa_text = "Q: 什么是RAG？\nA: 检索增强生成\n\nQ: 什么是LLM？\nA: 大语言模型"
        path = tmp_file_factory(qa_text, suffix=".txt")
        from rag.app.chunking import chunk
        chunks = chunk(path, lang="Chinese", parser_config={"parser_id": "qa"})
        assert isinstance(chunks, list)

    def test_laws_strategy(self, tmp_file_factory, sample_legal):
        """法律策略应能处理法律条文"""
        path = tmp_file_factory(sample_legal, suffix=".txt")
        from rag.app.chunking import chunk
        chunks = chunk(path, lang="Chinese", parser_config={"parser_id": "laws"})
        assert isinstance(chunks, list)

    def test_one_strategy_single_chunk(self, tmp_file_factory):
        """ONE 策略应当只产出 1 个 chunk"""
        content = "这是一小段文本。"
        path = tmp_file_factory(content, suffix=".txt")
        from rag.app.chunking import chunk
        chunks = chunk(path, lang="Chinese", parser_config={"parser_id": "one"})
        assert len(chunks) == 1


class TestParentChildChunking:
    """Parent-Child 分层分块测试"""

    def test_parent_child_structure(self, tmp_file_factory):
        from rag.app.chunking import chunk
        # 需要足够长的文本才能触发分层分块
        text = "\n\n".join([("这是一段测试文本，包含多个句子。" * 80)] * 5)
        path = tmp_file_factory(text, suffix=".txt")
        chunks = chunk(path, lang="Chinese", parser_config={
            "use_parent_child": True,
            "parent_token_num": 256,
            "child_token_num": 64,
        })
        assert len(chunks) > 0
        # 检查是否有 parent/child 类型标记
        types = {c.get("chunk_type_kwd") for c in chunks}
        # 如果文本足够长, 应该有 parent 和 child
        if len(chunks) > 1:
            assert "parent" in types or "child" in types, \
                f"启用 parent-child 后应有类型标记, 实际: {types}"

    def test_child_refs_valid_parent(self, tmp_txt_file):
        """每个 child 的 parent_id_kwd 都必须指向一个真实存在的 parent"""
        from rag.app.chunking import chunk
        chunks = chunk(tmp_txt_file, lang="Chinese", parser_config={
            "use_parent_child": True,
            "parent_token_num": 512,
            "child_token_num": 128,
        })
        parent_ids = {c["id"] for c in chunks if c.get("chunk_type_kwd") == "parent"}
        children = [c for c in chunks if c.get("chunk_type_kwd") == "child"]

        for child in children:
            assert "parent_id_kwd" in child
            assert child["parent_id_kwd"] in parent_ids

    def test_parent_has_no_parent_ref(self, tmp_txt_file):
        """parent chunks 不应有 parent_id_kwd 字段"""
        from rag.app.chunking import chunk
        chunks = chunk(tmp_txt_file, lang="Chinese", parser_config={
            "use_parent_child": True,
        })
        parents = [c for c in chunks if c.get("chunk_type_kwd") == "parent"]
        for p in parents:
            assert "parent_id_kwd" not in p


class TestSmartRouting:
    """PPT 智能路由拦截测试"""

    def test_ppt_extension_forces_presentation(self, tmp_file_factory):
        """即使 parser_id=naive，.pptx 文件也应路由到 presentation 策略"""
        content = "这是幻灯片内容"
        path = tmp_file_factory(content, suffix=".pptx")
        from rag.app.chunking import chunk
        # 不应报错（presentation 策略会被自动选用）
        chunks = chunk(path, lang="Chinese", parser_config={"parser_id": "naive"})
        assert isinstance(chunks, list)
