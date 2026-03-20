"""
文档解析器测试：TXT, JSON, HTML, Markdown 解析
"""
import json
import pytest


class TestTxtParser:
    """TXT 解析器测试"""

    def test_parse_basic(self):
        from rag.parser.other_parsers import parse_txt
        content = "第一段落\n\n第二段落\n\n第三段落"
        sections, tables = parse_txt("test.txt", content.encode("utf-8"))
        assert len(sections) == 3
        assert len(tables) == 0

    def test_parse_empty(self):
        from rag.parser.other_parsers import parse_txt
        sections, tables = parse_txt("empty.txt", b"")
        assert len(sections) == 0


class TestJsonParser:
    """JSON 解析器测试"""

    def test_parse_dict(self):
        from rag.parser.other_parsers import parse_json
        data = json.dumps({"name": "RAGFlow", "version": "1.0"})
        sections, _ = parse_json("test.json", data.encode("utf-8"))
        assert len(sections) > 0

    def test_parse_list(self):
        from rag.parser.other_parsers import parse_json
        data = json.dumps([{"a": 1}, {"b": 2}])
        sections, _ = parse_json("test.json", data.encode("utf-8"))
        assert len(sections) > 0


class TestMarkdownParser:
    """Markdown 解析器测试"""

    def test_parse_headers(self):
        from rag.parser.markdown_parser import parse
        md = "# 标题一\n内容一\n## 标题二\n内容二\n### 标题三\n内容三"
        sections, _ = parse("test.md", md.encode("utf-8"))
        assert len(sections) > 0

    def test_parse_empty(self):
        from rag.parser.markdown_parser import parse
        sections, _ = parse("empty.md", b"")
        assert len(sections) == 0


class TestHtmlParser:
    """HTML 解析器测试"""

    def test_parse_basic(self):
        from rag.parser.other_parsers import parse_html
        html = "<html><body><h1>Title</h1><p>Content paragraph</p></body></html>"
        sections, tables = parse_html("test.html", html.encode("utf-8"))
        assert len(sections) > 0

    def test_parse_with_table(self):
        from rag.parser.other_parsers import parse_html
        html = "<html><body><table><tr><td>A</td><td>B</td></tr></table></body></html>"
        sections, tables = parse_html("test.html", html.encode("utf-8"))
        # 至少能解析不报错
        assert isinstance(sections, list)
