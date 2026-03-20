"""
生成一个用于测试 DeepDoc 解析的包含正文和表格的简单 PDF
需要 reportlab
"""
import os
import sys

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
except ImportError:
    print("Please run: pip install reportlab")
    sys.exit(1)

def create_test_pdf(filename="test_doc.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # 标题和正文
    elements.append(Paragraph("测试 DeepDoc 深度视觉解析引擎", styles['Title']))
    elements.append(Paragraph("第一章：大模型与RAG技术简介", styles['Heading1']))
    elements.append(Paragraph("检索增强生成（RAG）是一种结合了信息检索和文本生成的技术。它通过在外挂知识库中检索相关文档，极大地缓解了大模型的幻觉问题，并使其能够回答基于专业领域知识的问题。", styles['Normal']))
    
    # 一个空行
    elements.append(Paragraph("<br/>", styles['Normal']))
    
    # 表格
    data = [
        ['模型名称', '参数量', '上下文长度', '适用场景'],
        ['Llama-3-8B', '8 Billion', '8K', '轻量级本地部署'],
        ['Qwen-2-72B', '72 Billion', '128K', '企业级核心推理'],
        ['DeepSeek-V3', '671 Billion', '128K', '高性价比复杂任务']
    ]
    
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    
    elements.append(t)
    
    # 表格后正文
    elements.append(Paragraph("<br/>", styles['Normal']))
    elements.append(Paragraph("如上表所示，不同模型在规模和上下文能力上存在显著差异。在企业级 RAG 系统中，通常使用 72B 级别的高性能模型来实现跨文档推理。在 RAGFlow 系统中，DeepDoc 解析器负责将复杂的排版（如双栏和类似上表的复杂表格）精确还原为结构化信息。", styles['Normal']))
    
    doc.build(elements)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_test_pdf(os.path.join(os.path.dirname(__file__), "test_doc.pdf"))
