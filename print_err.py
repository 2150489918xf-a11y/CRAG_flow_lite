import sys
try:
    with open('test.log', 'r', encoding='utf-16le') as f:
        content = f.read()
        
    start_idx = content.find('Traceback (most recent call last)')
    if start_idx == -1:
        start_idx = content.find('ERROR:rag.parser.pdf_parser')
        
    if start_idx != -1:
        important_part = content[start_idx:]
    else:
        important_part = content[-2000:]
        
    with open('traceback_clean.txt', 'w', encoding='utf-8') as f:
        f.write(important_part)
except Exception as e:
    with open('traceback_clean.txt', 'w', encoding='utf-8') as f:
        f.write(str(e))
