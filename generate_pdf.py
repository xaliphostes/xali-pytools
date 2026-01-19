import markdown
import os
import re
from weasyprint import HTML

def resolve_md_links(content, base_path):
    """Convert relative .md links to anchors"""
    # Pattern to find markdown links to .md files
    pattern = r'\[([^\]]+)\]\(([^)]+\.md)\)'
    return re.sub(pattern, r'[\1](#\2)', content)

def collect_md_files(main_file):
    """Recursively collect all linked markdown files"""
    collected = {}
    to_process = [main_file]
    base_dir = os.path.dirname(os.path.abspath(main_file))
    
    while to_process:
        current = to_process.pop(0)
        if current in collected:
            continue
            
        filepath = os.path.join(base_dir, current) if not os.path.isabs(current) else current
        
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        collected[current] = content
        
        # Find linked .md files
        links = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', content)
        for _, link in links:
            # Resolve relative path
            link_dir = os.path.dirname(current)
            resolved = os.path.normpath(os.path.join(link_dir, link))
            if resolved not in collected:
                to_process.append(resolved)
    
    return collected

def convert_to_pdf(main_file, output_pdf):
    files = collect_md_files(main_file)
    
    # Combine all content
    combined_md = ""
    for filepath, content in files.items():
        # Add anchor for this file
        combined_md += f'<a id="{filepath}"></a>\n\n'
        # Resolve internal links
        content = resolve_md_links(content, filepath)
        combined_md += content + "\n\n---\n\n"
    
    # Convert to HTML
    html_content = markdown.markdown(
        combined_md, 
        extensions=['tables', 'fenced_code', 'toc']
    )
    
    # Add basic styling
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; margin: 40px; }}
            code {{ background: #f4f4f4; padding: 2px 5px; }}
            pre {{ background: #f4f4f4; padding: 10px; }}
            hr {{ margin: 30px 0; }}
        </style>
    </head>
    <body>{html_content}</body>
    </html>
    """
    
    HTML(string=styled_html).write_pdf(output_pdf)

# Usage
convert_to_pdf('README.md', 'output.pdf')