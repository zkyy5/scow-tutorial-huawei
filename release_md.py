import os
import subprocess
from bs4 import BeautifulSoup
import shutil

target_directory = './release_web'

HTML_CONTENT_PLACE_HOLDER = 'HTML_CONTENT_PLACE_HOLDER'
html_template = f"""<html>
<head>
<meta charset="utf-8"/></head>
<style>
img {{
  max-width:700px;
  padding:14px;
}}
</style>
<body>
<div style="margin:0 auto;max-width:800px">
{HTML_CONTENT_PLACE_HOLDER}
</div>
</body>
</html>"""

def markdown_to_html(markdown_text):
    process = subprocess.run(
        ['pandoc', '--from=markdown', '--to=html', '--highlight-style=pygments'],
        input=markdown_text.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    
    html_output = process.stdout.decode('utf-8')
    return html_output

def convert_md_to_html(md_file_path, html_file_path):
    # 读取 Markdown 文件内容
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # 使用 markdown 库将 Markdown 转换为 HTML
    html_content = markdown_to_html(md_content)

    # 使用 BeautifulSoup 解析 HTML 内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 修改所有 <a> 标签中的链接后缀
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.ipynb'):
            a_tag['href'] = href[:-6] + '.html'  # 修改为 .html 后缀

    # 将修改后的 HTML 内容写入文件
    with open(html_file_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html_template.replace(HTML_CONTENT_PLACE_HOLDER, str(soup)))

def process_all_md_files_in_directory(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.md') and filename != 'README.md':
            md_file_path = os.path.join(directory, filename)
            html_file_path = os.path.join(directory, filename[:-3] + '.html')
            
            # 转换并修改链接
            convert_md_to_html(md_file_path, html_file_path)
            print(f'Converted {md_file_path} to {html_file_path}')
            target_html_path = os.path.join(target_directory, filename[:-3] + '.html')
            shutil.move(html_file_path, target_html_path)

# 执行处理当前目录下的所有 .md 文件
if __name__ == '__main__':
    current_directory = os.getcwd()
    process_all_md_files_in_directory(current_directory)
