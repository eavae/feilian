## Goal: Test the XPath for university_embark category

## HTML data file 
## /Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/university
## /Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/university/university-collegeboard\(2000\)/


## 数据探索：网页数据
import os
# 指定文件夹路径
htm_path = "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/university/university-collegeboard(2000)/" # 2000 htm files 

# 统计 .html 文件数量
html_file_count = sum([1 for file in os.listdir(htm_path) if file.endswith(".htm")])
print(f"该文件夹中有 {html_file_count} 个 htm 文件。")



## 数据探索：groundtruth 数据
gtruth_path = "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/groundtruth/university/" # 2000 txt files 
## Groundtruth: university-embark-name.txt: /Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-embark-name.txt
import pandas as pd

# 定义一个函数来读取文件并删除第一行
def load_and_process_file(file_path):
    # 读取文件
    df = pd.read_csv(file_path, sep=None, engine='python')  # 使用 engine='python' 进行灵活解析
    # 删除第一行并重置索引
    df = df.iloc[1:].reset_index(drop=True)
    return df

# 文件路径
file_paths = {
    "name": "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-embark-name.txt",
    "phone": "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-embark-phone.txt",
    "type": "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-embark-type.txt",
    "website": "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-embark-website.txt"}

# 使用函数处理每个文件
df_embark_name = load_and_process_file(file_paths['name'])
df_embark_phone = load_and_process_file(file_paths['phone'])
df_embark_type = load_and_process_file(file_paths['type'])
df_embark_website = load_and_process_file(file_paths['website'])


#print(df_embark_website.head(), end=("\n"))
# 通过 pd.concat 按列合并，指定 axis=1
df_gtruth = pd.concat([df_embark_name['embark'], 
                       df_embark_phone['embark'], 
                       df_embark_type['embark'], 
                       df_embark_website['embark']], 
                      axis=1)

# 重新命名列
df_gtruth.columns = ['name', 'phone', 'type', 'website']

# 显示合并后的 DataFrame
print(df_gtruth.head()) # 2000x4
print()


## XPath File Path
# 指定 CSV 文件路径
xpath_path = "/Users/brycewang/Desktop/智能爬虫/data/ranked_xpaths.csv"
# 读取 CSV 文件
df_xpath = pd.read_csv(xpath_path)
print(df_xpath.head())

xpath_name = df_xpath['xpath'].iloc[0] # xpath_name = "//title/text()" 
xpath_phone = df_xpath['xpath'].iloc[1] # xpath_phone = "//td[@class='tdContent']//span[@class='label' and text()='Phone:']/following-sibling::span[@class='data']/text()"
xpath_type = df_xpath['xpath'].iloc[2] # xpath_type ="//td[@class='tdContent']/span/table[@class='details']/tbody/tr/td[@class='label' and text()='School Type:']/following-sibling::td[@class='data'][1]/text()"
xpath_website = df_xpath['xpath'].iloc[3] # xpath_website ="//td[@class='tdContent']//span[@class='label' and text()='Website:']/following-sibling::span[@class='data']/a/@href"
print()
print(xpath_name,"\n", xpath_phone,"\n",  xpath_type,"\n",  xpath_website, end=("\n"))


## 数据检索 with XPath
import os
from lxml import etree

# 设置 HTML 文件所在的目录路径
htm_path = "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/university/university-collegeboard(2000)/"

# 定义要使用的 XPath 表达式
xpaths = {
    'name': df_xpath['xpath'].iloc[0],
    'phone': df_xpath['xpath'].iloc[1],
    'type': df_xpath['xpath'].iloc[2],
    'website': df_xpath['xpath'].iloc[3]
}

# 创建一个空的列表来存储结果
data = []

# 遍历文件夹中的所有 HTML 文件
for filename in os.listdir(htm_path):
    if filename.endswith(".htm") or filename.endswith(".html"):
        file_path = os.path.join(htm_path, filename)

        # 解析 HTML 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            tree = etree.HTML(content)

            # 从 HTML 文件中提取数据
            row_data = {'filename': filename}
            for field, xpath in xpaths.items():
                result = tree.xpath(xpath)
                # 提取的结果可能是列表，取第一个值（如果存在）
                row_data[field] = result #[0] if result else None

            # 将结果添加到 data 列表
            data.append(row_data)

# 将结果转换为 DataFrame
df_res = pd.DataFrame(data)

print(df_res.head(), end=("\n"))
print(df_res.shape, end=("\n"))
print(df_res.name[0]," ", df_res.phone[0]," ", df_res.type[0]," ", df_res.website[0])


## phone 
# 遍历文件夹中的所有 HTML 文件
data = []
for filename in os.listdir(htm_path):
    if filename.endswith(".htm") or filename.endswith(".html"):
        file_path = os.path.join(htm_path, filename)

        # 解析 HTML 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            tree = etree.HTML(content)

            # 从 HTML 文件中提取网页标题
            result = tree.xpath(xpath_phone)
            title = result #[0] if result else None

            # 将文件名和提取到的标题添加到数据中
            data.append({'filename': filename, 'phone': title})

# 将结果转换为 DataFrame
df = pd.DataFrame(data)
print(df.head(), end="\n")
print(df.shape, end="\n")

## 结果如下
# 0  1902.htm  [College Search - Concordia College - At a Gla...    []   []      []
# 1  1916.htm  [College Search - Cossatot Community College o...    []   []      []
# 2  1080.htm  [College Search - Simmons College - At a Glanc...    []   []      []
# 3  1094.htm  [College Search - Sojourner-Douglass College -...    []   []      []
# 4  0361.htm  [College Search - Johnson & Wales University: ...    []   []      []
# (2000, 5)
# ['College Search - Concordia College - At a Glance\n \n']   []   []   []
#    filename phone
# 0  1902.htm    []
# 1  1916.htm    []
# 2  1080.htm    []
# 3  1094.htm    []
# 4  0361.htm    []
# (2000, 2)

import os
import pandas as pd
import html5lib
from lxml import etree

# 设置 HTML 文件所在的目录路径
htm_path = "/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/university/university-collegeboard(2000)/"

# 定义要使用的 XPath 表达式
xpaths = {
    'name': df_xpath['xpath'].iloc[0],
    'phone': df_xpath['xpath'].iloc[1],
    'type': df_xpath['xpath'].iloc[2],
    'website': df_xpath['xpath'].iloc[3]
}

# 创建一个空的列表来存储结果
data = []

# 遍历文件夹中的所有 HTML 文件
for filename in os.listdir(htm_path):
    if filename.endswith(".htm") or filename.endswith(".html"):
        file_path = os.path.join(htm_path, filename)

        # 读取 HTML 文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_html = file.read()

            # 解析 HTML 文件
            tree = html5lib.parse(raw_html, treebuilder="lxml", namespaceHTMLElements=False)

            # 存储每个字段的提取结果
            row_data = {"filename": filename}

            # 遍历每个 XPath 表达式，提取对应的字段
            for field, xpath in xpaths.items():
                # 提取匹配的所有结果，存储为列表
                result = tree.xpath(xpath)
                row_data[field] = result if result else None  # 如果没有匹配结果，存为 None

            # 将行数据添加到数据列表
            data.append(row_data)

# 将结果转换为 DataFrame
df_res = pd.DataFrame(data, columns=["filename", "name", "phone", "type", "website"])

# 显示结果
print(df_res.head(), end=("\n"))
print(df_res.shape, end=("\n"))
print(df_res.name[0]," ", df_res.phone[0]," ", df_res.type[0]," ", df_res.website[0])

## 结果如下
#    filename                                               name phone  type website
# 0  1902.htm  [College Search - Concordia College - At a Gla...  None  None    None
# 1  1916.htm  [College Search - Cossatot Community College o...  None  None    None
# 2  1080.htm  [College Search - Simmons College - At a Glanc...  None  None    None
# 3  1094.htm  [College Search - Sojourner-Douglass College -...  None  None    None
# 4  0361.htm  [College Search - Johnson & Wales University: ...  None  None    None
# (2000, 5)