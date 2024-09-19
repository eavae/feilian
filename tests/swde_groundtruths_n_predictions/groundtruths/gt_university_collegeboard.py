import pandas as pd

## groundtruth of university_collegeboard_phone  
file_path = '/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-collegeboard-phone.txt'
# Define five column names, filling missing columns with NaN
df = pd.read_csv(file_path, sep='\t', header=None, engine='python', on_bad_lines='skip', 
                 names=["col0", 'col1', 'col2', 'col3', 'col4'], na_values='')
## drop top 2 rows 
df = df.drop(index=df.index[:2])
df = df.reset_index(drop=True)
# 重命名 'col0' 为 'id'
df.rename(columns={'col0': 'id'}, inplace=True)
# 将 'col2', 'col3', 'col4' 的值合并为一个列表，并生成一个新列 'groundtruth'
df['groundtruth'] = df[['col2', 'col3', 'col4']].apply(lambda x: [i for i in x if pd.notna(i)], axis=1)
# 删除 'col1', 'col2', 'col3', 'col4' 列
df.drop(columns=['col1', 'col2', 'col3', 'col4'], inplace=True)
df_gt_university_collegeboard_phone = df.copy()
#print(df_gt_university_collegeboard_phone.head())


## groundtruth of university_collegeboard_phone  
file_path = '/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-collegeboard-name.txt'
# Define five column names, filling missing columns with NaN
df = pd.read_csv(file_path, sep='\t', header=None, engine='python', on_bad_lines='skip', 
                 names=["col0", 'col1', 'col2'], na_values='')
## drop top 2 rows 
df = df.drop(index=df.index[:2])
df = df.reset_index(drop=True)
# 重命名 'col0' 为 'id'
df.rename(columns={'col0': 'id', 'col2': 'groundtruth'}, inplace=True)
# 删除 'col1' 列
df.drop(columns=['col1'], inplace=True)
df_gt_university_collegeboard_name = df.copy()
print(df_gt_university_collegeboard_name.head())


## groundtruth of university_collegeboard_type
file_path = '/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-collegeboard-type.txt'
# Define five column names, filling missing columns with NaN
df = pd.read_csv(file_path, sep='\t', header=None, engine='python', on_bad_lines='skip', 
                 names=["col0", 'col1', 'col2'], na_values='')
## drop top 2 rows 
df = df.drop(index=df.index[:2])
df = df.reset_index(drop=True)
# 重命名 'col0' 为 'id'
df.rename(columns={'col0': 'id', 'col2': 'groundtruth'}, inplace=True)
# 删除 'col1' 列
df.drop(columns=['col1'], inplace=True)
df_gt_university_collegeboard_type = df.copy()
print(df_gt_university_collegeboard_type.head())

## groundtruth of university_collegeboard_website
file_path = '/Users/brycewang/Desktop/智能爬虫/data/swde/sourceCode/sourceCode/groundtruth/university/university-collegeboard-website.txt'
# Define five column names, filling missing columns with NaN
df = pd.read_csv(file_path, sep='\t', header=None, engine='python', on_bad_lines='skip', 
                 names=["col0", 'col1', 'col2'], na_values='')
## drop top 2 rows 
df = df.drop(index=df.index[:2])
df = df.reset_index(drop=True)
# 重命名 'col0' 为 'id'
df.rename(columns={'col0': 'id', 'col2': 'groundtruth'}, inplace=True)
# 删除 'col1' 列
df.drop(columns=['col1'], inplace=True)
df_gt_university_collegeboard_website = df.copy()
print(df_gt_university_collegeboard_website.head())