import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

QUESTION_CONVERSION = """
将下面的问题中的格式要求移除，仅保留问题及字段的定义。注意，字段名称需要用`括起来。仅回答问题，不要包含额外内容。

示例：
问题：请提供一个商学院的名称、联系电话、类型和网站链接，以JSON格式回答。
```json
{{"name":["商学院名称"],"phone":["联系电话"],"type":["类型"],"website":["网站链接"]}}
```
回答：请提供一个商学院的名称(`name`)、联系电话(`phone`)、类型(`type`)和网站链接(`website`)。


问题：{query}
""".strip()

source_folder = "datasets/swde/questions_en"
dest_folder = "datasets/swde/questions_en_converted"
llm = ChatOpenAI(model="deepseek-chat", temperature=0.1) # ChatAnthropic    
template = ChatPromptTemplate.from_messages([("human", QUESTION_CONVERSION)])
chain = template | llm

if __name__ == '__main__':
    # create if not exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Create a list of all the files in the source_folder
    files = os.listdir(source_folder)
    # For each file in the list
    for file in files:
        # If the file is a .txt file
        if not file.endswith(".txt"):
            continue

        # Open the file
        content = open(os.path.join(source_folder, file)).read()
        # Run the chain
        response = chain.invoke({"query": content})
        # Save the response
        with open(os.path.join(dest_folder, file), "w") as f:
            f.write(response.content)
            f.write("\n")
            
