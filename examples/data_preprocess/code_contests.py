import pandas as pd

prefix = """The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. User: {query}\nAssistant: <think>"""

prefix2 = "\nThis is the problem:\n{prompt}\n"

train = pd.read_parquet("/workspace/data/code-contests/train.parquet")
for index, row in train.iterrows():
    system = row["prompt"][0]["content"]
    assert row["prompt"][0]["role"] == "system"
    human = row["prompt"][1]["content"]
    assert row["prompt"][1]["role"] == "user"
    
    a = system.split("Now, the user will present you with")[1]
    query = "Now, the user will present you with" + a + prefix2.format(prompt = human)
    
    question = prefix.format(query = query)
    
    train.loc[index, 'prompt'] = [{
        "role": "user",
        "content": question
    }]

test = pd.read_parquet("/workspace/data/code-contests/test.parquet")
for index, row in test.iterrows():
    system = row["prompt"][0]["content"]
    assert row["prompt"][0]["role"] == "system"
    human = row["prompt"][1]["content"]
    assert row["prompt"][1]["role"] == "user"
    
    a = system.split("Now, the user will present you with")[1]
    query = "Now, the user will present you with" + a + prefix2.format(prompt = human)
    
    question = prefix.format(query = query)
    
    test.loc[index, 'prompt'] = [{
        "role": "user",
        "content": question
    }]
    
train.to_parquet('/workspace/data/code-contests/train_xiugai.parquet')
test.to_parquet('/workspace/data/code-contests/test_xiugai.parquet')