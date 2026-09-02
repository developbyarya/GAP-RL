import ast
import urllib.request

url = "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v1.8.0/stable_baselines3/sac/policies.py"
response = urllib.request.urlopen(url)
source = response.read().decode('utf-8')

tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'Actor':
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                args = [a.arg for a in item.args.args]
                print(args)
