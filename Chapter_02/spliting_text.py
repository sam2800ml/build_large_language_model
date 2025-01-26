import re 
path = "Hello, world. this-- is a test"
result = re.split(r'([,.:;?_!()]|--|\s)', path)
print(result)

result = [item for item in result if item.strip()]
print(result)