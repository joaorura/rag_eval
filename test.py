import json

# String JSON com aspas escapadas
json_string = '\"{\\"feedback\": \\"Esta quest\u00e3o\\"}"'

# Decodificar a string JSON
decoded_json = json.loads(json_string)

# Imprimir o resultado
print(decoded_json)