import orjson

# load in data_response.json
with open("data_response.json", "r") as f:
    data = orjson.loads(f.read())

    jout = data['choices'][0]['message']['content']
    # delete everything before <channel|>
    jout = jout.split("<channel|>")[-1]
    # remove ````json` and ` ````
    jout = jout.replace("```json", "").replace("```", "")
    print(jout)
