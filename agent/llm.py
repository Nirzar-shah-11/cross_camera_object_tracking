import base64
import json
from openai import OpenAI

client = OpenAI(api_key="***")


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_person(image_path):
    image = image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-5-mini",          # or "gpt-5-mini"
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": """
Describe the person.

Return ONLY JSON in the following format:

{
    "shirt":"",
    "pants":"",
    "shoes":"",
    "bag":"",
    "gender":"",
    "accessories":[]
}
""",
                    },
                ],
            }
        ],
    )

    return json.loads(response.choices[0].message.content)


def verify_match(reference, candidate):
    ref = image_to_base64(reference)
    cand = image_to_base64(candidate)

    response = client.chat.completions.create(
        model="gpt-5-mini",          # or "gpt-5-mini"
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{ref}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{cand}"
                        },
                    },
                    {
                        "type": "text",
                        "text": """
Are these likely the same person?

Return ONLY JSON in the following format:

{
    "match": true,
    "confidence": 0.87,
    "reason": ""
}
""",
                    },
                ],
            }
        ],
    )

    return json.loads(response.choices[0].message.content)


attributes = describe_person("target.jpg")
print(attributes)

# Example
# result = verify_match("target.jpg", "candidate.jpg")
# print(result)