"""
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
"""

query = "Which planet is known as the Red Planet?"
passages = [
    "Venus is often called Earth's twin.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]
scores = model.predict([(query, passage) for passage in passages])
print(scores)

"""
def reload_run(filename="cross_encoder.py"):
    with open(filename, "r") as f:
        exec(f.read(), globals())
"""