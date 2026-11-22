import random

ROUTES = [
    ["e0", "e1"],
    ["e1_rev", "e0_rev"],
    ["e2", "e3"],
    ["e3_rev", "e2_rev"]
]

def generate_route():
    return random.choice(ROUTES)