import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state import TrackingState 
from llm import describe_person, verify_match
from src.match import search_candidates
from src.reid import OSNetReID
from langgraph.graph import StateGraph

def extract_attributes( state: TrackingState ):

    attributes = describe_person(
        state["reference_image"]
    )

    state["attributes"] = attributes

    return state

def find_candidates( state: TrackingState ):

    candidates = search_candidates(
        state["reference_embedding"]
    )

    state["candidates"] = candidates

    return state

def verify_candidates(
    state
):

    verified = []

    for candidate in state["candidates"]:

        score = candidate["similarity"]

        if score > 0.90:

            candidate["verified"]=True

            verified.append(candidate)

            continue

        if score < 0.60:

            continue

        if not candidate.get("image"):

            continue

        response = verify_match(

            state["reference_image"],

            candidate["image"]

        )

        if response["match"]:

            candidate["verified"]=True

            candidate["llm"]=response

            verified.append(candidate)

    state["verified_candidates"]=verified

    return state

def rank_results(state):

    candidates = search_candidates(
        state["reference_embedding"]
    )

    state["candidates"] = candidates

    return state


workflow = StateGraph(
    TrackingState
)

workflow.add_node(
    "attributes",
    extract_attributes
)

workflow.add_node(
    "retrieve",
    find_candidates
)

workflow.add_node(
    "verify",
    verify_candidates
)

workflow.add_node(
    "rank",
    rank_results
)

workflow.set_entry_point(
    "attributes"
)

workflow.add_edge(
    "attributes",
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "verify"
)

workflow.add_edge(
    "verify",
    "rank"
)

workflow.set_finish_point(
    "rank"
)

graph = workflow.compile()

if __name__ == "__main__":
    reference_image = ROOT / "data" / "reference" / "target.jpg"
    reference_embedding = OSNetReID().extract(str(reference_image))

    result = graph.invoke(
        {
            "reference_image": str(reference_image),
            "reference_embedding": reference_embedding,
        }
    )

    print(result)
