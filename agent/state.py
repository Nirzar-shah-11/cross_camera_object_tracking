from typing import List, TypedDict, Dict, Any


class TrackingState(TypedDict):
    reference_image: str
    reference_embedding: object
    attributes: Dict
    candidates: List
    verified_candidates: List
    final_results: List
