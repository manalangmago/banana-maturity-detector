"""
classification.py
------------------
Takes the raw detections from detect.py (which are just labels/boxes/confidence)
and turns them into a clean, frontend-friendly ripeness result.

This is where your BUSINESS LOGIC lives — e.g. "if the model says overripe,
tell the user it's best for banana bread", or "if two bananas are detected,
report on the ripest one". You can expand this however your app needs.

If later on you train a SEPARATE ripeness classifier (different from the
detection model), this file is also where you'd load and run that second
model. Right now it assumes best.pt already outputs ripeness as its classes.
"""

# Optional: friendly descriptions for each class your model was trained on.
# Update these strings to match your ACTUAL class names from training.
RIPENESS_INFO = {
    "unripe": {
        "description": "This banana is unripe (green). Best eaten in a few days.",
        "days_until_ripe": 3,
    },
    "ripe": {
        "description": "This banana is ripe and ready to eat.",
        "days_until_ripe": 0,
    },
    "overripe": {
        "description": "This banana is overripe. Great for banana bread or smoothies.",
        "days_until_ripe": 0,
    },
}


def classify(detections: list):
    """
    Args:
        detections: the list returned by detect.run_detection()

    Returns:
        {
            "banana_found": bool,
            "count": int,
            "results": [
                {
                    "label": "ripe",
                    "confidence": 0.93,
                    "box": [...],
                    "description": "...",
                    "days_until_ripe": 0
                },
                ...
            ]
        }
    """
    if not detections:
        return {
            "banana_found": False,
            "count": 0,
            "results": [],
        }

    enriched_results = []
    for detection in detections:
        label = detection["label"]
        info = RIPENESS_INFO.get(label, {})

        enriched_results.append({
            **detection,
            "description": info.get("description", "Unknown ripeness stage."),
            "days_until_ripe": info.get("days_until_ripe"),
        })

    return {
        "banana_found": True,
        "count": len(enriched_results),
        "results": enriched_results,
    }