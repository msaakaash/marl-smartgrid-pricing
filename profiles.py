DEFAULT_PROFILE_TEMPLATE = {
    "type": "residential",
    "flexibility": 0.5,
    "priority": 3,
    "willingness_to_shift": 0.5,
    "emergency_status": 0,
}

def normalize_profile(profile: dict):
    """Convert profile dict into a normalized feature vector."""
    type_code = 0 if profile["type"] == "residential" else 1
    return [
        type_code,
        float(profile["flexibility"]),
        profile["priority"] / 5.0,
        float(profile["willingness_to_shift"]),
        float(profile["emergency_status"]),
    ]
