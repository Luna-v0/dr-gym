def reward_function(params):
    """
    Reward the car for staying close to the center line.
    """
    track_width = params["track_width"]
    distance_from_center = params["distance_from_center"]

    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    if distance_from_center <= marker_1:
        return 100.0
    if distance_from_center <= marker_2:
        return 0.5
    if distance_from_center <= marker_3:
        return 0.1
    return 1e-3
