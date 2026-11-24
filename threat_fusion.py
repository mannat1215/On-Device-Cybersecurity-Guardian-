
def fuse_threats(sms_result=None, url_result=None, network_result=None):
    """
    Combines the results from SMS, URL, and Network detectors.

    Determines the overall risk level based on the highest
    risk reported by any active detector.
    """

    # Store the results received from all three detectors.
    results = {
        "sms": sms_result,
        "url": url_result,
        "network": network_result
    }

    # Assign a numerical score to each risk level.
    # A higher score represents a more serious threat.
    risk_scores = {
        "low": 1,
        "medium": 2,
        "high": 3
    }

    # Include only the detectors that actually produced a result.
    # Detectors with None results are ignored.
    active_results = {
        name: result
        for name, result in results.items()
        if result is not None
    }

    # If no detector produced a result, return an unknown risk.
    if not active_results:
        return {
            "overall_risk": "unknown",
            "threats_detected": [],
            "module_results": {}
        }

    # Initially assume that the overall risk is low.
    highest_risk = "low"

    # Store the names of detectors that identify medium or high risk.
    threats_detected = []

    # Check the risk level reported by each active detector.
    for module, result in active_results.items():

        # Convert the risk level to lowercase for consistency.
        risk = result["risk_level"].lower()

        # Check whether the detector returned a valid risk level.
        if risk not in risk_scores:
            raise ValueError("Invalid risk level: " + risk)

        # Update the overall risk if this detector reports
        # a higher risk than the previously recorded one.
        if risk_scores[risk] > risk_scores[highest_risk]:
            highest_risk = risk

        # Mark the detector as a threat source if its risk
        # level is medium or high.
        if risk in ("medium", "high"):
            threats_detected.append(module)

    # Return the combined results for further processing.
    return {
        "overall_risk": highest_risk,
        "threats_detected": threats_detected,
        "module_results": active_results
    }
