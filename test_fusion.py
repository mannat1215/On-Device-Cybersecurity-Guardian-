
# Import the threat fusion function.
from threat_fusion import fuse_threats


# Sample result received from the SMS detector.
# This example represents a high-risk spam message.
sms_result = {
    "module": "sms",
    "prediction": "spam",
    "spam_probability": 0.9523,
    "risk_level": "high"
}


# Sample result received from the URL detector.
# This example represents a legitimate URL with low risk.
url_result = {
    "module": "url",
    "prediction": "legitimate",
    "phishing_probability": 0.1999,
    "risk_level": "low"
}


# Sample result received from the Network detector.
# This example represents a detected network attack
# with a medium risk level.
network_result = {
    "module": "network_threat_detection",
    "prediction": "ATTACK",
    "attack_probability": 0.7057,
    "risk_level": "medium"
}


# Pass the results from all three detectors to the
# threat fusion function to calculate the overall risk.
result = fuse_threats(
    sms_result,
    url_result,
    network_result
)


# Display the combined threat assessment.
print("\nCYBERSECURITY GUARDIAN — FUSION TEST")
print("------------------------------------")

# Display the highest risk level among the active detectors.
print("Overall Risk:", result["overall_risk"].upper())

# Display which detectors reported medium or high risk.
print("Threats Detected:", result["threats_detected"])

# Display the individual results received from each detector.
print("Module Results:", result["module_results"])
