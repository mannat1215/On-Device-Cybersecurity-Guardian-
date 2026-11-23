

# ============================================================
# ON-DEVICE CYBERSECURITY GUARDIAN
# Module: Rule-Based Cybersecurity Explanation
#
# Purpose:
# Generate a structured cybersecurity report using
# predefined explanations and safety recommendations.
#
# This module does NOT use an AI language model.
# It provides deterministic explanations based on the
# threat-fusion results.
# ============================================================


# ============================================================
# 1. THREAT INFORMATION DATABASE
# ============================================================

# This dictionary stores predefined information for each
# supported threat category:
#
# SMS     -> Suspicious text messages
# URL     -> Potentially harmful websites
# NETWORK -> Potentially malicious network activity
#
# Each category contains:
# 1. name: Display name of the threat.
# 2. description: Explanation of the detected threat.
# 3. recommendations: Safety actions for the user.

THREAT_INFO = {

    # --------------------------------------------------------
    # SMS THREAT INFORMATION
    # --------------------------------------------------------

    "sms": {
        "name": "SMS",

        "description": (
            "The SMS detector identified a suspicious text message. "
            "It may contain fraudulent requests, phishing links, "
            "or attempts to steal personal information."
        ),

        "recommendations": [
            "Do not reply to suspicious messages or share passwords and OTPs.",
            "Verify unexpected prize, banking, and payment requests independently."
        ]
    },


    # --------------------------------------------------------
    # URL THREAT INFORMATION
    # --------------------------------------------------------

    "url": {
        "name": "URL",

        "description": (
            "The URL detector identified a potentially harmful website. "
            "It may be designed to steal login credentials or personal information."
        ),

        "recommendations": [
            "Do not open suspicious links or enter personal information on them.",
            "Visit the official website directly instead of following an unverified link."
        ]
    },


    # --------------------------------------------------------
    # NETWORK THREAT INFORMATION
    # --------------------------------------------------------

    "network": {
        "name": "NETWORK",

        "description": (
            "The network detector identified potentially malicious network activity. "
            "This may indicate an attack or unauthorized access attempt."
        ),

        "recommendations": [
            "Investigate the flagged network activity.",
            "Review device security and restrict unauthorized network access."
        ]
    }
}


# ============================================================
# 2. GENERATE CYBERSECURITY EXPLANATION
# ============================================================

def generate_explanation(fusion_result):

    # Read the overall risk calculated by the threat-fusion module.
    # Convert it to uppercase for consistent report formatting.
    risk = fusion_result.get("overall_risk", "unknown").upper()

    # Obtain the list of threat categories identified by fusion.
    # Example: ["sms", "network"]
    threats = fusion_result.get("threats_detected", [])


    # ========================================================
    # 3. DISPLAY REPORT HEADER AND OVERALL RISK
    # ========================================================

    print("\n========== CYBERSECURITY REPORT ==========")

    print("\nOVERALL RISK:")
    print(risk)


    # ========================================================
    # 4. DISPLAY DETECTED THREAT CATEGORIES
    # ========================================================

    print("\nTHREATS DETECTED:")

    # Display the detected categories in uppercase.
    # If no threats are present, display None.
    print(", ".join(threats).upper() if threats else "None")


    # ========================================================
    # 5. GENERATE VERIFIED EXPLANATIONS
    # ========================================================

    print("\nVERIFIED EXPLANATION:")

    # Filter the detected threats.
    #
    # Only categories present in THREAT_INFO are accepted.
    # This prevents the program from trying to access
    # explanations for unknown threat categories.

    valid_threats = [
        threat.lower() for threat in threats
        if threat.lower() in THREAT_INFO
    ]

    # If no recognized threat category is available,
    # display a default explanation.
    if not valid_threats:

        print("No medium- or high-risk threats were detected.")

    else:

        # Display the predefined explanation for each
        # recognized threat category.
        for threat in valid_threats:

            info = THREAT_INFO[threat]

            print(f"{info['name']}: {info['description']}")


    # ========================================================
    # 6. GENERATE SYSTEM SUMMARY
    # ========================================================

    print("\nSYSTEM SUMMARY:")

    # Case 1: No input was analysed.
    if risk == "UNKNOWN":

        print(
            "No inputs were analysed. Enter at least one input "
            "to generate a risk assessment."
        )

    # Case 2: No recognized medium- or high-risk threat exists.
    elif not valid_threats:

        print(
            "No medium- or high-risk threats were detected by the system."
        )

    # Case 3: One or more recognized threats were detected.
    else:

        # Obtain the display names of the detected threat categories.
        names = [
            THREAT_INFO[t]["name"] for t in valid_threats
        ]

        # Print the number and names of detected threats,
        # along with the overall risk determined by fusion.
        print(
            f"The system detected {len(names)} "
            f"medium- or high-risk threat(s): {', '.join(names)}. "
            f"The overall cybersecurity risk is {risk}."
        )


    # ========================================================
    # 7. DISPLAY SAFETY RECOMMENDATIONS
    # ========================================================

    print("\nRECOMMENDATIONS:")

    # If no recognized threats are present, provide
    # a general recommendation.
    if not valid_threats:

        print(
            "1. Continue monitoring messages, URLs, and network activity."
        )

    else:

        # Start recommendation numbering from 1.
        number = 1

        # Display recommendations associated with each threat.
        for threat in valid_threats:

            for recommendation in THREAT_INFO[threat]["recommendations"]:

                print(f"{number}. {recommendation}")

                # Increment the number for the next recommendation.
                number += 1


    # ========================================================
    # 8. END OF CYBERSECURITY REPORT
    # ========================================================

    print("\n==========================================")


# ============================================================
# 9. STANDALONE TESTING
# ============================================================

# This section executes only when this Python file is run
# directly, not when it is imported by main.py.
#
# It creates a sample fusion result to demonstrate
# how the explanation module works.

if __name__ == "__main__":

    # Sample result containing two detected threat categories.
    sample_result = {
        "overall_risk": "high",
        "threats_detected": ["sms", "network"]
    }

    print("\nGenerating cybersecurity explanation...\n")

    # Generate the sample cybersecurity report.
    generate_explanation(sample_result)


