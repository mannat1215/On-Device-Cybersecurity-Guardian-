
# ============================================================
# ON-DEVICE CYBERSECURITY GUARDIAN
# Main Application
# ============================================================

import sys
from pathlib import Path
import pandas as pd

# Get the directory containing this main.py file.
BASE_DIR = Path(__file__).resolve().parent

# Add detector directories to Python's module search path.
sys.path.insert(0, str(BASE_DIR / "SMS_Detection"))
sys.path.insert(0, str(BASE_DIR / "URL_Detection"))
sys.path.insert(0, str(BASE_DIR / "Network_Detection"))

# Import the individual detection modules.
from predict_sms_v2 import detect_sms
from predict_url import predict_url
from predict_network import predict_network

# Import Threat Fusion.
from threat_fusion import fuse_threats

# Import the rule-based explanation fallback.
from llm_explainer import generate_explanation

# Import the locally deployed Llama advisory module.
from LLM.executorch.llama_advisory import generate_advisory


def main():

    # Initialize detector results.
    sms_result = None
    url_result = None
    network_result = None

    # Display application header.
    print("\n======================================")
    print("       CYBERSECURITY GUARDIAN")
    print("======================================")

    # ========================================================
    # SMS THREAT DETECTION
    # ========================================================

    print("\n--- SMS THREAT DETECTION ---")

    sms = input("Enter an SMS (or press Enter to skip): ")

    if sms.strip():
        sms_result = detect_sms(sms)
        print("SMS Result:", sms_result)
    else:
        print("SMS Detection: Skipped")

    # ========================================================
    # URL THREAT DETECTION
    # ========================================================

    print("\n--- URL THREAT DETECTION ---")

    url = input("Enter a URL (or press Enter to skip): ")

    if url.strip():
        url_result = predict_url(url)
        print("URL Result:", url_result)
    else:
        print("URL Detection: Skipped")

    # ========================================================
    # NETWORK THREAT DETECTION
    # ========================================================

    print("\n--- NETWORK THREAT DETECTION ---")

    # Path to the UNSW-NB15 testing dataset.
    # This implementation analyses dataset records,
    # not live network traffic.
    data_path = BASE_DIR / "Network_Detection" / "UNSW_NB15_testing-set.csv"

    df = pd.read_csv(data_path)

    print("Available network test records:", len(df))
    print(
        "Note: Network detection uses dataset records, "
        "not live traffic."
    )

    record_number = input(
        "Enter record number (0 to skip network): "
    ).strip()

    # Validate the selected record number.
    if record_number.isdigit():

        record_number = int(record_number)

        if record_number == 0:
            print("Network Detection: Skipped")

        elif record_number < len(df):

            # Remove target and identifier columns.
            # Keep only the features required by the model.
            record = df.iloc[record_number].drop(
                labels=["label", "attack_cat", "id"],
                errors="ignore"
            ).to_dict()

            network_result = predict_network(record)
            print("Network Result:", network_result)

        else:
            print(
                "Invalid record number. "
                "Network Detection: Skipped"
            )

    else:
        print(
            "Invalid input. "
            "Network Detection: Skipped"
        )

    # ========================================================
    # THREAT FUSION
    # ========================================================

    # Combine all available detector results.
    final_result = fuse_threats(
        sms_result=sms_result,
        url_result=url_result,
        network_result=network_result
    )

    print("\n======================================")
    print("          THREAT FUSION")
    print("======================================")

    print("Overall Risk:", final_result["overall_risk"].upper())
    print("Threats Detected:", final_result["threats_detected"])

    # Display individual detector results.
    print("\nIndividual Module Results:")

    module_results = final_result["module_results"]

    if "sms" in module_results:
        print("SMS:", module_results["sms"])

    if "url" in module_results:
        print("URL:", module_results["url"])

    if "network" in module_results:
        print("NETWORK:", module_results["network"])

    # ========================================================
    # AI CYBERSECURITY ANALYSIS
    # ========================================================

    print("\n======================================")
    print("       AI CYBERSECURITY ANALYSIS")
    print("======================================")

    # Llama explains the detector findings.
    # It does not independently determine the threat risk.
    llama_success = generate_advisory(
        final_result,
        sms_text=sms,
        url_text=url
    )

    # Use the rule-based explanation if Llama fails.
    if not llama_success:

        print("\nLlama advisory unavailable or invalid.")
        print("Using verified rule-based explanation instead.")

        generate_explanation(final_result)

    # ========================================================
    # APPLICATION COMPLETION
    # ========================================================

    print("\n======================================")
    print("          ANALYSIS COMPLETE")
    print("======================================")


# Run the application when this file is executed directly.
if __name__ == "__main__":
    main()
