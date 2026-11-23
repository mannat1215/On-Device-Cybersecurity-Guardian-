# LLM-Based Cybersecurity Advisory Module

This module generates easy-to-understand cybersecurity explanations and safety recommendations based on the outputs of the URL, SMS, and Network Detection modules.

It consists of two components:

### 1. Llama Advisory Generation
Uses a locally deployed Llama 3.2 1B Instruct INT4 model through ExecuTorch to generate human-readable cybersecurity advisories based on threat-fusion results.

### 2. Rule-Based Explanation
Provides predefined explanations and safety recommendations as a fallback when the local Llama model is unavailable.

### Key Features
- Generates understandable cybersecurity threat explanations.
- Uses threat-fusion results from multiple detection modules.
- Provides safety recommendations to users.
- Supports offline advisory generation through local LLM deployment.
- Includes a rule-based fallback mechanism.

### Main Files
- `llm_explainer.py` – Rule-based cybersecurity explanation.
- `executorch/llama_advisory.py` – Local Llama advisory generation using ExecuTorch.
