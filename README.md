# On-Device Cybersecurity Guardian
### An Edge-Based, Multi-Source Cybersecurity Threat Detection and Advisory System

## 1. PROJECT OVERVIEW

The **On-Device Cybersecurity Guardian** is an AI-powered cybersecurity system designed to detect and assess potential digital threats directly on a Raspberry Pi 4 Model B.

The system analyzes three types of cybersecurity inputs:

- **SMS Detection:** Identifies potentially fraudulent, phishing, and spam messages.
- **URL Detection:** Examines URL characteristics to identify potentially malicious or phishing websites.
- **Network Threat Detection:** Classifies network traffic records to identify potentially suspicious activity.

The outputs from these individual detection modules are combined using a Threat Fusion mechanism to determine an overall risk level.

A locally deployed **Llama 3.2 1B Instruct model**, optimized using INT4 SpinQuant quantization and executed through ExecuTorch, generates a natural-language explanation of the detected risks.

Unlike systems that depend on cloud-based AI services, the cybersecurity advisory component runs locally on the Raspberry Pi. A rule-based explanation mechanism is also available as a fallback if the local LLM fails or produces an invalid response.

The project aims to demonstrate how multiple lightweight detection techniques and a locally deployed language model can work together to provide an accessible, resource-conscious cybersecurity solution.

---

## 2. PROBLEM STATEMENT

Digital communication and internet usage expose users to several cybersecurity threats, including phishing messages, malicious URLs, fraudulent SMS notifications, and suspicious network activity.

Many existing cybersecurity solutions focus on individual threat categories or depend on cloud-based services for advanced analysis. Such approaches may require continuous internet connectivity, external API access, or greater computational resources.

There is a need for a system that can:

- Analyze multiple categories of cybersecurity threats.
- Combine individual detection results into an overall risk assessment.
- Explain the results in simple, understandable language.
- Operate on affordable edge hardware.
- Reduce dependence on external cloud-based AI services.

The On-Device Cybersecurity Guardian addresses these requirements through a modular detection architecture and local AI-assisted advisory generation.

---

## 3. PROJECT OBJECTIVES

1. Develop independent detection modules for SMS messages, URLs, and network traffic records.
2. Combine individual detection results through a centralized Threat Fusion mechanism.
3. Classify the overall threat level into meaningful risk categories.
4. Integrate a locally deployed language model to explain the detection results.
5. Implement a rule-based fallback for reliable advisory generation when the LLM is unavailable.
6. Optimize the language model deployment for resource-constrained Raspberry Pi hardware.
7. Demonstrate an edge-based cybersecurity architecture that minimizes dependence on cloud AI services.

---

## 4. WHAT MAKES OUR PROJECT UNIQUE?

The main contribution of this project is not simply using machine learning to classify suspicious inputs. It is the integration of multiple detection modules, centralized risk assessment, and local AI-based explanation within a resource-constrained edge device.

### 4.1 Multi-Source Threat Analysis

Instead of analyzing only one type of cybersecurity threat, the system accepts three different categories of inputs:

- SMS messages
- URLs
- Network traffic records

Each module performs its own analysis and produces a detection result. These results are then passed to a common Threat Fusion mechanism.

This modular approach makes the system extendable without requiring the entire application to be redesigned whenever a new detection module is introduced.

### 4.2 Centralized Threat Fusion

Individual detection results are not treated as unrelated outputs.

The Threat Fusion module combines the available results and determines an overall risk level. It also maintains information about which detection modules identified potential threats.

This provides a consolidated view instead of forcing the user to interpret multiple independent classifier outputs.

### 4.3 On-Device AI Advisory

A distinctive feature is the integration of a locally executed Llama 3.2 1B Instruct model.

The language model receives the detector and fusion results and generates a short natural-language explanation.

The LLM is used for interpretation and advisory generation, not as the primary threat detector. The underlying detection modules and Threat Fusion results remain authoritative.

### 4.4 Rule-Based Fallback

The system does not depend entirely on successful LLM execution.

If the Llama runner fails, times out, or produces an invalid response, the application can use a rule-based explanation mechanism instead.

This provides an alternative advisory path and prevents the complete application from depending on the availability of the language model.

### 4.5 Edge Deployment

The complete system is designed around a Raspberry Pi 4 Model B rather than requiring a high-performance GPU or cloud-hosted LLM.

This demonstrates the practical use of optimized machine learning and local language-model inference on affordable edge hardware.

---

## 5. SYSTEM ARCHITECTURE

The system follows a modular processing architecture.

```text
              USER INPUT
                  |
       +----------+----------+
       |          |          |
      SMS        URL      NETWORK
       |          |       RECORD
       v          v          v
  SMS DETECTOR  URL       NETWORK
              DETECTOR   DETECTOR
       |          |          |
       +----------+----------+
                  |
                  v
           THREAT FUSION
                  |
          OVERALL RISK
                  |
                  v
          LOCAL LLAMA LLM
                  |
          +-------+-------+
          |               |
      Valid Output     Failure /
          |            Invalid Output
          |               |
          |               v
          |        RULE-BASED FALLBACK
          |               |
          +-------+-------+
                  |
                  v
        CYBERSECURITY ADVISORY
```

### Processing Flow

1. The user provides an SMS message, URL, or network traffic record.
2. The corresponding detection module analyzes the input.
3. Each module produces its prediction and associated risk information.
4. Threat Fusion combines the available module results.
5. The overall risk assessment is passed to the local Llama advisory module.
6. Llama generates a short explanation based on the available detection evidence.
7. If Llama execution fails or its response is invalid, the rule-based fallback generates an explanation.
8. The final result is presented to the user.

---

## 6. PROJECT MODULES

### 6.1 SMS Detection Module

The SMS Detection module analyzes text messages to identify potentially fraudulent or spam messages.

**Main operations:**

- Text preprocessing and feature extraction.
- TF-IDF-based text representation.
- Machine-learning-based message classification.
- Spam probability estimation.
- Risk-level assignment.

The module returns information such as:

- Prediction: HAM or SPAM.
- Spam probability.
- Risk level.

The model is designed to identify suspicious message patterns, including phishing-related language and fraudulent requests.

**Important:** Classification results are probabilistic. A HAM prediction does not guarantee that a message is safe or that its sender is genuine.

### 6.2 URL Detection Module

The URL Detection module analyzes URL characteristics to identify potentially malicious or phishing links.

It uses lexical URL features extracted directly from the supplied URL.

The implemented feature extraction includes characteristics such as:

- URL length.
- Number of special characters.
- Number of dots and subdomains.
- Presence of IP addresses.
- Suspicious URL patterns.
- Other structural characteristics of the URL.

The extracted features are passed to a locally loaded machine-learning model for classification.

The URL analysis shown in the implementation uses local feature extraction and model prediction rather than relying on an external URL reputation API.

### 6.3 Network Threat Detection Module

The Network Detection module analyzes network traffic records using a trained machine-learning model.

The project uses the **UNSW-NB15 dataset** for network traffic classification.

The module processes traffic-record features and predicts whether the record is normal or potentially malicious.

The current implementation is dataset-record-based. It does not represent a live packet-capture or continuous network-monitoring system.

### 6.4 Threat Fusion Module

Threat Fusion is responsible for combining the outputs of the individual detection modules.

It consolidates the available predictions, probabilities, and risk levels to determine the overall risk assessment.

The output includes:

- Overall risk level.
- Individual module results.
- Detected threat sources.
- Relevant classification information.

This module acts as the central decision layer between individual detectors and the advisory-generation system.

### 6.5 Local LLM Advisory Module

The project integrates **Llama 3.2 1B Instruct** to generate understandable explanations of the detector results.

The model is deployed using:

- ExecuTorch runtime.
- INT4 SpinQuant model.
- Native `llama_main` runner.
- Local tokenizer.
- CPU-based inference on Raspberry Pi.

The LLM receives a structured explanation prompt containing the available detection evidence.

The prompt is designed to encourage:

- Short and understandable explanations.
- Responses grounded in detector results.
- Avoidance of unsupported threat claims.
- Clear recommendations based on the available evidence.

The language model does not independently verify whether a sender, website, or network connection is trustworthy.

### 6.6 Rule-Based Explanation Fallback

A separate rule-based explanation mechanism is implemented as a fallback.

It is used when the LLM cannot successfully generate a valid advisory.

This reduces dependence on successful LLM execution and allows the application to retain an explanation mechanism even when the local model encounters an error.

---

## 7. OPTIMIZATION AND EDGE DEPLOYMENT

A major part of this project involved adapting a language model to run on Raspberry Pi hardware.

A conventional large language model deployment can be computationally expensive for a resource-constrained device. We therefore selected a smaller model and used model quantization and runtime-level configuration to make local inference practical.

### 7.1 INT4 SpinQuant Quantization

The project uses the Llama 3.2 1B Instruct model in an INT4 SpinQuant-optimized format.

Quantization reduces the precision used to represent model parameters.

Using INT4 representation helps reduce model storage requirements and memory bandwidth demands compared with higher-precision representations.

The deployed model file is approximately 1.1 GB.

This makes local deployment more practical on the Raspberry Pi than using a substantially larger, unquantized language model.

### 7.2 ExecuTorch Runtime

Instead of relying on a cloud-hosted language model API, the project uses ExecuTorch for local model execution.

The model is executed through a native `llama_main` runner on the Raspberry Pi.

The model and tokenizer are stored locally, and inference is performed on the device CPU.

This removes the need for a cloud LLM inference service for advisory generation.

### 7.3 CPU Thread Configuration

The Llama runner is configured to use four CPU threads:

```text
--cpu_threads=4
```

This allows the inference workload to use multiple CPU threads on the Raspberry Pi.

### 7.4 Controlled Token Generation

The advisory-generation configuration uses:

```text
--max_new_tokens=128
--temperature=0
--cpu_threads=4
```

These settings serve different purposes:

- `max_new_tokens=128`: Limits the maximum number of newly generated tokens.
- `temperature=0`: Uses greedy decoding for deterministic token selection.
- `cpu_threads=4`: Configures the number of CPU threads used by the runner.

The application prompt also requests a short advisory rather than a long generated report.

This helps limit unnecessary output generation and keeps the response focused on the detector results.

### 7.5 Conditional LLM Execution

The language model is used after the detection and fusion stages.

It is not required to perform every individual classification operation.

This separates the detection workload from natural-language explanation and avoids using the LLM as a replacement for the specialized detection models.

### 7.6 Execution Timeout and Fallback

The Llama advisory integration uses a subprocess timeout of 60 seconds.

If execution fails or does not produce an acceptable response, the application can use the rule-based fallback.

This is important because local language-model inference can take noticeably longer on low-power hardware.

### Summary of Optimization Techniques

| Optimization | Purpose |
|---|---|
| Smaller 1B language model | Reduces model complexity |
| INT4 SpinQuant | Reduces model representation size |
| ExecuTorch runtime | Enables local model execution |
| Four CPU threads | Configures CPU inference |
| Maximum 128 new tokens | Controls output length |
| Temperature zero | Makes generation deterministic |
| Short advisory prompt | Avoids unnecessarily long responses |
| Conditional LLM execution | Separates detection from explanation |
| Rule-based fallback | Provides an alternative when LLM execution fails |

---

## 8. TECHNOLOGY STACK

| Component | Technology |
|---|---|
| Programming language | Python |
| Edge hardware | Raspberry Pi 4 Model B |
| SMS detection | TF-IDF and machine learning |
| URL detection | Lexical feature extraction and machine learning |
| Network detection | Machine learning |
| Network dataset | UNSW-NB15 |
| Threat assessment | Python-based Threat Fusion |
| Language model | Llama 3.2 1B Instruct |
| Model optimization | INT4 SpinQuant |
| LLM runtime | ExecuTorch |
| Native inference | `llama_main` |
| Model serialization | Pickle / Joblib, depending on module |
| Version control | Git and GitHub |

---

## 9. HARDWARE AND SOFTWARE REQUIREMENTS

### Hardware

- Raspberry Pi 4 Model B.
- MicroSD card.
- Power supply for Raspberry Pi.
- Network connection for initial setup and file transfer, where required.

### Software

- Raspberry Pi OS.
- Python 3.
- Python virtual environment.
- Required Python libraries for the detection modules.
- ExecuTorch native deployment environment.
- Llama model file and tokenizer.

### LLM Deployment Files

The active local Llama deployment uses:

| File | Purpose |
|---|---|
| `llama_main` | Native ExecuTorch inference runner |
| `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` | Optimized model |
| `tokenizer.model` | Local tokenizer asset |
| Required shared libraries | Runtime dependencies |

The model, tokenizer, native runner, and required shared libraries must be present in their configured locations for local LLM inference.

The model binary and native runtime dependencies are separate from the Python source code.

---

## 10. REPOSITORY STRUCTURE

The repository is organized into separate modules for detection, fusion, and language-model advisory generation.

```text
On-Device-Cybersecurity-Guardian/
│
├── SMS_Detection/
│   ├── predict_sms_v2.py
│   ├── train.py
│   ├── evaluate_sms.py
│   └── evaluate_hinglish.py
│
├── URL_Detection/
│   ├── predict_url.py
│   └── ...
│
├── Network_Detection/
│   ├── predict_network.py
│   ├── train_network.py
│   └── ...
│
├── LLM/
│   └── executorch/
│       └── llama_advisory.py
│
├── LLM_Module/
│   └── ...
│
├── main.py
├── threat_fusion.py
├── test_fusion.py
└── README.md
```

The structure above shows the main project components. Model artifacts, datasets, native binaries, and runtime libraries may be maintained separately and must match the paths expected by the corresponding scripts.

---

## 11. INSTALLATION AND EXECUTION

### Step 1: Prepare the Raspberry Pi

Install Raspberry Pi OS and Python 3.

Create a project directory and navigate into it.

```bash
mkdir CybersecurityGuardian
cd CybersecurityGuardian
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

Install the Python dependencies required by the individual modules.

For example:

```bash
pip install numpy pandas scikit-learn joblib
```

Additional dependencies may be required depending on the final versions of the detection modules.

### Step 4: Configure the Detection Modules

Place the required trained model files, vectorizers, thresholds, and datasets in the paths expected by their respective modules.

The SMS, URL, and Network Detection modules must be configured independently before running the complete application.

### Step 5: Configure the Local LLM

Place the following files in the configured locations:

- ExecuTorch native runner.
- INT4 SpinQuant model.
- Tokenizer.
- Required shared libraries.

Verify that the paths configured in `LLM/executorch/llama_advisory.py` match the actual deployment paths on the Raspberry Pi.

### Step 6: Run the Application

Activate the virtual environment and execute:

```bash
python main.py
```

The application prompts the user for the relevant SMS, URL, and network-record inputs and produces a consolidated threat assessment and advisory.

**Note:** The installation commands above establish the Python environment. They do not automatically download or configure the model artifacts, datasets, native ExecuTorch runner, or all module-specific dependencies.

---

## 12. PERFORMANCE AND RESOURCE MEASUREMENTS

The system was tested on a Raspberry Pi 4 Model B.

The following results were observed during standalone Llama execution and complete application execution.

### 12.1 Standalone Llama Performance

| Parameter | Observed Result |
|---|---:|
| Model | Llama 3.2 1B Instruct |
| Quantization | INT4 SpinQuant |
| Runtime | ExecuTorch |
| CPU threads | 4 |
| Maximum new tokens configured | 128 |
| Prompt tokens | 61 |
| Generated tokens | 46 |
| Model loading time | Approximately 11.8 seconds |
| Inference interval | Approximately 15.05 seconds |
| Prompt processing speed | 14.37 tokens/second |
| Decode speed | 4.26 tokens/second |
| Total runner wall-clock time | 27.306 seconds |

The standalone measurement includes model loading, prompt processing, and generation.

The reported inference interval is approximately 15.05 seconds. The observed decode speed was approximately 4.26 tokens per second.

### 12.2 Complete Application Performance

A separate measurement was performed by executing the complete Python application.

| Parameter | Observed Result |
|---|---:|
| Application execution time | 84.47 seconds |
| Reported maximum child-process RSS | 1378.21 MB |
| Exit code | 0 |

The application test included SMS classification, Threat Fusion, and local Llama advisory generation. URL and network inputs were skipped in that particular run.

The memory value is the maximum resident-set-size value reported by the measurement script for its child process. It should not be interpreted as a verified combined process-tree peak or as an isolated Llama memory measurement.

### 12.3 Interpretation

These measurements demonstrate that the optimized language model can execute locally on Raspberry Pi hardware without using a cloud LLM inference API.

However, local inference is computationally demanding compared with conventional lightweight classification.

The measured performance also shows why model quantization, controlled token generation, and a smaller language model are important for edge deployment.

Performance may vary depending on input length, system load, model initialization, and hardware conditions.

---

## 13. OFFLINE CAPABILITY

The project is designed around local execution.

The following components use local processing:

- SMS model inference.
- URL feature extraction and model inference.
- Network model inference on supplied traffic records.
- Threat Fusion.
- Rule-based advisory generation.
- ExecuTorch Llama inference.

The URL detector implementation inspected uses local lexical feature extraction and a locally loaded model rather than an external URL reputation API.

The Llama advisory is also generated locally using the model and tokenizer stored on the Raspberry Pi.

No cloud LLM API is required for advisory generation.

**Offline operation still depends on the required software packages, model artifacts, datasets, and runtime libraries being available locally.** A complete end-to-end offline test should be performed to verify the behavior of the final assembled system.

---

## 14. LIMITATIONS

The current implementation has several limitations:

1. **SMS classification:** Machine-learning predictions may produce false positives and false negatives, particularly for OTP messages, mixed-language messages, and sophisticated phishing attempts.

2. **URL detection:** Lexical analysis cannot guarantee detection of every malicious URL. A URL that appears structurally normal may still lead to a malicious website.

3. **Network detection:** The current network module analyzes dataset-based traffic records rather than capturing and monitoring live network packets.

4. **LLM response reliability:** A language model may generate explanations that go beyond the evidence provided by the detection modules. Its output must not be treated as an independent security verdict.

5. **Inference latency:** Running a language model on Raspberry Pi CPU hardware introduces noticeable loading and inference delays.

6. **Hardware constraints:** Memory, CPU performance, and storage limitations influence the size and speed of models that can be deployed.

7. **Offline verification:** Although the main components use local processing, complete offline operation depends on all required dependencies and files being available on the device.

---

## 15. FUTURE SCOPE

The project can be extended in several directions:

- Improve SMS classification for OTP messages, Hinglish, and multilingual phishing attempts.
- Expand URL detection with additional lexical and contextual features.
- Integrate live network traffic monitoring and real-time anomaly detection.
- Improve Threat Fusion using confidence-aware and context-aware risk assessment.
- Further optimize LLM inference latency and memory usage.
- Improve advisory validation to prevent unsupported or misleading explanations.
- Introduce additional cybersecurity detection modules.
- Explore hardware acceleration for edge inference.
- Develop a graphical dashboard for displaying detection results and risk assessments.

---

## 16. CONCLUSION

The On-Device Cybersecurity Guardian demonstrates a modular approach to cybersecurity threat analysis using multiple machine-learning detection modules and centralized Threat Fusion.

Its key contribution is the integration of SMS, URL, and network threat analysis with a locally deployed language model that converts technical detection results into understandable security advisories.

By using Llama 3.2 1B Instruct with INT4 SpinQuant quantization and the ExecuTorch runtime, the project demonstrates local language-model execution on Raspberry Pi 4 hardware.

The addition of a rule-based fallback, controlled token generation, and modular detection architecture makes the system more suitable for resource-constrained edge deployment.

The project provides a foundation for developing accessible, locally executed cybersecurity assistance without requiring a cloud-based LLM service for its advisory component.


## 17. CONTRIBUTORS

Developed as part of the Bharat AI-SoC Challenge.

**Project:** On-Device Cybersecurity Guardian

**Hardware Platform:** Raspberry Pi 4 Model B

## 18. DISCLAIMER

This project is an educational cybersecurity prototype intended to demonstrate machine-learning-based threat detection and local AI-assisted advisory generation.

Its predictions are not guaranteed to identify every malicious message, URL, or network activity. The system should not replace professional cybersecurity tools or independent security verification.

Always verify suspicious messages and links through trusted official channels before taking action.
