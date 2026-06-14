# TTB Label-Check Engine

[cite_start]A containerized pre-flight filter web application designed for the TTB Compliance Division to optimize the label auditing process[cite: 2, 3, 21]. [cite_start]This prototype balances trust, speed, and usability to help compliance agents surface discrepancies quickly while retaining human judgment[cite: 1, 4].

---

## 🚀 Key Features

* [cite_start]**Pre-Flight Filtering:** Designed to quickly highlight "low-hanging fruit" discrepancies so senior agents can reserve cognitive energy for nuanced judgment calls[cite: 3, 4].
* [cite_start]**Sub-5 Second Processing:** Built on an asynchronous processing pipeline to guarantee rapid feedback from multi-file batch uploads[cite: 10, 11, 31].
* [cite_start]**Fuzzy Text Matching:** Implements Gestalt Pattern Matching (similarity ratio metrics) to gracefully handle case variations, punctuation drops, and minor typos in brand names[cite: 17, 29].
* [cite_start]**Dave's "Secret Sauce" Agent Override:** Protects professional agency by providing interactive, manual override triggers that allow agents to sign off on a label even if the AI flags a warning[cite: 35, 37, 38].
* [cite_start]**Clear UX Hierarchy:** Employs a high-contrast, large-button design passing the "73-Year-Old Test" with intuitive Green/Yellow/Red (Pass/Review/Fail) signaling[cite: 12, 13, 14, 15].
* [cite_start]**Marcus's Firewall Proxy Diagnostic:** Includes a dedicated environment verification endpoint to verify outbound network clearance and ensure compliance with strict infrastructure firewall architectures[cite: 19, 20, 25, 26].

---

## 🛠️ Architecture & Tech Stack

* **Backend Engine:** Python / FastAPI (Async concurrency control via Semaphores)
* **Vision Core:** Google Gemini 2.5 Flash via native Pillow Image processing streams
* **Frontend Layer:** HTML5 / JavaScript (ES6) / Tailwind CSS 
* [cite_start]**Deployment System:** Docker Containerization [cite: 21]

---

## 💻 Quick Start & Deployment

### 1. Prerequisites
[cite_start]Ensure you have **Docker** installed on your host machine[cite: 21], and procure an API key from **Google AI Studio**.

### 2. Build the Container Environment
[cite_start]Clone this repository locally, navigate to the root directory, and run the following command to compile the isolated environment[cite: 21]:
```bash
docker build -t ttb-verifier:latest .

#### 3.
docker run -d \
  -p 8080:8000 \
  -e GEMINI_API_KEY="YOUR_ACTUAL_API_KEY_HERE" \
  --name ttb-compliance-app \
  ttb-verifier:latest

### 4.
http://localhost:8080/
