import os
import asyncio
import io
import difflib  # Built-in library used for Levenshtein/Fuzzy matching string distances
from typing import Optional, Dict, Literal, List
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI(
    title="TTB Batch Compliance Engine",
    description="Locally hosted backend running custom Gemini Gem logic for label compliance verification.",
    version="3.0.0"
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set inside the container.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Limit concurrent Gemini API calls to maintain our sub-5 second latency targets
CONCURRENCY_LIMIT = 3
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


# ==========================================
# 1. DATA STRUCTURE DEFINITIONS
# ==========================================

class TTBLabelData(BaseModel):
    cola_id: str
    brand_name: str
    fanciful_name: str
    class_type: str
    abv: str
    net_contents: str
    country_of_origin: str
    government_warning_present: bool


class ComplianceResult(BaseModel):
    filename: str
    status: Literal["PASS", "REVIEW", "FAIL"]
    ai_confidence_pct: int  # Tracks extraction and matching confidence scores
    extracted_data: TTBLabelData
    approved_data_found: Optional[dict] = None
    discrepancies: List[str] = []


class BatchComplianceResult(BaseModel):
    total_processed: int
    passed_count: int
    review_count: int
    fail_count: int
    grouped_results: Dict[str, List[ComplianceResult]]


# ==========================================
# 2. MOCK REGISTRY DATABASE
# ==========================================
MOCK_APPROVED_COLA_DB: Dict[str, dict] = {
    "24012001000123": {
        "cola_id": "24012001000123",
        "brand_name": "CHATEAU PYTHON",
        "fanciful_name": "RESERVE COUPE",
        "class_type": "Grape Wine",
        "abv": "14.2%",
        "net_contents": "750 ml",
        "country_of_origin": "France"
    },
    "24012001000456": {
        "cola_id": "24012001000456",
        "brand_name": "FAST BEER CO",
        "fanciful_name": "ASYNC IPA",
        "class_type": "Malt Beverage",
        "abv": "6.5%",
        "net_contents": "12 fl. oz.",
        "country_of_origin": "USA"
    }
}


# ==========================================
# 3. FUZZY MATCHING & ENGINE LOGIC
# ==========================================

def calculate_fuzzy_match(str1: str, str2: str) -> float:
    """Calculates string similarity ratio via Gestalt Pattern Matching (similar to Levenshtein)."""
    return difflib.SequenceMatcher(None, str1.strip().lower(), str2.strip().lower()).ratio()


async def extract_label_data_with_gemini(file_bytes: bytes, mime_type: str) -> TTBLabelData:
    """Executes visual OCR using pure JSON instructions to completely bypass SDK schema bugs."""
    
    full_prompt = """
    You are an AI assistant acting as a pre-flight filter for a TTB Compliance Auditor. 
    Your goal is to quickly locate and extract text values from beverage labels so a human agent can verify them.
    
    You MUST return a valid JSON object containing exactly these keys:
    - "cola_id": (string) The 14-digit TTB COLA ID if visible on the label. Return an empty string "" if not found.
    - "brand_name": (string) The brand name of the beverage.
    - "fanciful_name": (string) The fanciful name. Return an empty string "" if not applicable.
    - "class_type": (string) The class or type designation (e.g., 'Grape Wine', 'Malt Beverage').
    - "abv": (string) The Alcohol by Volume percentage, e.g., '13.5%' or '6.5%'.
    - "net_contents": (string) The fluid volume statement, e.g., '750 ml' or '12 fl. oz.'.
    - "country_of_origin": (string) The country where the product originates.
    - "government_warning_present": (boolean) true if the paragraph starting with 'GOVERNMENT WARNING:' is physically visible on the label, otherwise false.

    Do not output any markdown code blocks, backticks, or extra text. Output ONLY the raw JSON object.
    """
    
    async with semaphore:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    contents=[image, full_prompt],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json", # Forces raw JSON execution mode
                        temperature=0.1
                    )
                )
            )
            # Pydantic cleanly compiles the raw text output string
            return TTBLabelData.model_validate_json(response.text)
        except Exception as e:
            print(f"CRITICAL EXTRACTION ERROR: {str(e)}")
            raise RuntimeError(f"Gemini processing error: {str(e)}")


def find_approved_record(extracted: TTBLabelData) -> Optional[dict]:
    """Finds matching record by COLA ID first, falling back to a Fuzzy Brand Name search."""
    if extracted.cola_id and extracted.cola_id.strip() != "":
        record = MOCK_APPROVED_COLA_DB.get(extracted.cola_id.strip())
        if record: return record
            
    best_match = None
    highest_score = 0.0
    
    for record in MOCK_APPROVED_COLA_DB.values():
        score = calculate_fuzzy_match(extracted.brand_name, record["brand_name"])
        if score > highest_score and score >= 0.80:
            highest_score = score
            best_match = record
            
    return best_match


def evaluate_compliance(filename: str, extracted: TTBLabelData, approved: Optional[dict]) -> ComplianceResult:
    """Compares metrics using fuzzy thresholds and compiles an AI confidence metric."""
    if not approved:
        return ComplianceResult(
            filename=filename,
            status="REVIEW",
            ai_confidence_pct=50,
            extracted_data=extracted,
            approved_data_found=None,
            discrepancies=["No approved baseline match found for this brand/COLA ID in the database."]
        )
        
    discrepancies = []
    scores = []

    if not extracted.government_warning_present:
        discrepancies.append("Compliance Error: Mandatory TTB Government Warning block missing.")
        scores.append(0.0)
    else:
        scores.append(1.0)

    brand_score = calculate_fuzzy_match(extracted.brand_name, approved["brand_name"])
    scores.append(brand_score)
    if brand_score < 1.0:
        if brand_score >= 0.85:
            discrepancies.append(f"Fuzzy Match Notification: Brand variation detected. Label says '{extracted.brand_name}', Database expects '{approved['brand_name']}'.")
        else:
            discrepancies.append(f"Brand Mismatch: Label text '{extracted.brand_name}' varies too far from approved '{approved['brand_name']}'.")

    if extracted.abv.strip().replace(" ", "") != approved["abv"].strip().replace(" ", ""):
        discrepancies.append(f"ABV Mismatch: Label says '{extracted.abv}', Database expects '{approved['abv']}'")
        scores.append(0.0)
    else:
        scores.append(1.0)
        
    if extracted.net_contents.strip().lower() != approved["net_contents"].strip().lower():
        discrepancies.append(f"Volume Mismatch: Label says '{extracted.net_contents}', Database expects '{approved['net_contents']}'")
        scores.append(0.0)
    else:
        scores.append(1.0)

    if extracted.country_of_origin.strip().lower() != approved["country_of_origin"].strip().lower():
        discrepancies.append(f"Country Mismatch: Label says '{extracted.country_of_origin}', Database expects '{approved['country_of_origin']}'")
        scores.append(0.0)
    else:
        scores.append(1.0)

    avg_score = sum(scores) / len(scores)
    ai_confidence = int(avg_score * 100)

    if any("Mismatch" in d or "Compliance Error" in d for d in discrepancies):
        status_verdict = "FAIL"
    elif discrepancies:
        status_verdict = "REVIEW"  
    else:
        status_verdict = "PASS"

    return ComplianceResult(
        filename=filename,
        status=status_verdict,
        ai_confidence_pct=ai_confidence,
        extracted_data=extracted,
        approved_data_found=approved,
        discrepancies=discrepancies
    )


async def process_single_label(file: UploadFile) -> ComplianceResult:
    """Safely runs processing tasks inside our concurrent worker pool loops."""
    try:
        content_type = file.content_type or "image/jpeg"
        file_bytes = await file.read()
        extracted = await extract_label_data_with_gemini(file_bytes, content_type)
        approved = find_approved_record(extracted)
        return evaluate_compliance(file.filename or "unknown_file", extracted, approved)
    except Exception as e:
        return ComplianceResult(
            filename=file.filename or "unknown_file",
            status="FAIL",
            ai_confidence_pct=0,
            extracted_data=TTBLabelData(
                cola_id="", brand_name="UNPARSEABLE", fanciful_name="", class_type="Unknown", 
                abv="Unknown", net_contents="Unknown", country_of_origin="Unknown", government_warning_present=False
            ),
            discrepancies=[f"System failed to parse label image: {str(e)}"]
        )


# ==========================================
# 4. ENVIRONMENT DIAGNOSTIC & API ENDPOINTS
# ==========================================

@app.get("/verify-environment")
async def verify_environment():
    """Marcus's 4-hour Environment Verification hook to test outward network clearance."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, 
            lambda: model.generate_content("Ping", generation_config=genai.GenerationConfig(max_output_tokens=1))
        )
        return {"status": "SUCCESS", "message": "Outbound API bridge clear. Firewall rules are operating correctly."}
    except Exception as e:
        return {
            "status": "BLOCKED", 
            "message": f"Outbound connection failed. Secure environment firewalls are blocking traffic: {str(e)}"
        }


@app.post("/verify-batch", response_model=BatchComplianceResult)
async def verify_batch(files: List[UploadFile] = File(...)):
    """Processes batch lists concurrently and organizes item records into unified brand segments."""
    tasks = [process_single_label(file) for file in files]
    results = await asyncio.gather(*tasks)

    passed, review, fail = 0, 0, 0
    grouped = defaultdict(list)

    for r in results:
        brand_key = r.extracted_data.brand_name.strip().upper()
        if not brand_key or brand_key == "UNPARSEABLE":
            brand_key = "UNKNOWN / UNCLASSIFIED"
            
        grouped[brand_key].append(r)
        
        if r.status == "PASS": passed += 1
        elif r.status == "REVIEW": review += 1
        else: fail += 1

    return BatchComplianceResult(
        total_processed=len(results),
        passed_count=passed,
        review_count=review,
        fail_count=fail,
        grouped_results=dict(grouped)
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serves an advanced interface containing confidence mapping arrays and override controllers."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TTB Operational Compliance Engine</title>
        <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen font-sans selection:bg-blue-600 selection:text-white">
        <div class="max-w-6xl mx-auto px-4 py-10">
            
            <header class="mb-10 flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-800 pb-6 gap-4">
                <div>
                    <h1 class="text-3xl font-black tracking-tight text-white">TTB Label-Check Engine</h1>
                    <p class="text-slate-400 text-sm mt-1">Pre-flight filter platform optimized for automated compliance auditing.</p>
                </div>
                <div class="flex items-center gap-3">
                    <button onclick="runNetworkDiagnostics()" class="text-xs bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 font-semibold px-4 py-2.5 rounded-xl transition">
                        🌐 Run Proxy Diagnostic
                    </button>
                    <div class="flex items-center bg-blue-600 border border-blue-500 px-5 py-2.5 rounded-xl cursor-pointer hover:bg-blue-700 transition shadow-lg shadow-blue-950">
                        <input type="file" id="batchFiles" multiple accept="image/*" class="hidden" />
                        <label for="batchFiles" class="cursor-pointer text-sm font-bold text-white flex items-center gap-2">
                            📥 Select Label Files
                        </label>
                    </div>
                </div>
            </header>

            <div id="metricsRow" class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10 hidden">
                <div class="bg-slate-900 p-4 border border-slate-800 rounded-xl"><p class="text-xs text-slate-500 font-bold uppercase tracking-wider">Total Staged</p><h3 id="mTotal" class="text-2xl font-black mt-1 text-white">0</h3></div>
                <div class="bg-slate-900 p-4 border border-slate-800 rounded-xl"><p class="text-xs text-emerald-500 font-bold uppercase tracking-wider">Approved (Pass)</p><h3 id="mPass" class="text-2xl font-black mt-1 text-emerald-400">0</h3></div>
                <div class="bg-slate-900 p-4 border border-slate-800 rounded-xl"><p class="text-xs text-amber-500 font-bold uppercase tracking-wider">Review Flagged</p><h3 id="mReview" class="text-2xl font-black mt-1 text-amber-400">0</h3></div>
                <div class="bg-slate-900 p-4 border border-slate-800 rounded-xl"><p class="text-xs text-rose-500 font-bold uppercase tracking-wider">Rejected (Fail)</p><h3 id="mFail" class="text-2xl font-black mt-1 text-rose-400">0</h3></div>
            </div>

            <main>
                <div id="loading" class="hidden text-center bg-slate-900 border border-slate-800 rounded-2xl p-16">
                    <div class="animate-spin inline-block w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mb-4"></div>
                    <p class="text-slate-300 font-medium text-base">Running image parsing engines & building verification clusters...</p>
                </div>

                <div id="emptyState" class="text-center bg-slate-900 border border-slate-800 border-dashed rounded-2xl p-20">
                    <p class="text-slate-500 max-w-sm mx-auto">Workspace vacant. Import a collection of beverage labels to generate localized audit pipelines.</p>
                </div>

                <div id="brandClusters" class="space-y-8"></div>
            </main>
        </div>

        <script>
            let globalMetrics = { total: 0, pass: 0, review: 0, fail: 0 };

            async function runNetworkDiagnostics() {
                alert("Initiating environment pathway diagnostics...");
                try {
                    const res = await fetch('/verify-environment');
                    const data = await res.json();
                    alert(`Diagnostic Output Status: ${data.status}\\n\\nDetail: ${data.message}`);
                } catch(e) {
                    alert("Fatal: Failed to connect to local app routing environment.");
                }
            }

            document.getElementById('batchFiles').addEventListener('change', async (e) => {
                const files = e.target.files;
                if (!files.length) return;

                document.getElementById('emptyState').classList.add('hidden');
                document.getElementById('brandClusters').innerHTML = '';
                document.getElementById('metricsRow').classList.add('hidden');
                document.getElementById('loading').classList.remove('hidden');

                const formData = new FormData();
                for (let i = 0; i < files.length; i++) {
                    formData.append('files', files[i]);
                }

                try {
                    const response = await fetch('/verify-batch', { method: 'POST', body: formData });
                    const payload = await response.json();
                    
                    globalMetrics.total = payload.total_processed;
                    globalMetrics.pass = payload.passed_count;
                    globalMetrics.review = payload.review_count;
                    globalMetrics.fail = payload.fail_count;
                    
                    renderBatchDashboard(payload);
                } catch (err) {
                    alert('Runtime structural compile crash. Review service engine standard out streams.');
                    console.error(err);
                } finally {
                    document.getElementById('loading').classList.add('hidden');
                }
            });

            function executeAgentOverride(cardId, initialVerdict) {
                const card = document.getElementById(cardId);
                const badge = card.querySelector('.verdict-badge');
                const alertBox = card.querySelector('.discrepancy-container');
                const overrideBtn = card.querySelector('.override-action-hook');

                badge.className = "verdict-badge text-[10px] font-black tracking-wider border px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border-emerald-900";
                badge.innerText = "PASS (OVERRIDDEN)";
                
                if (alertBox) alertBox.classList.add('hidden');
                if (overrideBtn) overrideBtn.classList.add('hidden');

                if (initialVerdict === 'FAIL') globalMetrics.fail--;
                if (initialVerdict === 'REVIEW') globalMetrics.review--;
                globalMetrics.pass++;

                document.getElementById('mPass').innerText = globalMetrics.pass;
                document.getElementById('mReview').innerText = globalMetrics.review;
                document.getElementById('mFail').innerText = globalMetrics.fail;
            }

            function renderBatchDashboard(data) {
                document.getElementById('metricsRow').classList.remove('hidden');
                document.getElementById('mTotal').innerText = globalMetrics.total;
                document.getElementById('mPass').innerText = globalMetrics.pass;
                document.getElementById('mReview').innerText = globalMetrics.review;
                document.getElementById('mFail').innerText = globalMetrics.fail;

                const clusterContainer = document.getElementById('brandClusters');
                let cardIndexTracker = 0;

                Object.keys(data.grouped_results).forEach(brandName => {
                    const brandGroup = data.grouped_results[brandName];
                    const brandSection = document.createElement('section');
                    brandSection.className = "bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden p-6 shadow-xl";

                    brandSection.innerHTML = `
                        <div class="border-b border-slate-800 pb-4 mb-4 flex justify-between items-center bg-slate-900/50">
                            <div>
                                <span class="text-[10px] font-bold tracking-widest text-blue-400 uppercase bg-blue-950 px-2 py-0.5 rounded border border-blue-900">Brand Cluster Group</span>
                                <h2 class="text-xl font-black text-white mt-1">${brandName}</h2>
                            </div>
                            <span class="text-xs text-slate-400 font-mono bg-slate-950 border border-slate-800 px-3 py-1 rounded-full">
                                ${brandGroup.length} items cataloged
                            </span>
                        </div>
                        <div class="space-y-4" id="cards-${brandName.replace(/[^a-zA-Z0-9]/g, '')}"></div>
                    `;

                    clusterContainer.appendChild(brandSection);
                    const cardsMount = document.getElementById(`cards-${brandName.replace(/[^a-zA-Z0-9]/g, '')}`);

                    brandGroup.forEach(item => {
                        cardIndexTracker++;
                        const currentCardId = `compliance-card-node-${cardIndexTracker}`;
                        const fileCard = document.createElement('div');
                        fileCard.id = currentCardId;
                        fileCard.className = "bg-slate-950 border border-slate-800 p-5 rounded-xl space-y-4 transition hover:border-slate-700";

                        let badgeColor = "bg-rose-950 text-rose-400 border-rose-900";
                        if(item.status === 'PASS') badgeColor = "bg-emerald-950 text-emerald-400 border-emerald-900";
                        if(item.status === 'REVIEW') badgeColor = "bg-amber-950 text-amber-400 border-amber-900";

                        let discrepanciesHtml = '';
                        let actionButtonHtml = '';
                        
                        if (item.status !== 'PASS') {
                            actionButtonHtml = `
                                <button onclick="executeAgentOverride('${currentCardId}', '${item.status}')" class="override-action-hook text-xs bg-slate-900 border border-slate-800 text-slate-300 font-bold px-3 py-1.5 rounded-lg hover:bg-slate-800 hover:text-white transition">
                                    ✍️ Override & Approve Label
                                </button>
                            `;
                        }

                        if (item.discrepancies && item.discrepancies.length > 0) {
                            discrepanciesHtml = `
                                <div class="discrepancy-container bg-red-950/20 border border-red-950 p-4 rounded-xl text-xs text-red-300 space-y-2">
                                    <p class="font-bold uppercase tracking-wider text-red-400">Flagged Exceptions Matrix:</p>
                                    <ul class="list-disc pl-4 space-y-1">
                                        ${item.discrepancies.map(d => `<li>${d}</li>`).join('')}
                                    </ul>
                                </div>
                            `;
                        }

                        let displayCola = item.extracted_data.cola_id === "" ? "—" : item.extracted_data.cola_id;
                        let displayFanciful = item.extracted_data.fanciful_name === "" ? "—" : item.extracted_data.fanciful_name;

                        fileCard.innerHTML = `
                            <div class="flex justify-between items-start gap-4">
                                <div class="space-y-1">
                                    <span class="text-xs font-mono text-slate-300 font-bold bg-slate-900 border border-slate-800/80 px-2 py-1 rounded">📄 ${item.filename}</span>
                                    <div class="text-[11px] text-slate-500 pt-1">
                                        AI Engine Match Confidence: <span class="font-mono text-blue-400 font-bold">${item.ai_confidence_pct}% confident</span>
                                    </div>
                                </div>
                                <div class="flex items-center gap-2">
                                    ${actionButtonHtml}
                                    <span class="verdict-badge text-[10px] font-black tracking-wider border px-2.5 py-0.5 rounded-full ${badgeColor}">${item.status}</span>
                                </div>
                            </div>
                            
                            ${discrepanciesHtml}
                            
                            <div class="grid grid-cols-2 sm:grid-cols-6 gap-3 text-[11px] pt-2 border-t border-slate-900">
                                <div><span class="text-slate-500 block uppercase tracking-wider text-[9px]">COLA ID</span><span class="text-slate-200 font-mono">${displayCola}</span></div>
                                <div><span class="text-slate-500 block uppercase tracking-wider text-[9px]">Fanciful Name</span><span class="text-slate-200">${displayFanciful}</span></div>
                                <div><span class="text-slate-500 block uppercase tracking-wider text-[9px]">Class Type</span><span class="text-slate-200">${item.extracted_data.class_type}</span></div>
                                <div><span class="text-slate-500 block uppercase tracking-wider text-[9px]">ABV Statement</span><span class="text-slate-200 font-semibold text-blue-400">${item.extracted_data.abv}</span></div>
                                <div><span class="text-slate-500 block uppercase tracking-wider text-[9px]">Net Contents</span><span class="text-slate-200">${item.extracted_data.net_contents}</span></div>
                                <div><span class="text-slate-500 block uppercase tracking-wider text-[9px]">Origin Country</span><span class="text-slate-200">${item.extracted_data.country_of_origin}</span></div>
                            </div>
                        `;
                        cardsMount.appendChild(fileCard);
                    });
                });
            }
        </script>
    </body>
    </html>
    """
