import os
import json
from datetime import datetime
from fpdf import FPDF, XPos, YPos

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKER_DATA_PATH = os.path.join(PROJECT_ROOT, "outputs", "tracker", "tracker_state.json")
OUTPUT_PDF_PATH = os.path.join(PROJECT_ROOT, "outputs", "tracker", "ES335_PCam_Project_Progress_Tracker.pdf")

INITIAL_DATA = {
    "project_title": "Comparing Machine Learning Approaches for Tumour Detection in Tissue Patches",
    "course": "ES 335 - Machine Learning",
    "team": [
        {"name": "Devarsh Patel", "roll": "24110245"},
        {"name": "Dhrumil Bhutaiya", "roll": "24110075"}
    ],
    "last_updated": datetime.now().strftime("%d %B %Y, %I:%M %p IST"),
    "dataset_info": "PatchCamelyon (PCam) - 327,680 RGB patches (96x96 px) | 262,144 train, 32,768 val, 32,768 test",
    "current_phase": 1,
    "phases": [
        {
            "id": 0,
            "title": "Phase 0: Proposal and Registration",
            "status": "In Progress (3/4)",
            "tasks": [
                {"desc": "Team registered on ES 335 portal (two members, shared proposal record)", "done": True},
                {"desc": "Proposal written and submitted (Rev 141, deadline 15 Oct 2026, 11:59 PM IST)", "done": True},
                {"desc": "Dataset, baseline, candidate methods, evaluation plan and website idea fixed", "done": True},
                {"desc": "Re-open portal and confirm proposal still shows as submitted. Note dates of next deliverables", "done": False}
            ]
        },
        {
            "id": 1,
            "title": "Phase 1: Setup and Data Access (CURRENT FOCUS)",
            "status": "In Progress (1/7)",
            "tasks": [
                {"desc": "Initialize clean repo layout: data/, notebooks/, src/, features/, results/, docs/, tracker/", "done": True},
                {"desc": "Create requirements.txt with all dependencies and pin versions", "done": False},
                {"desc": "Download PCam dataset from official source (basveeling/pcam or TFDS)", "done": False},
                {"desc": "Verify published split sizes (262,144 train, 32,768 val, 32,768 test)", "done": False},
                {"desc": "Write data loader function in src/data.py with fixed global random seed", "done": False},
                {"desc": "Configure .gitignore to protect data/ and large feature files", "done": False},
                {"desc": "Decide compute strategy & stratified subsampling budget for heavy models (RBF SVM)", "done": False}
            ]
        },
        {
            "id": 2,
            "title": "Phase 2: Exploratory Data Analysis (EDA)",
            "status": "Not Started (0/5)",
            "tasks": [
                {"desc": "Class balance analysis per split (tumour vs normal)", "done": False},
                {"desc": "Visualization grid of tumour/normal patches with central 32x32 region bounding box", "done": False},
                {"desc": "Per-class color channel distributions (RGB and HSV) and mean intensity", "done": False},
                {"desc": "Duplicate/corrupted patches check and documentation of slide-level leakage caveat", "done": False},
                {"desc": "Export top 3-4 figures into docs/img/ for report & website", "done": False}
            ]
        },
        {
            "id": 3,
            "title": "Phase 3: Features & Dimensionality Reduction",
            "status": "Not Started (0/7)",
            "tasks": [
                {"desc": "Representation A: Raw pixels scaled [0, 1] (27,648-dim) + downsampled version", "done": False},
                {"desc": "Representation B: Hand-crafted features (color stats, LBP/GLCM texture, HOG edges)", "done": False},
                {"desc": "Generate both representations for full 96x96 patch AND central 32x32 crop", "done": False},
                {"desc": "Persist matrices to features/ with deterministic versioning", "done": False},
                {"desc": "Fit StandardScaler strictly on train set, transform val and test", "done": False},
                {"desc": "Fit PCA on train only, evaluate 90%, 95%, 99% variance thresholds", "done": False},
                {"desc": "Sanity checks: exact shapes, no NaNs, row-to-label alignment", "done": False}
            ]
        },
        {
            "id": 4,
            "title": "Phase 4: Baseline - Logistic Regression",
            "status": "Not Started (0/4)",
            "tasks": [
                {"desc": "Shared evaluate(model, X, y) module (ROC-AUC, PR-AUC, F1, specificity, FPR, CM)", "done": False},
                {"desc": "Train baseline logistic regression on raw & hand-crafted features (+/- PCA)", "done": False},
                {"desc": "Chance verification (ROC-AUC > 0.5) to validate pipeline soundness", "done": False},
                {"desc": "Log metrics to results/baseline.csv", "done": False}
            ]
        },
        {
            "id": 5,
            "title": "Phase 5: Candidate Models Exploration",
            "status": "Not Started (0/7)",
            "tasks": [
                {"desc": "Linear SVM evaluation", "done": False},
                {"desc": "Non-linear RBF SVM on PCA features / stratified subsample", "done": False},
                {"desc": "Random Forest ensemble", "done": False},
                {"desc": "Gradient Boosted Trees (HistGradientBoosting / XGBoost)", "done": False},
                {"desc": "Track wall-clock training time, inference latency, and peak memory", "done": False},
                {"desc": "Optional loss function comparison using SGDClassifier", "done": False},
                {"desc": "Optional lightweight CNN upper-bound benchmark", "done": False}
            ]
        },
        {
            "id": 6,
            "title": "Phase 6: Hyperparameter Tuning (Validation Set Only)",
            "status": "Not Started (0/5)",
            "tasks": [
                {"desc": "Define search space grids and computational budget per model family", "done": False},
                {"desc": "Run systematic tuning on validation split only, log all trials", "done": False},
                {"desc": "Freeze optimal configuration per family into config file", "done": False},
                {"desc": "Dimensionality ablation study (metric vs PCA components vs training time)", "done": False},
                {"desc": "Synthesize 3-4 key early insights in findings log", "done": False}
            ]
        },
        {
            "id": 7,
            "title": "Phase 7: Final Test Evaluation (ONE-SHOT)",
            "status": "Not Started (0/7)",
            "tasks": [
                {"desc": "Retrain frozen models on train/val as designed, evaluate test set EXACTLY ONCE", "done": False},
                {"desc": "Comprehensive metric extraction (ROC-AUC, PR-AUC, Precision, Recall, Specificity, FPR)", "done": False},
                {"desc": "Combined ROC and Precision-Recall multi-model curves", "done": False},
                {"desc": "Decision threshold sensitivity vs false-positive trade-off analysis", "done": False},
                {"desc": "Probability calibration curves and Brier score evaluation", "done": False},
                {"desc": "Bootstrap 95% confidence intervals on core metrics", "done": False},
                {"desc": "Explicit evaluation of research questions: model complexity & PCA efficiency", "done": False}
            ]
        },
        {
            "id": 8,
            "title": "Phase 8: Error Analysis and Diagnostic Insights",
            "status": "Not Started (0/5)",
            "tasks": [
                {"desc": "Isolate false positives and false negatives into image inspection grids", "done": False},
                {"desc": "Categorize shared consensus errors vs model-specific failures", "done": False},
                {"desc": "Extract feature importances (LogReg coefficients, Tree importances)", "done": False},
                {"desc": "Formulate 3-5 concrete pathology insights (stain variation, boundary ambiguities)", "done": False},
                {"desc": "Formally document slide-overlap limitation in project discussion", "done": False}
            ]
        },
        {
            "id": 9,
            "title": "Phase 9: Interactive Showcase Website (GitHub Pages)",
            "status": "Not Started (0/9)",
            "tasks": [
                {"desc": "Select deployment architecture (static JS + precomputed probabilities)", "done": False},
                {"desc": "Export curated test patch predictions and model probabilities to JSON", "done": False},
                {"desc": "Build Page 1: Overview, research question, PCam histology details", "done": False},
                {"desc": "Build Page 2: Interactive patch inspector with model selection & live proba", "done": False},
                {"desc": "Interactive threshold slider with real-time sensitivity / FPR indicator", "done": False},
                {"desc": "Build interactive error gallery with diagnostic commentary", "done": False},
                {"desc": "Build results dashboard with interactive curves & EDA charts", "done": False},
                {"desc": "Mobile & desktop responsive QA", "done": False},
                {"desc": "Deploy to GitHub Pages via docs/ branch and verify live URL", "done": False}
            ]
        },
        {
            "id": 10,
            "title": "Phase 10: Final Synthesis, Report & Submission",
            "status": "Not Started (0/6)",
            "tasks": [
                {"desc": "Audit portal rubric and confirm all deliverable criteria are met", "done": False},
                {"desc": "Write comprehensive production README (setup, pipeline, folder map)", "done": False},
                {"desc": "Draft full technical report (methods, results, CIs, error analysis, limitations)", "done": False},
                {"desc": "Clean environment reproducibility pass (fresh clone + notebook execution)", "done": False},
                {"desc": "Peer cross-review of report and interactive website", "done": False},
                {"desc": "Submit on course portal and archive submission confirmation screenshot", "done": False}
            ]
        }
    ],
    "decisions_log": [
        {
            "date": "17 Oct 2026",
            "decision": "Established dedicated project directory 'pcam_tumor_detection' with modular folders.",
            "rationale": "Separates data, notebooks, modular src, features, and docs to avoid clutter across all 11 phases."
        },
        {
            "date": "17 Oct 2026",
            "decision": "Integrated automated progress PDF generation in tracker/.",
            "rationale": "Enables continuous, non-intrusive progress documentation after every major working session."
        }
    ]
}

class ProjectTrackerPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(100, 110, 120)
        self.cell(0, 5, 'ES 335: Machine Learning - PCam Tumour Detection Project Tracker', align='L')
        self.cell(0, 5, f'Updated: {datetime.now().strftime("%d-%b-%Y")}', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(210, 215, 225)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 6, f'Page {self.page_no()}/{{nb}} | Automated Progress Log - pcam_tumor_detection', align='C')

def generate_pdf():
    if not os.path.exists(TRACKER_DATA_PATH):
        with open(TRACKER_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(INITIAL_DATA, f, indent=2)
        data = INITIAL_DATA
    else:
        with open(TRACKER_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    pdf = ProjectTrackerPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title & Header
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 8, data['project_title'], new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(70, 90, 120)
    pdf.cell(0, 6, f"Course: {data['course']} | Snapshot: {data['last_updated']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')

    team_str = "Team: " + ", ".join([f"{m['name']} ({m['roll']})" for m in data['team']])
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 5, team_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, f"Dataset: {data['dataset_info']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Executive Progress Box
    total_tasks = sum(len(p['tasks']) for p in data['phases'])
    done_tasks = sum(sum(1 for t in p['tasks'] if t['done']) for p in data['phases'])
    pct = (done_tasks / total_tasks) * 100

    pdf.set_fill_color(240, 245, 255)
    pdf.set_draw_color(180, 205, 240)
    pdf.rect(10, pdf.get_y(), 190, 16, 'DF')
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(90, 6, f"Overall Progress: {done_tasks} / {total_tasks} Tasks Complete ({pct:.1f}%)")
    curr_phase_title = next((p['title'] for p in data['phases'] if p['id'] == data['current_phase']), "Phase 1")
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(180, 50, 20)
    pdf.cell(90, 6, f"Active Focus: {curr_phase_title.split(':')[0]}", align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Progress Bar inside box
    bar_x = 14
    bar_y = pdf.get_y() + 1
    pdf.set_draw_color(160, 175, 195)
    pdf.set_fill_color(220, 225, 235)
    pdf.rect(bar_x, bar_y, 182, 4, 'F')
    pdf.set_fill_color(34, 150, 70)
    pdf.rect(bar_x, bar_y, 182 * (done_tasks / total_tasks), 4, 'F')
    pdf.set_y(bar_y + 8)

    # Golden Rule Alert
    pdf.set_fill_color(255, 248, 235)
    pdf.set_draw_color(230, 180, 80)
    pdf.rect(10, pdf.get_y(), 190, 10, 'DF')
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(150, 75, 0)
    pdf.cell(0, 6, "GOLDEN PROTOCOL: Test set is touched ONCE in Phase 7. Scalers, PCA and tuning use train/val only.")
    pdf.set_y(pdf.get_y() + 9)

    # Phases & Tasks
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 7, "Phased Execution Roadmap", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for phase in data['phases']:
        is_curr = (phase['id'] == data['current_phase'])
        pdf.set_font('Helvetica', 'B', 9.5)
        if is_curr:
            pdf.set_fill_color(225, 238, 255)
            pdf.set_text_color(10, 50, 140)
            status_tag = f" [YOU ARE HERE - {phase['status']}]"
        else:
            pdf.set_fill_color(245, 247, 250)
            pdf.set_text_color(40, 40, 50)
            status_tag = f" [{phase['status']}]"

        pdf.cell(150, 6, f"  {phase['title']}", fill=True)
        pdf.set_font('Helvetica', 'I', 8.5)
        pdf.cell(40, 6, status_tag, fill=True, align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font('Helvetica', '', 8.5)
        for t in phase['tasks']:
            pdf.set_x(14)
            if t['done']:
                pdf.set_text_color(30, 130, 50)
                pdf.set_font('Helvetica', 'B', 8.5)
                box_str = "[X] "
            else:
                pdf.set_text_color(90, 95, 100)
                pdf.set_font('Helvetica', '', 8.5)
                box_str = "[  ] "
            
            pdf.cell(8, 4.5, box_str)
            pdf.set_text_color(40, 45, 55)
            pdf.multi_cell(168, 4.5, t['desc'])
        pdf.ln(1.5)

    # Decisions Log Section
    if pdf.get_y() > 220:
        pdf.add_page()
    else:
        pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 6, "Engineering Decisions & Architecture Log", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(230, 235, 245)
    pdf.cell(24, 6, "Date", border=1, fill=True)
    pdf.cell(86, 6, "Decision", border=1, fill=True)
    pdf.cell(80, 6, "Rationale / Impact", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font('Helvetica', '', 7.5)
    for dec in data['decisions_log']:
        date_txt = dec['date']
        dec_txt = dec['decision']
        rat_txt = dec['rationale']
        
        # calculate max height
        pdf.cell(24, 8, date_txt, border=1)
        curr_x = pdf.get_x()
        curr_y = pdf.get_y()
        pdf.cell(86, 8, dec_txt[:60] + ("..." if len(dec_txt) > 60 else ""), border=1)
        pdf.cell(80, 8, rat_txt[:60] + ("..." if len(rat_txt) > 60 else ""), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(OUTPUT_PDF_PATH)
    print(f"Progress PDF generated successfully at: {OUTPUT_PDF_PATH}")

def update_task_status(phase_id, task_idx, done=True):
    with open(TRACKER_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for p in data['phases']:
        if p['id'] == phase_id:
            if 0 <= task_idx < len(p['tasks']):
                p['tasks'][task_idx]['done'] = done
                done_cnt = sum(1 for t in p['tasks'] if t['done'])
                tot_cnt = len(p['tasks'])
                if done_cnt == tot_cnt:
                    p['status'] = f"Complete ({done_cnt}/{tot_cnt})"
                elif done_cnt == 0:
                    p['status'] = f"Not Started (0/{tot_cnt})"
                else:
                    p['status'] = f"In Progress ({done_cnt}/{tot_cnt})"
    
    data['last_updated'] = datetime.now().strftime("%d %B %Y, %I:%M %p IST")
    with open(TRACKER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    generate_pdf()

def record_decision(decision_text, rationale_text):
    with open(TRACKER_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data['decisions_log'].append({
        "date": datetime.now().strftime("%d %b %Y"),
        "decision": decision_text,
        "rationale": rationale_text
    })
    data['last_updated'] = datetime.now().strftime("%d %B %Y, %I:%M %p IST")
    with open(TRACKER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    generate_pdf()

if __name__ == "__main__":
    generate_pdf()

