# 🎓 JEE AI Counselor

An intelligent college and branch recommendation system that goes beyond simple rank-based cutoffs. The JEE AI Counselor considers a student's **career goals, subject interests, and personal priorities** to recommend the best possible engineering branches and institutes, powered by AI explanations and a dynamic risk classifier.

---

## 🚀 Features

- **Beyond-Rank Recommendations:** Instead of just filtering historical cutoffs, the engine scores every eligible branch against the student's *Career Persona*, *Coding Interest*, *Research Goals*, and *Location Flexibility*.
- **AI-Powered Explanations:** Uses generative AI (Groq, Anthropic Claude, or Google Gemini) to provide personalised justifications for *why* a specific institute and branch is recommended, along with a custom career roadmap.
- **Dynamic Risk Classification:** Automatically categorizes chances into *Dream*, *Target*, *Safe*, and *Very Safe* using a sigmoid-based probability model adjusted for historical cutoff variance.
- **Interactive Branch Comparison:** Head-to-head comparison of branches (e.g., *Computer Science vs. Mathematics & Computing*) highlighting differences in median salary, coding culture, and MBA transition prospects.
- **AI Chat Counselor:** A conversational interface grounded strictly in the JoSAA knowledge base. Students can ask follow-up questions like "Why not IIT Kanpur?" and get answers based on data, not hallucinations.

## 🧠 Algorithm Approach

The recommendation engine goes beyond traditional rank-filtering by employing a multi-dimensional scoring algorithm:

1. **Persona Inference:** The student's inputs (interest sliders, career goals) are mapped to predefined career personas (e.g., *AI Researcher*, *Core Engineering Leader*, *Startup Founder*).
2. **Initial Filtering (DuckDB):** Fetches historically eligible cutoff data from DuckDB, applying a 35% rank buffer to catch "Dream" possibilities.
3. **Multi-Factor Scoring (`scorer.py`):**
   - **Rank Fit (40%):** Proximity of the student's rank to the historical closing rank.
   - **Interest Match (25%):** Alignment of the branch's subject matter with the student's sliders.
   - **Institute Strength (15%):** Extracted from `institute_tiers.json`, factoring in NIRF rank, median placement, and research output.
   - **Career Alignment (12%):** How well the branch's typical outcomes align with the student's long-term goals (MBA, Startup, MS).
   - **Home State Bonus (5%) & Flexibility (3%).**
4. **Risk Classification (`risk_classifier.py`):** Uses a sigmoid function over the normalized rank gap to compute a true probabilistic chance of admission, classifying options into *Dream* (< 45%), *Target* (45–75%), *Safe* (75–90%), and *Very Safe* (> 90%).
5. **LLM Explainer (RAG):** The top options are packed into a `RecommendationContext` with localized facts (median salaries, top recruiters) and passed to the LLM to generate the final human-readable justification.

## 🏗️ Architecture

The application is built on a fast, lightweight, and modern stack designed for high throughput during counseling season:

1. **Frontend:** Single-page application built with React (CDN) and modern CSS. No build step required, served directly by the backend.
2. **Backend API:** High-performance REST API built with **FastAPI**.
3. **Recommendation Engine:** Custom scoring algorithm (`scorer.py`) that matches student interests against JSON-based branch profiles and institute metadata.
4. **Data Layer:** Uses **DuckDB** for blazing-fast in-memory analytics of historical JoSAA cutoffs, avoiding the overhead of heavy RDBMS for read-heavy operations.
5. **RAG Pipeline:** A specialized Retrieval-Augmented Generation module (`rag.py`) that anchors LLM responses strictly to internal knowledge bases.

## 🛠️ Technology Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic
- **Database:** DuckDB, Pandas
- **Frontend:** React 18, Babel (Standalone), Vanilla CSS
- **AI & RAG:** Groq API / Anthropic API / Google Gemini API (with Mock fallback)
- **Deployment:** Docker, Docker Compose

## ⚙️ Setup & Installation

### Option 1: Using Docker (Recommended)

1. Clone the repository and navigate to the project root.
2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
3. Add your AI API key to `.env` (e.g., `GROQ_API_KEY=your_key_here` and `LLM_PROVIDER=groq`).
4. Start the application:
   ```bash
   docker-compose up -d --build
   ```
5. Open `http://localhost:8000/ui` in your browser.

### Option 2: Local Development

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -e .
   ```
2. Set up environment variables (`.env`).
3. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Access the UI at `http://localhost:8000/ui` and API docs at `http://localhost:8000/docs`.

## 📡 Key API Endpoints

- `POST /recommend`: Generates personalised institute recommendations based on the student's profile.
- `GET /compare/branches`: Returns a head-to-head comparison of two engineering branches.
- `POST /counselor/chat`: AI chat endpoint that answers follow-up questions using context from the recommendation engine.
- `GET /health`: Health check and system readiness status.

## 🔮 Future Work

- **Vector Database Integration:** Migrate the static JSON knowledge base to a vector store (e.g., Qdrant or Milvus) for semantic search capabilities in the AI Counselor.
- **Round-wise Analytics:** Add predictive modeling for cutoffs across all 6 JoSAA rounds.
- **PDF Export:** Implement a backend service using ReportLab to generate downloadable, professional PDF reports of the recommendations.
- **User Accounts:** Add persistent user sessions and saved preference lists.
