"""
POST /counselor/chat — AI Counselor follow-up questions

Allows natural language follow-up after recommendations.
Grounds every answer in: student profile + knowledge base + session data.
Never answers from generic internet knowledge.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import knowledge_loader, rag_pipeline
from app.config import get_settings
from app.models.request import StudentProfile
from app.services.knowledge_loader import KnowledgeLoader
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


class CounselorChatRequest(BaseModel):
    student:    StudentProfile
    question:   str  = Field(..., min_length=5, max_length=500)
    history:    list[dict] = Field(default_factory=list)  # [{"role": "user/ai", "text": "..."}]
    session_id: str | None = None   # if set, uses saved session context

    model_config = {
        "json_schema_extra": {
            "example": {
                "student": {
                    "jee_advanced_rank": 3500,
                    "gender": "male",
                    "category": "OPEN",
                    "home_state": "Maharashtra",
                    "interest_coding": 0.9,
                    "interest_ai_ml": 0.8,
                    "interest_research": 0.4,
                    "interest_core_engineering": 0.1,
                    "interest_electronics": 0.3,
                    "interest_mechanical": 0.0,
                    "interest_civil": 0.0,
                    "interest_chemical": 0.0,
                    "wants_startup": True,
                    "salary_priority": 0.8,
                },
                "question": "Why not IIT Kanpur CSE? And is MnC better than CSE?",
            }
        }
    }


class CounselorChatResponse(BaseModel):
    question:  str
    answer:    str
    sources:   list[str]  # which knowledge sources were used
    faq_match: dict | None = None


def _build_counselor_system_prompt(student: StudentProfile, kl: KnowledgeLoader) -> str:
    """Build grounded system prompt for the AI counselor."""
    from app.engine.persona import infer_persona
    from app.engine.interest_matcher import get_top_branches_for_student

    persona    = infer_persona(student)
    top_branches = get_top_branches_for_student(student, top_n=3)
    weights    = kl.scorer_weights()

    top_branch_str = ", ".join(b["branch"] for b in top_branches)

    return f"""You are a JEE admission counselor. Answer questions using ONLY the information provided below.
Do not use general internet knowledge. If information is not in the context, say so honestly.

STUDENT PROFILE:
- Rank: {student.effective_rank} ({student.category} category)
- Gender: {student.gender}
- Home state: {student.home_state}
- Career persona: {persona.label} (confidence: {persona.confidence:.0%})
- Active goals: {', '.join(student.active_goals) or 'not specified'}
- Interest in coding: {student.interest_coding:.0%}
- Interest in AI/ML: {student.interest_ai_ml:.0%}
- Interest in research: {student.interest_research:.0%}
- Interest in core engineering: {student.interest_core_engineering:.0%}
- Top branch matches by interest: {top_branch_str}

SCORING SYSTEM USED:
Recommendations are scored using: {weights}

CRITICAL RULES:
1. Be specific, honest, and reference the student's actual profile when answering.
2. Keep answers under 150 words. Use plain language, not jargon.
3. Never recommend something just because it sounds good — ground every claim in the data above.
4. ANTI-HALLUCINATION GUARDRAIL: If the user asks a question entirely unrelated to engineering, colleges, careers, or the JEE admission process, politely decline to answer."""


def _get_grounded_context(question: str, kl: KnowledgeLoader) -> tuple[str, list[str]]:
    """
    Retrieve relevant knowledge base context for the question.
    Returns (context_string, list_of_sources_used).
    """
    context_parts = []
    sources = []
    q_lower = question.lower()

    # FAQ match
    faq_results = kl.search_faq(question)
    if faq_results:
        best_faq = faq_results[0]
        context_parts.append(f"RELEVANT FAQ:\nQ: {best_faq['question']}\nA: {best_faq['answer']}")
        sources.append("faq.json")

    # Branch-specific context
    branch_keywords = {
        "cse": "Computer Science and Engineering",
        "cs":  "Computer Science and Engineering",
        "computer science": "Computer Science and Engineering",
        "mnc": "Mathematics and Computing",
        "mathematics": "Mathematics and Computing",
        "ece": "Electronics and Communication Engineering",
        "electronics": "Electronics and Communication Engineering",
        "mechanical": "Mechanical Engineering",
        "civil": "Civil Engineering",
        "chemical": "Chemical Engineering",
        "ee": "Electrical Engineering",
        "electrical": "Electrical Engineering",
        "ep": "Engineering Physics",
        "physics": "Engineering Physics",
    }
    for kw, branch_name in branch_keywords.items():
        if kw in q_lower:
            profile = kl.get_branch(branch_name)
            career  = kl.get_branch_career(branch_name)
            if profile:
                context_parts.append(
                    f"BRANCH DATA ({branch_name}):\n"
                    f"- Career paths: {', '.join(profile.get('career_paths',[])[:4])}\n"
                    f"- Median LPA: ₹{profile.get('median_lpa')} LPA\n"
                    f"- Coding intensity: {profile.get('coding_intensity')}/5\n"
                    f"- Research scope: {profile.get('research_scope')}/5\n"
                    f"- Suits goals: {', '.join(profile.get('suits_goals',[]))}\n"
                    f"- MBA transition: {career.get('mba_transition','N/A')}\n"
                    f"- Startup friendliness: {career.get('startup_friendliness','N/A')}"
                )
                sources.append("branch_profiles.json")
            break

    # Institute context
    institute_keywords = [
        "iit bombay", "iit delhi", "iit madras", "iit kanpur", "iit kharagpur",
        "iit roorkee", "iit hyderabad", "iit guwahati", "nit trichy", "nit warangal",
        "nit surathkal", "iiit hyderabad",
    ]
    for inst_kw in institute_keywords:
        if inst_kw in q_lower:
            inst_name = inst_kw.upper().replace("IIT ", "IIT ").title()
            # Normalize
            name_map = {
                "Iit Bombay": "IIT Bombay", "Iit Delhi": "IIT Delhi",
                "Iit Madras": "IIT Madras", "Iit Kanpur": "IIT Kanpur",
                "Iit Kharagpur": "IIT Kharagpur", "Iit Roorkee": "IIT Roorkee",
                "Iit Hyderabad": "IIT Hyderabad", "Iit Guwahati": "IIT Guwahati",
                "Nit Trichy": "NIT Trichy", "Nit Warangal": "NIT Warangal",
                "Nit Surathkal": "NIT Surathkal", "Iiit Hyderabad": "IIIT Hyderabad",
            }
            inst_name = name_map.get(inst_name, inst_name)
            meta = kl.get_institute(inst_name)
            plac = kl.get_institute_placement(inst_name)
            if meta:
                context_parts.append(
                    f"INSTITUTE DATA ({inst_name}):\n"
                    f"- Known for: {meta.get('known_for')}\n"
                    f"- NIRF rank: {meta.get('nirf_rank')}\n"
                    f"- Research score: {meta.get('research_score')}/5\n"
                    f"- Placement median: ₹{plac.get('median_lpa')} LPA\n"
                    f"- Top recruiters: {', '.join(plac.get('top_recruiters',[])[:4])}\n"
                    f"- Notable: {plac.get('notable')}"
                )
                sources.append("institute_tiers.json + placements.json")
            break

    # Comparison context
    if "vs" in q_lower or "better" in q_lower or "compare" in q_lower:
        comparisons = kl.get_all_comparisons()
        for key, comp in comparisons.items():
            branches = [b.lower() for b in comp.get("branches", [])]
            if any(b in q_lower for b in branches):
                context_parts.append(
                    f"BRANCH COMPARISON:\n"
                    f"Verdict: {comp.get('verdict')}\n"
                    f"Key insight: {comp.get('key_insight')}"
                )
                sources.append("branch_comparison.json")
                break

    # Startup / MBA / GATE context
    if "startup" in q_lower:
        advice = kl.startup_general_advice()
        context_parts.append(f"STARTUP CONTEXT:\n{advice}")
        sources.append("startup_ecosystem.json")

    if "abroad" in q_lower or "ms" in q_lower or "phd" in q_lower or "gre" in q_lower:
        general = kl._cache.get("higher_studies", {}).get("general_advice", {})
        context_parts.append(f"HIGHER STUDIES GENERAL ADVICE:\n{general}")
        sources.append("higher_studies.json")

    if "gate" in q_lower or "psu" in q_lower or "government" in q_lower:
        faq = kl.get_faq("gate_after_btech")
        if faq:
            context_parts.append(f"GATE/PSU INFO:\n{faq['answer']}")
            sources.append("faq.json (gate_after_btech)")

    context = "\n\n".join(context_parts) if context_parts else "No specific knowledge retrieved."
    return context, list(set(sources))


@router.post("/counselor/chat", response_model=CounselorChatResponse)
def counselor_chat(
    request:  CounselorChatRequest,
    kl=Depends(knowledge_loader),
    pipeline=Depends(rag_pipeline),
):
    """
    AI counselor follow-up Q&A.

    Examples:
      "Why not IIT Kanpur?"
      "Is MnC better than CSE for me?"
      "Will MBA be easier after Mechanical?"
      "What if I want to go abroad for PhD?"
      "Should I choose IIT Madras EE over IIT Guwahati CSE?"
    """
    settings = get_settings()
    student  = request.student
    question = request.question.strip()

    # Check FAQ first (fast path — no LLM needed)
    faq_results = kl.search_faq(question)
    faq_match   = faq_results[0] if faq_results else None

    # Retrieve grounded context
    context_str, sources = _get_grounded_context(question, kl)

    # Try LLM via the shared provider abstraction (Gemini / Claude / None)
    answer = None
    if settings.enable_llm_explanations:
        system   = _build_counselor_system_prompt(student, kl)
        
        # Format history
        history_text = ""
        if request.history:
            history_text = "PREVIOUS CONVERSATION HISTORY:\n"
            # Send last 4 turns to avoid context bloat
            for msg in request.history[-4:]:
                role = "User" if msg.get("role") == "user" else "AI"
                history_text += f"{role}: {msg.get('text')}\n"
            history_text += "\n"

        user_msg = f"""{system}

{history_text}KNOWLEDGE BASE CONTEXT:
{context_str}

STUDENT'S QUESTION: {question}

Answer in 3-5 sentences. Be specific to this student's rank ({student.effective_rank}),
goals ({', '.join(student.active_goals) or 'general'}), and home state ({student.home_state}).
Only use facts from the context above."""

        llm = pipeline._llm_provider  # reuse the configured provider (Gemini/Claude/None)
        try:
            answer = llm.generate(user_msg)
            if answer:
                provider_name = type(llm).__name__.replace("Provider", "").lower()
                sources.append(provider_name)
        except Exception as e:
            log.warning(f"Counselor LLM failed: {e}")
            answer = None

    if not answer:
        answer = _template_answer(question, faq_match, student, kl)

    return CounselorChatResponse(
        question=question,
        answer=answer,
        sources=sources,
        faq_match=faq_match,
    )


def _template_answer(
    question: str,
    faq_match: dict | None,
    student: StudentProfile,
    kl: KnowledgeLoader,
) -> str:
    """Fallback answer when LLM is unavailable."""
    if faq_match:
        return (
            f"{faq_match['answer']} "
            f"(Note: this is a general answer — enable AI for personalised advice.)"
        )

    q_lower = question.lower()
    if "vs" in q_lower or "better" in q_lower:
        comps = kl.get_all_comparisons()
        for _, comp in comps.items():
            branches = [b.lower() for b in comp.get("branches", [])]
            if any(b in q_lower for b in branches):
                return comp.get("verdict", "No comparison data available.")

    return (
        f"I don't have a specific answer for '{question}' without AI enabled. "
        f"Based on your rank {student.effective_rank} and goals "
        f"({', '.join(student.active_goals) or 'general'}), "
        f"I'd recommend enabling AI explanations in settings for personalised advice."
    )
