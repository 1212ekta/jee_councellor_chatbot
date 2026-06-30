"""
Career Persona Engine — Phase 2

Infers a Career Persona from a student's interest vector and goals.
The persona shapes the recommendation narrative and counselor tone.

Personas:
  1. Future Software Engineer    — loves coding, wants high salary, industry-focused
  2. AI / ML Researcher          — coding + AI + research, wants PhD or top AI lab
  3. Entrepreneur / Startup      — coding + startup goals, risk-tolerant
  4. MBA Aspirant                — any branch + MBA goal, placement & brand focused
  5. Core Engineer               — mechanical/civil/chemical, hands-on, industry
  6. Government / PSU Aspirant   — stability, prefers NIT/IIT for GATE/PSU route
  7. Higher Studies Abroad       — research + top institute brand for GRE/MS/PhD
  8. Electronics & VLSI          — ECE/EE focused, chip design, embedded systems
  9. Undecided Explorer          — no strong signal, needs broad options

Each persona has:
  - A scoring function (returns 0.0–1.0)
  - A label and short description
  - Recommended branch domains
  - Institute preferences
  - A counselor voice template
"""

from dataclasses import dataclass, field

from app.models.request import StudentProfile
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CareerPersona:
    """Inferred career persona with full context for personalisation."""
    id:           str          # machine key e.g. 'ai_researcher'
    label:        str          # display name e.g. 'AI / ML Researcher'
    confidence:   float        # 0.0–1.0 how strongly this persona fits
    description:  str          # 2-sentence description of this persona
    icon:         str          # emoji for UI
    secondary:    str | None   # secondary persona id if close match

    # Recommendation guidance
    preferred_domains:    list[str]   = field(default_factory=list)
    preferred_inst_types: list[str]   = field(default_factory=list)
    weight_overrides:     dict        = field(default_factory=dict)

    # Counselor narrative fragments
    counselor_opener:     str         = ""
    branch_advice:        str         = ""
    institute_advice:     str         = ""
    career_horizon:       str         = ""


# ── Persona definitions ───────────────────────────────────────────────────────

PERSONA_DEFINITIONS = {
    "software_engineer": {
        "label":       "Future Software Engineer",
        "icon":        "💻",
        "description": (
            "You're drawn to building software, solving algorithmic problems, "
            "and working in the tech industry. High placement packages and a "
            "strong coding culture matter most to you."
        ),
        "preferred_domains":    ["CS", "MnC"],
        "preferred_inst_types": ["IIT", "NIT", "IIIT"],
        "weight_overrides":     {"interest_match": 0.30, "institute_strength": 0.20},
        "counselor_opener": (
            "As someone with a strong software engineering orientation, "
            "your priority should be securing the best CS/MnC seat your rank allows."
        ),
        "branch_advice": (
            "CSE and Mathematics & Computing are your top targets. "
            "At any IIT, even a 'lower' branch like EE gives you access to "
            "coding clubs, internship pipelines, and branch change opportunities."
        ),
        "institute_advice": (
            "For SWE roles, IIT brand > NIT brand >> IIIT for top-tier FAANG offers. "
            "But NIT Trichy/Warangal/Surathkal CS outperforms many new IITs in placements."
        ),
        "career_horizon": (
            "Year 1–2: DSA + web fundamentals. "
            "Year 3: Intern at product company. "
            "Year 4: Placement at top tech firm (₹20–60 LPA range). "
            "5 years out: Senior SWE or startup CTO."
        ),
    },

    "ai_researcher": {
        "label":       "AI / ML Researcher",
        "icon":        "🧠",
        "description": (
            "You're interested in the science behind AI, not just applying it. "
            "You want to do research, publish papers, and possibly pursue a PhD "
            "at a top global university or work at an AI lab."
        ),
        "preferred_domains":    ["CS", "MnC", "EE", "EP"],
        "preferred_inst_types": ["IIT", "IIIT"],
        "weight_overrides":     {
            "interest_match": 0.30,
            "career_alignment": 0.18,
            "institute_strength": 0.17,
        },
        "counselor_opener": (
            "Your profile points strongly toward AI/ML research. "
            "The institute you choose will define your PhD application profile more "
            "than the specific branch — prioritise research culture over branch name."
        ),
        "branch_advice": (
            "MnC (Mathematics & Computing) is the hidden gem for AI researchers — "
            "stronger theoretical foundations than CSE. "
            "IIT Hyderabad's AI BTech and IIIT Hyderabad's CSE are exceptional "
            "for ML research exposure early in your degree."
        ),
        "institute_advice": (
            "IIT Bombay, Madras, Delhi, and Kanpur have the strongest ML research groups. "
            "IIIT Hyderabad punches above its weight for AI research despite not being an IIT."
        ),
        "career_horizon": (
            "Year 1–2: Strong math foundations (linear algebra, probability, calculus). "
            "Year 3: Research internship at IISc / abroad. "
            "Year 4: Strong thesis → PhD applications to top-10 global programs. "
            "Post-PhD: Research scientist at DeepMind / OpenAI / top lab."
        ),
    },

    "entrepreneur": {
        "label":       "Entrepreneur / Startup Founder",
        "icon":        "🚀",
        "description": (
            "You want to build your own company. You value a strong alumni "
            "network, entrepreneurship cells, and coding culture over "
            "traditional placement packages."
        ),
        "preferred_domains":    ["CS", "MnC", "EE"],
        "preferred_inst_types": ["IIT", "NIT"],
        "weight_overrides":     {
            "interest_match":    0.28,
            "institute_strength":0.20,
            "flexibility":       0.08,
        },
        "counselor_opener": (
            "With startup ambitions, your college will be your first investor — "
            "choose one with a strong E-cell, alumni angel network, "
            "and a culture that tolerates unconventional paths."
        ),
        "branch_advice": (
            "CSE is the clearest path for a tech startup founder. "
            "But don't underestimate MnC — the quantitative depth opens "
            "fintech and AI startup doors. Avoid niche branches that lock "
            "you into a single industry."
        ),
        "institute_advice": (
            "IIT Bombay has the best startup alumni network (Zepto, Razorpay, Meesho). "
            "IIT Delhi for fintech and consulting adjacent startups. "
            "The E-cell at IIT Madras and IIT Kharagpur are also strong."
        ),
        "career_horizon": (
            "Year 1–2: Build side projects, join E-cell, find co-founders. "
            "Year 3: Launch first product during internship break. "
            "Year 4: Either take placement for 2 years of runway, or go direct. "
            "5 years out: Series A or bust — either way, valuable experience."
        ),
    },

    "mba_aspirant": {
        "label":       "MBA Aspirant",
        "icon":        "📊",
        "description": (
            "You plan to do an MBA from IIM or a top business school after "
            "your engineering degree. You value brand name, placement quality, "
            "and a broad network over technical depth."
        ),
        "preferred_domains":    ["CS", "MECH", "EE", "ECE"],
        "preferred_inst_types": ["IIT", "NIT"],
        "weight_overrides":     {
            "institute_strength":0.25,
            "career_alignment":  0.18,
            "flexibility":       0.07,
        },
        "counselor_opener": (
            "For an MBA path, the IIT brand on your resume is your biggest "
            "asset for CAT interviews. Institute matters more than branch here — "
            "an IIT Kharagpur Mechanical will outperform a lesser-known CSE for IIM calls."
        ),
        "branch_advice": (
            "Any branch at a top IIT works. Mechanical and EE are historically "
            "strong for IIM admits. CSE gives you tech-to-MBA optionality. "
            "Avoid branches that feel like dead-ends (mining, textile) "
            "unless the institute is truly exceptional."
        ),
        "institute_advice": (
            "IITs produce the most IIM toppers. Among NITs, NIT Trichy and Warangal "
            "have the strongest MBA admit records. Brand trumps branch for this path."
        ),
        "career_horizon": (
            "Year 1–4: Engineering degree + internships at consulting/finance firms. "
            "Year 5: CAT prep (target 99+ percentile). "
            "Year 6–7: IIM A/B/C MBA. "
            "Year 7+: Management consulting, investment banking, or corporate strategy."
        ),
    },

    "core_engineer": {
        "label":       "Core Engineer",
        "icon":        "⚙️",
        "description": (
            "You genuinely enjoy mechanical, civil, chemical, or manufacturing "
            "engineering. You want to work in the physical world — "
            "design, manufacturing, infrastructure, or R&D at large firms."
        ),
        "preferred_domains":    ["MECH", "CIVIL", "CHEM", "ECE"],
        "preferred_inst_types": ["IIT", "NIT"],
        "weight_overrides":     {
            "interest_match":    0.30,
            "career_alignment":  0.18,
            "institute_strength":0.12,
        },
        "counselor_opener": (
            "As someone drawn to core engineering, you're in the minority — "
            "and that's actually an advantage. Core roles at L&T, ISRO, DRDO, "
            "and Tata are competitive and deeply rewarding for people who love the work."
        ),
        "branch_advice": (
            "Go for the branch that genuinely excites you, not the highest-ranked one. "
            "Mechanical at IIT Madras, Civil at IIT Roorkee, Chemical at IIT Bombay "
            "— these are world-class in their domains. "
            "GATE after BTech keeps PSU options open."
        ),
        "institute_advice": (
            "For core engineering, old IITs have the strongest labs and industry ties. "
            "IIT Roorkee for civil, IIT Madras for mechanical/ocean, "
            "IIT Bombay for chemical. NITs are excellent for core roles too."
        ),
        "career_horizon": (
            "Year 1–4: Deep technical foundation + internship at core industry firm. "
            "Year 4: Placements at Tata, L&T, BHEL, or GATE for M.Tech/PSU. "
            "5 years out: Senior engineer at large firm or government R&D."
        ),
    },

    "govt_psu_aspirant": {
        "label":       "Government / PSU Aspirant",
        "icon":        "🏛️",
        "description": (
            "You value job security, work-life balance, and public service. "
            "You plan to appear for UPSC, GATE (for PSUs), or state services. "
            "A stable career with social impact appeals to you."
        ),
        "preferred_domains":    ["CIVIL", "MECH", "EE", "CHEM"],
        "preferred_inst_types": ["IIT", "NIT"],
        "weight_overrides":     {
            "career_alignment":  0.20,
            "institute_strength":0.12,
            "flexibility":       0.03,
        },
        "counselor_opener": (
            "For a government/PSU career, your engineering branch determines "
            "your GATE paper and PSU eligibility. Choose strategically — "
            "Civil for IES/UPSC, EE/Mech/Instrumentation for PSU GATE."
        ),
        "branch_advice": (
            "Civil Engineering is the strongest branch for IES/ESE and UPSC. "
            "EE and Mechanical open doors to NTPC, BHEL, PGCIL, IOCL via GATE. "
            "Computer Science GATE is extremely competitive for PSUs. "
            "Core branches give you the best PSU diversity."
        ),
        "institute_advice": (
            "For GATE, what matters is the quality of your preparation, not just the "
            "institute. But NIT/IIT exposure gives you peer group advantage for GATE. "
            "Location matters for UPSC — NITs in your home state simplify logistics."
        ),
        "career_horizon": (
            "Year 1–4: Engineering degree + GATE preparation in parallel. "
            "Year 4: GATE exam, PSU interviews. "
            "Year 5: Join PSU (NTPC/BHEL/ONGC) or attempt UPSC/IES. "
            "10 years out: Senior government officer or PSU executive."
        ),
    },

    "higher_studies_abroad": {
        "label":       "Higher Studies Abroad",
        "icon":        "🌍",
        "description": (
            "You plan to pursue an MS or PhD at a top global university after "
            "your BTech. Research publications, GPA, GRE, and professor "
            "connections matter more than domestic placements."
        ),
        "preferred_domains":    ["CS", "MnC", "EE", "ECE", "EP", "CHEM"],
        "preferred_inst_types": ["IIT", "IIIT"],
        "weight_overrides":     {
            "institute_strength":0.22,
            "career_alignment":  0.17,
            "interest_match":    0.27,
        },
        "counselor_opener": (
            "For MS/PhD abroad, your institute's global reputation and research "
            "output matters enormously. IIT professors have strong connections "
            "to foreign universities — their recommendation letters open doors."
        ),
        "branch_advice": (
            "Choose the branch you genuinely want to research for 5+ years. "
            "A strong GPA in a less-glamorous branch at IIT Bombay beats "
            "a weak GPA in CSE at a lesser institute. "
            "Research projects and publications matter more than branch name for PhD admits."
        ),
        "institute_advice": (
            "Old IITs have the strongest research infrastructure and professor networks. "
            "IIT Bombay, Delhi, Madras, Kanpur have the most US/EU PhD placements. "
            "IIIT Hyderabad is exceptional for CS PhD applications specifically."
        ),
        "career_horizon": (
            "Year 1–4: Strong GPA + research project + ideally 1 publication. "
            "Year 4: GRE + SOP + applications to MIT/Stanford/CMU/ETH/etc. "
            "Year 5–10: MS/PhD abroad. "
            "Post-PhD: Research scientist, professor, or industry R&D."
        ),
    },

    "electronics_vlsi": {
        "label":       "Electronics & VLSI Engineer",
        "icon":        "🔌",
        "description": (
            "You're fascinated by chips, circuits, embedded systems, and hardware. "
            "You want to work in semiconductor design, VLSI, embedded systems, "
            "or RF engineering — the physical layer of technology."
        ),
        "preferred_domains":    ["ECE", "EE"],
        "preferred_inst_types": ["IIT", "NIT", "IIIT"],
        "weight_overrides":     {
            "interest_match":    0.32,
            "career_alignment":  0.18,
        },
        "counselor_opener": (
            "Electronics and VLSI is experiencing a golden age — India's semiconductor "
            "mission means chip design roles are booming. Your interest in hardware "
            "puts you at the intersection of two of the hottest domains."
        ),
        "branch_advice": (
            "ECE with VLSI specialisation is your primary target. "
            "EE with IC Design is equally strong. "
            "Look specifically for institutes with Microelectronics/VLSI labs — "
            "IIT Madras (IITM-PRAVARTAK), IIT Bombay, IIT Kharagpur have world-class facilities."
        ),
        "institute_advice": (
            "IIT Madras is India's hub for chip design. "
            "IIT Bombay and IIT Kharagpur have strong VLSI research. "
            "NIT Trichy ECE is exceptional for core electronics placements. "
            "Qualcomm, Texas Instruments, Intel recruit heavily from these institutes."
        ),
        "career_horizon": (
            "Year 1–2: Circuit theory, digital electronics, HDL (Verilog/VHDL). "
            "Year 3: VLSI design course + internship at TI/Qualcomm/Intel. "
            "Year 4: Core placement at semiconductor firm (₹12–25 LPA). "
            "5 years out: Chip design lead or MS in ECE abroad."
        ),
    },

    "undecided_explorer": {
        "label":       "Undecided Explorer",
        "icon":        "🧭",
        "description": (
            "You haven't found your passion yet — and that's completely okay. "
            "You need a branch and institute that keeps maximum options open "
            "while you discover what excites you."
        ),
        "preferred_domains":    ["CS", "EE", "ECE", "MnC"],
        "preferred_inst_types": ["IIT", "NIT"],
        "weight_overrides":     {
            "institute_strength":0.25,
            "flexibility":       0.10,
            "interest_match":    0.15,
        },
        "counselor_opener": (
            "Not knowing what you want is actually a sign of intellectual honesty. "
            "Your strategy should be: maximise optionality. "
            "Choose a branch and institute that lets you pivot easily."
        ),
        "branch_advice": (
            "CSE is the highest-optionality branch — you can pivot to finance, "
            "product, research, or startups from there. "
            "EE is a strong second — 'soft' enough to code, technical enough for core. "
            "Avoid committing to niche branches until you're certain."
        ),
        "institute_advice": (
            "Prioritise institute quality over branch when undecided. "
            "A better institute gives you a stronger peer group, better internship "
            "access, and more time to figure out your direction."
        ),
        "career_horizon": (
            "Year 1: Explore — attend every tech fest, club, and seminar you can. "
            "Year 2: Narrow down to 2–3 areas of genuine interest. "
            "Year 3: Internship in your chosen direction. "
            "Year 4: Placement or higher studies based on what you discovered."
        ),
    },
}


# ── Scoring functions ─────────────────────────────────────────────────────────

def _score_software_engineer(s: StudentProfile) -> float:
    return (
        0.40 * s.interest_coding
        + 0.20 * s.interest_ai_ml
        + 0.20 * s.salary_priority
        + 0.10 * (1.0 - s.interest_research)
        + 0.10 * (0.0 if s.wants_govt_job else 0.5)
    )


def _score_ai_researcher(s: StudentProfile) -> float:
    return (
        0.30 * s.interest_ai_ml
        + 0.25 * s.interest_research
        + 0.25 * s.interest_coding
        + 0.10 * (1.0 if s.wants_research else 0.0)
        + 0.10 * (1.0 if s.wants_higher_studies_abroad else 0.0)
    )


def _score_entrepreneur(s: StudentProfile) -> float:
    return (
        0.35 * (1.0 if s.wants_startup else 0.0)
        + 0.25 * s.interest_coding
        + 0.20 * s.interest_ai_ml
        + 0.10 * s.salary_priority
        + 0.10 * (1.0 - s.brand_priority)
    )


def _score_mba_aspirant(s: StudentProfile) -> float:
    return (
        0.50 * (1.0 if s.wants_mba else 0.0)
        + 0.20 * s.brand_priority
        + 0.20 * s.salary_priority
        + 0.10 * (1.0 - s.interest_research)
    )


def _score_core_engineer(s: StudentProfile) -> float:
    return (
        0.30 * s.interest_core_engineering
        + 0.20 * s.interest_mechanical
        + 0.15 * s.interest_civil
        + 0.15 * s.interest_chemical
        + 0.10 * (1.0 - s.interest_coding)
        + 0.10 * (1.0 if s.wants_govt_job else 0.0)
    )


def _score_govt_psu(s: StudentProfile) -> float:
    return (
        0.45 * (1.0 if s.wants_govt_job else 0.0)
        + 0.20 * (1.0 - s.salary_priority)
        + 0.15 * s.interest_core_engineering
        + 0.10 * (1.0 - s.wants_startup)
        + 0.10 * s.interest_mechanical
    )


def _score_higher_studies(s: StudentProfile) -> float:
    return (
        0.45 * (1.0 if s.wants_higher_studies_abroad else 0.0)
        + 0.25 * s.interest_research
        + 0.15 * s.brand_priority
        + 0.15 * (1.0 if s.wants_research else 0.0)
    )


def _score_electronics_vlsi(s: StudentProfile) -> float:
    return (
        0.45 * s.interest_electronics
        + 0.25 * s.interest_core_engineering
        + 0.15 * s.interest_coding
        + 0.10 * (1.0 - s.interest_mechanical)
        + 0.05 * (1.0 - s.interest_civil)
    )


def _score_undecided(s: StudentProfile) -> float:
    """High score when no other signal is strong."""
    all_interests = s.interest_vector
    variance = sum((x - 0.5) ** 2 for x in all_interests) / len(all_interests)
    # High variance = strong opinions = not undecided
    # Low variance (all near 0.5) = undecided
    return max(0.0, 0.8 - variance * 4)


SCORERS = {
    "software_engineer":    _score_software_engineer,
    "ai_researcher":        _score_ai_researcher,
    "entrepreneur":         _score_entrepreneur,
    "mba_aspirant":         _score_mba_aspirant,
    "core_engineer":        _score_core_engineer,
    "govt_psu_aspirant":    _score_govt_psu,
    "higher_studies_abroad":_score_higher_studies,
    "electronics_vlsi":     _score_electronics_vlsi,
    "undecided_explorer":   _score_undecided,
}


# ── Main inference function ───────────────────────────────────────────────────

def infer_persona(student: StudentProfile) -> CareerPersona:
    """
    Infer the best-fitting Career Persona for a student.

    Returns a CareerPersona with:
      - Primary persona (highest score)
      - Secondary persona (if second score is within 0.15 of first)
      - Confidence score
    """
    scores = {
        persona_id: scorer(student)
        for persona_id, scorer in SCORERS.items()
    }

    sorted_personas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_id, top_score = sorted_personas[0]
    second_id, second_score = sorted_personas[1]

    # Normalise confidence: top score relative to theoretical max (1.0)
    confidence = min(1.0, top_score)

    # Secondary persona if within 15% of primary
    secondary = second_id if (top_score - second_score) < 0.15 else None

    defn = PERSONA_DEFINITIONS[top_id]

    log.info(
        f"Persona inferred: {defn['label']} "
        f"(confidence={confidence:.2f}, secondary={secondary})"
    )

    return CareerPersona(
        id=top_id,
        label=defn["label"],
        confidence=round(confidence, 3),
        description=defn["description"],
        icon=defn["icon"],
        secondary=secondary,
        preferred_domains=defn["preferred_domains"],
        preferred_inst_types=defn["preferred_inst_types"],
        weight_overrides=defn.get("weight_overrides", {}),
        counselor_opener=defn["counselor_opener"],
        branch_advice=defn["branch_advice"],
        institute_advice=defn["institute_advice"],
        career_horizon=defn["career_horizon"],
    )


def get_all_persona_scores(student: StudentProfile) -> list[dict]:
    """Return all persona scores sorted — useful for debugging and UI radar chart."""
    scores = {
        persona_id: round(scorer(student), 3)
        for persona_id, scorer in SCORERS.items()
    }
    return sorted(
        [
            {
                "id": pid,
                "label": PERSONA_DEFINITIONS[pid]["label"],
                "icon": PERSONA_DEFINITIONS[pid]["icon"],
                "score": score,
            }
            for pid, score in scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
