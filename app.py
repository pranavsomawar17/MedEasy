"""
Health Article Simplifier & Medical Q&A (Streamlit + LangChain + Groq + FAISS)
"""

from __future__ import annotations

import csv
import io
from typing import Dict

import streamlit as st

from extractor import extract_from_pdf, extract_from_url
from rag_qa import answer_question, build_faiss_index, load_embeddings
from simplifier import simplify_article


st.set_page_config(
    page_title="Health Article Simplifier",
    page_icon="🩺",
    layout="wide",
)

GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Recommended)",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant (Fastest)",
    "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B (High Reasoning)",
    "__custom__": "Custom Groq Model ID...",
}

LANGUAGES = [
    "Simple English",
    "Marathi (मराठी)",
    "Hindi (हिन्दी)",
    "Gujarati (ગુજરાતી)",
    "Tamil (தமிழ்)",
    "Telugu (తెలుగు)",
    "Bengali (বাংলা)",
    "Kannada (ಕನ್ನಡ)",
    "Malayalam (മലയാളം)",
    "Punjabi (ਪੰਜਾਬੀ)",
    "Urdu (اردو)",
    "Spanish (Español)",
]

READING_LEVELS = [
    "Everyday Reader (Grade 6–8)",
    "Very Simple (Grade 4–5)",
    "Patient & Caregiver Summary",
]

SAMPLE_ARTICLES: Dict[str, Dict[str, str]] = {
    "Hypertension & ACE Inhibitors": {
        "title": "Essential Hypertension & ACE Inhibitor Therapy",
        "text": (
            "Essential hypertension is a chronic cardiovascular condition characterized by persistently "
            "elevated systemic arterial blood pressure (≥140/90 mmHg in clinic readings). Prolonged "
            "vascular resistance increases left ventricular afterload, accelerating myocardial "
            "hypertrophy, atherosclerosis, and target-organ damage including nephropathy and retinopathy. "
            "First-line pharmacotherapy commonly includes angiotensin-converting enzyme (ACE) inhibitors, "
            "calcium channel blockers, or thiazide-like diuretics, alongside non-pharmacological "
            "interventions such as restricting dietary sodium intake to less than 2,000 mg per day and "
            "performing at least 150 minutes of moderate-intensity aerobic exercise weekly. Patients "
            "initiated on ACE inhibitors should have serum creatinine and potassium levels monitored "
            "within 2 to 4 weeks to detect acute kidney injury or hyperkalemia, and ACE inhibitors are "
            "strictly contraindicated during pregnancy due to fetal toxicity."
        ),
    },
    "Type 2 Diabetes & HbA1c": {
        "title": "Type 2 Diabetes Mellitus Management Overview",
        "text": (
            "Type 2 diabetes mellitus (T2DM) is a progressive metabolic disorder defined by peripheral "
            "insulin resistance and relative pancreatic beta-cell dysfunction, resulting in chronic "
            "hyperglycemia. Diagnosis is typically confirmed by a glycated hemoglobin (HbA1c) level of "
            "6.5% or higher, or a fasting plasma glucose of 126 mg/dL or above on two separate occasions. "
            "Uncontrolled glycemic variability elevates the risk of microvascular complications—such as "
            "peripheral neuropathy, diabetic retinopathy, and albuminuria—as well as macrovascular events. "
            "Metformin remains the preferred initial oral hypoglycemic agent unless contraindicated by an "
            "estimated glomerular filtration rate (eGFR) below 30 mL/min/1.73 m². Patients should monitor "
            "for symptoms of hypoglycemia (blood glucose <70 mg/dL), particularly when sulfonylureas or "
            "exogenous insulin are added to the regimen."
        ),
    },
}


@st.cache_resource(show_spinner="Loading local embedding model (all-MiniLM-L6-v2)...")
def get_cached_embeddings():
    return load_embeddings()


def _init_state():
    defaults = {
        "article_title": "",
        "article_text": "",
        "article_source": "",
        "result": None,
        "result_language": "",
        "was_truncated": False,
        "vectorstore": None,
        "vectorstore_key": "",
        "chat_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _set_loaded_article(title: str, text: str, source: str):
    st.session_state.article_title = title
    st.session_state.article_text = text
    st.session_state.article_source = source
    st.session_state.result = None
    st.session_state.vectorstore = None
    st.session_state.vectorstore_key = ""
    st.session_state.chat_history = []


def _format_txt_export(result: dict, language: str) -> str:
    lines = [
        result.get("simplified_title", "Simplified Health Article"),
        f"Language: {language}",
        "",
    ]
    for para in result.get("simplified_article", []):
        lines.append(para)
        lines.append("")

    facts = result.get("key_facts_preserved", [])
    if facts:
        lines.append("KEY CLINICAL FACTS PRESERVED:")
        for fact in facts:
            lines.append(f"- {fact}")
        lines.append("")

    glossary = result.get("glossary", [])
    if glossary:
        lines.append("MEDICAL TERMS GLOSSARY:")
        for item in glossary:
            lines.append(f"- {item['term']}: {item['simple_meaning']}")
            if item.get("context"):
                lines.append(f"  Context: {item['context']}")
    return "\n".join(lines).strip()


def _format_glossary_csv(glossary: list) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Medical Term", "Simple Meaning", "Context in Article"])
    for item in glossary:
        writer.writerow([item.get("term", ""), item.get("simple_meaning", ""), item.get("context", "")])
    return buf.getvalue()


_init_state()

# --- Sidebar: API Key & Settings ---
with st.sidebar:
    st.header("⚙️ Groq API & Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free API key from console.groq.com/keys. Your key stays in this session only.",
    )
    if groq_api_key.strip():
        st.success("API key entered", icon="✅")
    else:
        st.info("Paste your free Groq API key above to run the model.")

    model_choice = st.selectbox(
        "Groq Model",
        options=list(GROQ_MODELS.keys()),
        format_func=lambda k: GROQ_MODELS[k],
        index=0,
    )
    if model_choice == "__custom__":
        selected_model = st.text_input("Custom Groq Model ID", value="llama-3.3-70b-versatile").strip()
    else:
        selected_model = model_choice

    st.divider()
    target_language = st.selectbox("Output Language", options=LANGUAGES, index=0)
    reading_level = st.selectbox("Simplification Level", options=READING_LEVELS, index=0)

    st.divider()
    st.caption(
        "**Stack:** Python · Streamlit · LangChain · Groq API · FAISS + local HuggingFace embeddings.\n\n"
        "**Disclaimer:** This tool simplifies health articles for readability only and is not medical advice."
    )

# --- Header ---
st.title("🩺 Health Article Simplifier")
st.caption(
    "Fetch any health article from a URL, paste text, or upload a PDF → rewrite in simple English or "
    "a regional language without changing clinical facts → explore the Medical Glossary & ask Q&A via FAISS."
)

# --- Step 1: Load Article ---
input_tab_url, input_tab_paste, input_tab_file = st.tabs(
    ["🔗 From Article Link (URL)", "📝 Paste Article Text", "📄 Upload PDF / TXT"]
)

with input_tab_url:
    url_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
    with url_col:
        article_url = st.text_input(
            "Health Article URL",
            placeholder="https://www.who.int/news-room/fact-sheets/detail/hypertension",
        )
    with btn_col:
        fetch_clicked = st.button("Fetch Article", type="secondary", use_container_width=True)

    if fetch_clicked:
        if not article_url.strip():
            st.warning("Please enter an article URL first.")
        else:
            with st.spinner("Fetching and cleaning article content..."):
                try:
                    title, body = extract_from_url(article_url)
                    _set_loaded_article(title, body, article_url.strip())
                    st.success(f"Loaded **{title}** ({len(body.split()):,} words).")
                except Exception as exc:
                    st.error(str(exc))

with input_tab_paste:
    sample_col1, sample_col2, _ = st.columns([1.4, 1.4, 2])
    with sample_col1:
        if st.button("Load Sample: Hypertension", use_container_width=True):
            s = SAMPLE_ARTICLES["Hypertension & ACE Inhibitors"]
            _set_loaded_article(s["title"], s["text"], "Built-in Sample")
    with sample_col2:
        if st.button("Load Sample: Type 2 Diabetes", use_container_width=True):
            s = SAMPLE_ARTICLES["Type 2 Diabetes & HbA1c"]
            _set_loaded_article(s["title"], s["text"], "Built-in Sample")

    pasted_text = st.text_area(
        "Paste health article text",
        value=st.session_state.article_text if st.session_state.article_source in {"Pasted Text", "Built-in Sample"} else "",
        height=190,
        placeholder="Paste a medical or health article here...",
    )
    if st.button("Use Pasted Text", type="secondary"):
        if len(pasted_text.strip()) < 30:
            st.warning("Please paste at least a few sentences.")
        else:
            _set_loaded_article("Pasted Health Article", pasted_text.strip(), "Pasted Text")
            st.success(f"Loaded pasted text ({len(pasted_text.split()):,} words).")

with input_tab_file:
    uploaded = st.file_uploader("Upload a health article (.pdf or .txt)", type=["pdf", "txt"])
    if uploaded is not None and st.button("Load Uploaded File", type="secondary"):
        try:
            raw_bytes = uploaded.read()
            if uploaded.name.lower().endswith(".pdf"):
                title, body = extract_from_pdf(raw_bytes, uploaded.name)
            else:
                title = uploaded.name.rsplit(".", 1)[0]
                body = raw_bytes.decode("utf-8", errors="ignore").strip()
            _set_loaded_article(title, body, f"File: {uploaded.name}")
            st.success(f"Loaded **{title}** ({len(body.split()):,} words).")
        except Exception as exc:
            st.error(str(exc))

# --- Loaded Article Status & Simplify CTA ---
if st.session_state.article_text:
    word_count = len(st.session_state.article_text.split())
    with st.expander(
        f"Loaded Article: {st.session_state.article_title} ({word_count:,} words) — click to preview",
        expanded=False,
    ):
        st.caption(f"Source: {st.session_state.article_source}")
        st.write(st.session_state.article_text[:3000] + ("..." if len(st.session_state.article_text) > 3000 else ""))

    if st.button("✨ Simplify Article & Build Glossary", type="primary", use_container_width=True):
        if not groq_api_key.strip():
            st.error("Please paste your Groq API Key in the left sidebar first.")
        elif not selected_model:
            st.error("Please select or enter a Groq model ID in the left sidebar.")
        else:
            with st.spinner(f"Simplifying into {target_language} with Groq ({selected_model})..."):
                try:
                    result, was_truncated = simplify_article(
                        article_text=st.session_state.article_text,
                        api_key=groq_api_key.strip(),
                        model_name=selected_model,
                        language=target_language,
                        reading_level=reading_level,
                    )
                    st.session_state.result = result
                    st.session_state.result_language = target_language
                    st.session_state.was_truncated = was_truncated
                except Exception as exc:
                    st.error(f"Groq API error: {exc}")

# --- Step 2: Results & FAISS Q&A ---
if st.session_state.article_text:
    st.divider()
    out_tab_simple, out_tab_glossary, out_tab_qa = st.tabs(
        ["📖 Simplified Article", "📚 Medical Glossary", "💬 Ask About This Article (FAISS RAG)"]
    )

    with out_tab_simple:
        res = st.session_state.result
        if not res:
            st.info("Click **Simplify Article & Build Glossary** above to generate the simplified rewrite.")
        else:
            if st.session_state.was_truncated:
                st.warning(
                    "This article was very long, so the first 16,000 characters were simplified to stay within Groq free-tier token limits. "
                    "Use the **Ask About This Article (FAISS RAG)** tab to query the entire article."
                )

            st.subheader(res["simplified_title"])
            st.caption(f"Language: **{st.session_state.result_language}**")

            for para in res["simplified_article"]:
                st.write(para)

            if res["key_facts_preserved"]:
                st.markdown("#### ✅ Key Clinical Facts Preserved")
                for fact in res["key_facts_preserved"]:
                    st.markdown(f"- {fact}")

            st.download_button(
                label="⬇️ Download Simplified Article (.txt)",
                data=_format_txt_export(res, st.session_state.result_language),
                file_name="simplified_health_article.txt",
                mime="text/plain",
            )

    with out_tab_glossary:
        res = st.session_state.result
        if not res:
            st.info("Click **Simplify Article & Build Glossary** above to extract medical terms.")
        else:
            glossary = res.get("glossary", [])
            if not glossary:
                st.info("No complex medical terms were extracted from this article.")
            else:
                filter_query = st.text_input("Filter medical terms", placeholder="Type a term or keyword...")
                q = filter_query.strip().lower()
                filtered = [
                    item
                    for item in glossary
                    if not q
                    or q in item["term"].lower()
                    or q in item["simple_meaning"].lower()
                    or q in item.get("context", "").lower()
                ]

                st.caption(f"Showing {len(filtered)} of {len(glossary)} medical terms")
                for item in filtered:
                    with st.container(border=True):
                        st.markdown(f"**{item['term']}**")
                        st.write(item["simple_meaning"])
                        if item.get("context"):
                            st.caption(f"In this article: {item['context']}")

                st.download_button(
                    label="⬇️ Download Glossary (.csv)",
                    data=_format_glossary_csv(glossary),
                    file_name="medical_glossary.csv",
                    mime="text/csv",
                )

    with out_tab_qa:
        st.caption(
            "Ask any follow-up question about the loaded article. "
            "LangChain + FAISS retrieves the most relevant excerpts and Groq answers in your chosen language."
        )

        # Build FAISS index lazily when user asks a question or clicks index button
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("View retrieved article excerpts"):
                        for idx, excerpt in enumerate(msg["sources"], start=1):
                            st.caption(f"Excerpt {idx}")
                            st.write(excerpt)

        user_question = st.chat_input("Ask a question about this health article (e.g., What are the warning signs or dosage limits?)")
        if user_question:
            if not groq_api_key.strip():
                st.error("Please paste your Groq API Key in the left sidebar first.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.write(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching article with FAISS and generating answer..."):
                        try:
                            # Build FAISS vectorstore once per loaded article
                            current_key = f"{st.session_state.article_title}:{len(st.session_state.article_text)}"
                            if st.session_state.vectorstore is None or st.session_state.vectorstore_key != current_key:
                                embeddings = get_cached_embeddings()
                                st.session_state.vectorstore = build_faiss_index(
                                    st.session_state.article_text,
                                    st.session_state.article_title,
                                    embeddings,
                                )
                                st.session_state.vectorstore_key = current_key

                            answer, docs = answer_question(
                                vectorstore=st.session_state.vectorstore,
                                question=user_question,
                                api_key=groq_api_key.strip(),
                                model_name=selected_model,
                                language=target_language,
                            )
                            st.write(answer)
                            excerpts = [d.page_content for d in docs]
                            with st.expander("View retrieved article excerpts"):
                                for idx, excerpt in enumerate(excerpts, start=1):
                                    st.caption(f"Excerpt {idx}")
                                    st.write(excerpt)

                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer, "sources": excerpts}
                            )
                        except Exception as exc:
                            st.error(f"Q&A error: {exc}")
