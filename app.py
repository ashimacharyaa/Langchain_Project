
import streamlit as st

from rag import (
    add_pdf,
    retrieve_evidence,
    generate_answer
)

from verifier import verify_answer


st.set_page_config(
    page_title="Self-Verifying Medical RAG",
    layout="wide"
)

st.title("Self-Verifying Medical AI")
st.caption("Medical RAG with evidence-based claim verification")

# Sidebar: PDF upload
with st.sidebar:
    st.header("Medical Knowledge Base")

    files = st.file_uploader(
        "Upload medical research PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Index PDFs"):
        if not files:
            st.warning("Please upload a PDF first.")
        else:
            total = 0

            with st.spinner("Indexing documents..."):
                for file in files:
                    chunks = add_pdf(file)
                    st.write("PDF:", file.name)
                    st.write("Chunks created:", chunks)
                    total += chunks

            st.success(f"Indexed {total} chunks.")


# Question input
question = st.text_input(
    "Ask a medical research question",
    placeholder="What are is disease?"
)

if st.button("Ask and Verify") and question.strip():

    with st.spinner("Retrieving medical evidence..."):
        evidence = retrieve_evidence(question)

    if not evidence:
        st.warning(
            "No evidence was found. "
            "Please upload relevant medical documents."
        )
        st.stop()

    with st.spinner("Generating initial answer..."):
        answer = generate_answer(question, evidence)

    with st.spinner("Checking medical claims..."):
        try:
            verification = verify_answer(
                question, answer, evidence
            )
        except Exception:
            st.error(
                "Verification failed. "
                "The unverified answer will not be displayed."
            )
            st.stop()

   
    st.subheader("Verification Results")

    supported = [
        claim for claim in verification.claims
        if claim.verdict == "SUPPORTED"
        and claim.source_ids
    ]

    rejected = [
        claim for claim in verification.claims
        if claim.verdict != "SUPPORTED"
        or not claim.source_ids
    ]

    col1, col2 = st.columns(2)

    col1.metric("Supported claims", len(supported))
    col2.metric("Flagged claims", len(rejected))

    if supported:
        st.subheader("Evidence-Supported Findings")

        for claim in supported:
            st.success(
                claim.statement
                + " "
                + " ".join(
                    f"[{sid}]" for sid in claim.source_ids
                )
            )
    else:
        st.warning(
            "No claims passed verification. "
            "The system abstains from answering."
        )

    if rejected:
        with st.expander("Flagged or uncertain claims"):
            for claim in rejected:
                st.warning(
                    f"{claim.verdict}: {claim.statement}"
                )
                st.write(claim.explanation)

    st.subheader("Source Evidence")

    for item in evidence:
        with st.expander(
            f"[{item['id']}] "
            f"{item['source']} — Page {item['page']}"
        ):
            st.write(item["text"])

    with st.expander("Original unverified draft"):
        st.warning(
            "Research/debug view only. "
            "This draft may contain unsupported claims."
        )
        st.write(answer)

st.divider()
st.caption(
    "Research prototype only. Not for clinical diagnosis "
    "or treatment decisions."
)
