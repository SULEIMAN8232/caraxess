import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from agent import run_agent
from tools import search_jobs_tool


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CareerPilot AI",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
PDF_COLLECTION = "career_pilot_uploaded_pdfs"


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.markdown(
    """
    <div style="padding: 20px 0;">
        <h1 style="font-size: 42px;">🎓 CareerPilot AI</h1>
        <p style="font-size: 18px; color: #8da4c4;">
            Learn smarter. Discover opportunities. Build your future.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Ask questions about your course materials or discover job "
    "opportunities based on your skills, location, and salary preferences."
)


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab_learning, tab_jobs, tab_upload = st.tabs(
    [
        "📚 Learning Assistant",
        "💼 Find Jobs",
        "📄 Upload Course PDF",
    ]
)


# --------------------------------------------------
# TAB 1: LEARNING ASSISTANT
# --------------------------------------------------

with tab_learning:
    st.header("📚 Ask About Your Course Materials")

    learning_question = st.text_area(
        "Your question",
        placeholder="Example: Explain how RAG works.",
        key="learning_question",
    )

    if st.button("Ask CareerPilot", key="ask_learning"):

        if not learning_question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("CareerPilot is thinking..."):

                try:
                    answer = run_agent(learning_question)
                    st.markdown("### Answer")
                    st.write(answer)

                except Exception as error:
                    st.error(f"Unable to answer your question: {error}")


# --------------------------------------------------
# TAB 2: JOB SEARCH
# --------------------------------------------------

with tab_jobs:
    st.header("💼 Find Your Next Opportunity")

    st.write(
        "Search for jobs and internships across different industries "
        "using Adzuna."
    )

    with st.form("job_search_form"):

        topic = st.text_input(
            "Job title or skill",
            placeholder=(
                "Python Developer, Accountant, "
                "Nurse, Marketing Executive..."
            ),
        )

        location = st.text_input(
            "Location",
            value="India",
            placeholder="Ahmedabad, Gujarat",
        )

        col1, col2 = st.columns(2)

        with col1:
            minimum_salary = st.number_input(
                "Minimum annual salary",
                min_value=0,
                value=0,
                step=50000,
                help="Enter 0 if you do not want a minimum salary filter.",
            )

        with col2:
            maximum_salary = st.number_input(
                "Maximum annual salary",
                min_value=0,
                value=0,
                step=50000,
                help="Enter 0 if you do not want a maximum salary filter.",
            )

        search_clicked = st.form_submit_button("🔎 Search Jobs")

    if search_clicked:

        if not topic.strip():
            st.warning("Please enter a job title or skill.")

        elif maximum_salary > 0 and minimum_salary > maximum_salary:
            st.warning("Minimum salary cannot exceed maximum salary.")

        else:
            with st.spinner("Searching Adzuna for job listings..."):

                try:
                    result = search_jobs_tool.invoke(
                        {
                            "topic": topic.strip(),
                            "location": location.strip() or "India",
                            "minimum_salary": int(minimum_salary),
                            "maximum_salary": int(maximum_salary),
                        }
                    )

                    st.markdown("### 🔎 Search Results")
                    st.markdown(result)

                except Exception as error:
                    st.error(f"Job search failed: {error}")


# --------------------------------------------------
# TAB 3: PDF UPLOAD AND QUESTION ANSWERING
# --------------------------------------------------

with tab_upload:
    st.header("📄 Upload Your Course Material")

    st.write(
        "Upload a PDF of your course notes or textbook. "
        "CareerPilot will index the document and answer questions "
        "using its contents."
    )

    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="course_pdf_upload",
    )

    # ----------------------------------------------
    # PROCESS UPLOADED PDF
    # ----------------------------------------------

    if uploaded_pdf is not None:

        st.caption(
            f"Selected file: {uploaded_pdf.name} "
            f"({uploaded_pdf.size / 1024:.1f} KB)"
        )

        if st.button("Process PDF", key="process_pdf"):

            temporary_path = None

            try:
                from document_loader import load_and_split_document
                from langchain_chroma import Chroma
                from langchain_ollama import OllamaEmbeddings

                with st.spinner(
                    "Reading, splitting, and indexing your PDF..."
                ):

                    # 1. Calculate a stable hash for this PDF.
                    file_bytes = uploaded_pdf.getvalue()

                    file_hash = hashlib.sha256(
                        file_bytes
                    ).hexdigest()

                    # 2. Save the uploaded PDF temporarily.
                    with NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    ) as temporary_file:

                        temporary_file.write(file_bytes)
                        temporary_path = Path(temporary_file.name)

                    # 3. Load and split the PDF.
                    chunks = load_and_split_document(
                        str(temporary_path)
                    )

                    st.write(
                        f"Extracted {len(chunks)} text chunks."
                    )

                    if not chunks:

                        st.error(
                            "No readable text was found in this PDF. "
                            "If it contains scanned pages, OCR may be needed."
                        )

                    else:
                        # 4. Attach metadata to every chunk.
                        for chunk in chunks:
                            chunk.metadata["file_hash"] = file_hash
                            chunk.metadata["uploaded_filename"] = (
                                uploaded_pdf.name
                            )

                        # 5. Initialize embeddings.
                        embeddings = OllamaEmbeddings(
                            model="nomic-embed-text"
                        )

                        # 6. Open the dedicated PDF collection.
                        pdf_store = Chroma(
                            collection_name=PDF_COLLECTION,
                            embedding_function=embeddings,
                            persist_directory=CHROMA_DIR,
                        )

                        # 7. Remove previous chunks for this same PDF.
                        # This allows the same file to be reprocessed.
                        try:
                            pdf_store.delete(
                                where={"file_hash": file_hash}
                            )
                        except Exception as delete_error:
                            st.warning(
                                "Could not remove previous indexed "
                                f"chunks: {delete_error}"
                            )

                        # 8. Add the new chunks with stable IDs.
                        chunk_ids = [
                            f"{file_hash}_{index}"
                            for index in range(len(chunks))
                        ]

                        pdf_store.add_documents(
                            documents=chunks,
                            ids=chunk_ids,
                        )

                        # 9. Save the active PDF details in the session.
                        st.session_state["uploaded_pdf_name"] = (
                            uploaded_pdf.name
                        )

                        st.session_state["uploaded_pdf_hash"] = (
                            file_hash
                        )

                        st.session_state["uploaded_pdf_processed"] = True

                        st.success(
                            f"Successfully indexed {len(chunks)} "
                            f"text chunks from {uploaded_pdf.name}!"
                        )

            except Exception as error:
                st.error(f"Could not process PDF: {error}")

            finally:
                # 10. Remove the temporary PDF file.
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    # ----------------------------------------------
    # DISPLAY ACTIVE PDF
    # ----------------------------------------------

    if st.session_state.get("uploaded_pdf_processed", False):

        st.success(
            "Active document: "
            + st.session_state.get(
                "uploaded_pdf_name",
                "Uploaded PDF",
            )
        )

        # ------------------------------------------
        # ASK QUESTIONS ABOUT THE PDF
        # ------------------------------------------

        pdf_question = st.text_input(
            "Ask a question about your uploaded PDF",
            placeholder=(
                "Example: Summarize the main concepts in this document."
            ),
            key="pdf_question",
        )

        if st.button("Ask About My PDF", key="ask_pdf"):

            if not pdf_question.strip():
                st.warning("Please enter a question.")

            else:
                try:
                    from langchain_chroma import Chroma
                    from langchain_ollama import OllamaEmbeddings
                    from langchain_ollama import ChatOllama
                    from langchain_core.messages import (
                        HumanMessage,
                        SystemMessage,
                    )

                    with st.spinner(
                        "Searching the PDF and generating an answer..."
                    ):

                        # 1. Open the same PDF collection.
                        embeddings = OllamaEmbeddings(
                            model="nomic-embed-text"
                        )

                        pdf_store = Chroma(
                            collection_name=PDF_COLLECTION,
                            embedding_function=embeddings,
                            persist_directory=CHROMA_DIR,
                        )

                        # 2. Retrieve passages from the active PDF only.
                        active_hash = st.session_state[
                            "uploaded_pdf_hash"
                        ]

                        results = pdf_store.similarity_search(
                            pdf_question,
                            k=4,
                            filter={
                                "file_hash": active_hash
                            },
                        )

                        if not results:

                            st.warning(
                                "No relevant passages were found "
                                "in this PDF. Try a different question."
                            )

                        else:
                            # 3. Build context from retrieved passages.
                            context = "\n\n".join(
                                document.page_content
                                for document in results
                            )

                            # 4. Ask the local language model.
                            model = ChatOllama(
                                model="llama3.2:3b",
                                temperature=0,
                            )

                            response = model.invoke(
                                [
                                    SystemMessage(
                                        content=(
                                            "You are CareerPilot, a course "
                                            "document assistant. Answer the "
                                            "user's question using only the "
                                            "provided PDF passages. Do not "
                                            "invent facts. If the answer is "
                                            "not present in the passages, "
                                            "say that you could not find it "
                                            "in the uploaded document. "
                                            "Keep the answer clear and useful."
                                        )
                                    ),
                                    HumanMessage(
                                        content=(
                                            f"Question:\n{pdf_question}\n\n"
                                            f"Retrieved PDF passages:\n{context}"
                                        )
                                    ),
                                ]
                            )

                            # 5. Display the answer.
                            st.markdown("### 💡 Answer")
                            st.write(response.content)

                            # 6. Show the evidence.
                            with st.expander(
                                "📖 View retrieved passages"
                            ):

                                for index, document in enumerate(
                                    results,
                                    start=1,
                                ):

                                    st.markdown(
                                        f"**Passage {index}**"
                                    )

                                    st.write(
                                        document.page_content
                                    )

                                    st.caption(
                                        "Source: "
                                        + str(
                                            document.metadata.get(
                                                "uploaded_filename",
                                                "Uploaded PDF",
                                            )
                                        )
                                    )

                except Exception as error:
                    st.error(
                        f"Could not answer your question: {error}"
                    )