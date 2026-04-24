from src.section2.rag_pipeline import RAGPipeline

def test_rag_refusal_on_empty_dir(tmp_path):
    pipeline = RAGPipeline(pdf_dir=str(tmp_path))
    try:
        pipeline.query("What is the notice period?")
        assert False, "Expected FileNotFoundError when no PDFs exist"
    except FileNotFoundError:
        assert True
