import arxiv
import os
import nltk
from rake_nltk import Rake

nltk.download("stopwords")
nltk.download("punkt")


def refine_query(query):
    rake = Rake()
    rake.extract_keywords_from_text(query)
    keywords = rake.get_ranked_phrases()
    return " ".join(keywords)

def download_papers(query, numresults, path):
    # Construct the default API client.
    client = arxiv.Client()
    
    # Extract the keywords from the query
    refined_query = refine_query(query)
    results = []
    # Search the arxiv papers based on the keywords
    search = arxiv.Search(
        query=refined_query,
        max_results=numresults,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = client.results(search) #list(search.results())
    # Download the papers
    for i, p in enumerate(papers):
        filename = f"data_{i}.pdf"
        print(f"Downloading {p.title}... to {path} and name {filename}")
        p.download_pdf(dirpath=path, filename=filename)

        #os.unlink(file_path)
        # DEfine the papers metadata
        paper_doc = {"url": p.pdf_url, "path": os.path.join(path, filename), "title": p.title} # Can we extract the abstract from the paper
        results.append(paper_doc)
    return results

