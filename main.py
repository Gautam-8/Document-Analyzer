from fastmcp import FastMCP
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import re
from textblob import TextBlob
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

# File processing imports
from docx import Document
from io import BytesIO
import PyPDF2

# Document storage utilities
DOCUMENTS_FILE = "documents.json"

def load_documents() -> Dict[str, Any]:
    """Load documents from JSON file"""
    if not os.path.exists(DOCUMENTS_FILE):
        return {"documents": {}, "metadata": {"total_documents": 0, "last_updated": None, "next_id": 1}}
    
    with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_documents(data: Dict[str, Any]) -> None:
    """Save documents to JSON file"""
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Analysis functions
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        # Access sentiment attributes more explicitly
        polarity = 0.0
        subjectivity = 0.0
        
        if hasattr(blob, 'sentiment'):
            sentiment_data = blob.sentiment
            if hasattr(sentiment_data, 'polarity'):
                polarity = float(sentiment_data.polarity)  # type: ignore
            if hasattr(sentiment_data, 'subjectivity'):
                subjectivity = float(sentiment_data.subjectivity)  # type: ignore
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "confidence": round(abs(polarity), 3)
        }
    except Exception as e:
        return {
            "sentiment": "neutral",
            "polarity": 0.0,
            "subjectivity": 0.0,
            "confidence": 0.0,
            "error": str(e)
        }

def extract_keywords_from_text(text: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Extract keywords from text using TF-IDF with sklearn"""
    try:
        # Clean and preprocess text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into sentences for TF-IDF
        sentences = [sent.strip() for sent in re.split(r'[.!?]+', text) if sent.strip()]
        
        # If we have only one sentence, split into smaller chunks
        if len(sentences) < 2:
            words = text.split()
            chunk_size = max(10, len(words) // 5)  # Create chunks of reasonable size
            sentences = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        # Ensure we have at least 2 documents for TF-IDF
        if len(sentences) < 2:
            sentences = [text, text]  # Duplicate if needed
        
        # Create TF-IDF vectorizer with custom settings
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
        )
        
        # Fit and transform the text
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Get feature names (words/phrases)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF scores across all documents
        # Convert sparse matrix to dense array for mean calculation
        dense_matrix = tfidf_matrix.toarray()  # type: ignore
        mean_scores = np.mean(dense_matrix, axis=0)
        
        # Create keyword-score pairs
        keyword_scores = list(zip(feature_names, mean_scores))
        
        # Sort by TF-IDF score (descending)
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to desired format with normalized frequency scores
        max_score = keyword_scores[0][1] if keyword_scores else 1
        results = []
        
        for keyword, score in keyword_scores[:limit]:
            # Normalize score to a frequency-like integer (1-100)
            normalized_freq = max(1, int((score / max_score) * 100))
            results.append({"keyword": keyword, "frequency": normalized_freq})
        
        return results
        
    except Exception as e:
        return [{"error": str(e)}]

def calculate_readability(text: str) -> Dict[str, Any]:
    """Calculate readability scores using textstat"""
    try:
        return {
            "flesch_reading_ease": round(getattr(textstat, 'flesch_reading_ease', lambda x: 0)(text), 2),
            "flesch_kincaid_grade": round(getattr(textstat, 'flesch_kincaid_grade', lambda x: 0)(text), 2),
            "automated_readability_index": round(getattr(textstat, 'automated_readability_index', lambda x: 0)(text), 2),
            "dale_chall_readability_score": round(getattr(textstat, 'dale_chall_readability_score', lambda x: 0)(text), 2)
        }
    except Exception as e:
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "automated_readability_index": 0,
            "dale_chall_readability_score": 0,
            "error": str(e)
        }

def calculate_basic_stats(text: str) -> Dict[str, Any]:
    """Calculate basic text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(text.replace(' ', '')),
        "sentence_count": len(sentences),
        "average_words_per_sentence": round(len(words) / len(sentences) if sentences else 0, 2),
        "average_characters_per_word": round(len(text.replace(' ', '')) / len(words) if words else 0, 2)
    }

mcp = FastMCP(name="Document Analyzer", instructions="You are a document analyzer. You are given a document and you need to analyze it. You need to analyze the document and return the analysis in a JSON format.")   

@mcp.tool
def get_sentiment(text: str) -> str:
    """Sentiment analysis for any text"""
    try:
        result = analyze_sentiment(text)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze sentiment: {str(e)}"})

@mcp.tool
def extract_keywords(text: str, limit: int = 10) -> str:
    """Extract top keywords from text"""
    try:
        result = extract_keywords_from_text(text, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to extract keywords: {str(e)}"})

@mcp.tool
def add_document(title: str, file_content: str, file_type: str, author: str = "Unknown", category: str = "General", tags: List[str] | None = None) -> str:
    """Add a document from file content (base64 encoded)"""
    try:
        if tags is None:
            tags = []
        
        # Validate inputs
        if not title.strip():
            return json.dumps({"error": "Title cannot be empty"})
        
        if not file_content.strip():
            return json.dumps({"error": "File content cannot be empty"})
        
        if not file_type.strip():
            return json.dumps({"error": "File type must be specified"})
        
        # Validate file type
        supported_types = ['txt', 'docx', 'pdf']
        if file_type.lower() not in supported_types:
            return json.dumps({"error": f"Unsupported file type: {file_type}. Supported types: {', '.join(supported_types)}"})
        
        final_content = ""
        
        try:
            import base64
            # Decode base64 content
            decoded_content = base64.b64decode(file_content)
            
            # Process based on file type
            if file_type.lower() == 'txt':
                final_content = decoded_content.decode('utf-8')
            elif file_type.lower() == 'docx':
                final_content = extract_text_from_docx_bytes(decoded_content)
            elif file_type.lower() == 'pdf':
                final_content = extract_text_from_pdf_bytes(decoded_content)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to process file content: {str(e)}"})
        
        if not final_content.strip():
            return json.dumps({"error": "File appears to be empty or contains no readable text"})
        
        data = load_documents()
        doc_id = str(data["metadata"]["next_id"])
        
        # Calculate basic stats
        basic_stats = calculate_basic_stats(final_content)
        
        document_entry = {
            "id": doc_id,
            "title": title.strip(),
            "content": final_content.strip(),
            "author": author.strip(),
            "category": category.strip(),
            "tags": [tag.strip() for tag in tags if tag.strip()],
            "created_at": datetime.now().isoformat(),
            "word_count": basic_stats.get("word_count", 0),
            "character_count": basic_stats.get("character_count", 0),
            "basic_stats": basic_stats,
            "source_file": {
                "file_type": file_type.lower(),
                "file_size": len(base64.b64decode(file_content)),
                "processed_from": "base64_content"
            }
        }
        
        data["documents"][doc_id] = document_entry
        data["metadata"]["total_documents"] += 1
        data["metadata"]["next_id"] += 1
        
        save_documents(data)
        
        return json.dumps({
            "status": "success",
            "message": f"Document '{title}' added successfully from {file_type.upper()} file",
            "document_id": doc_id,
            "document_info": {
                "title": title,
                "author": author,
                "category": category,
                "tags": tags,
                "word_count": basic_stats.get("word_count", 0),
                "character_count": basic_stats.get("character_count", 0),
                "file_type": file_type.lower(),
                "file_size": len(base64.b64decode(file_content))
            }
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to add document: {str(e)}"})

# Helper functions for processing file bytes
def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """Extract text from DOCX file bytes"""
    try:
        doc = Document(BytesIO(file_bytes))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n'.join(paragraphs)
    except Exception as e:
        raise Exception(f"Failed to read DOCX content: {str(e)}")

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text_content = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)
            except Exception as e:
                text_content.append(f"[Error reading page {page_num + 1}: {str(e)}]")
        
        return '\n'.join(text_content)
    except Exception as e:
        raise Exception(f"Failed to read PDF content: {str(e)}")

@mcp.tool
def search_documents(query: str) -> str:
    """Search documents by content, title, or tags"""
    try:
        data = load_documents()
        results = []
        query_lower = query.lower()
        
        for doc_id, doc in data["documents"].items():
            # Search in title, content, author, category, and tags
            searchable_text = " ".join([
                doc.get("title", ""),
                doc.get("content", ""),
                doc.get("author", ""),
                doc.get("category", ""),
                " ".join(doc.get("tags", []))
            ]).lower()
            
            if query_lower in searchable_text:
                # Calculate relevance score (simple word count)
                relevance = searchable_text.count(query_lower)
                results.append({
                    "document_id": doc_id,
                    "title": doc.get("title", ""),
                    "author": doc.get("author", ""),
                    "category": doc.get("category", ""),
                    "relevance_score": relevance,
                    "preview": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return json.dumps({
            "query": query,
            "total_results": len(results),
            "results": results[:10]  # Limit to top 10 results
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to search documents: {str(e)}"})

@mcp.tool
def analyze_document(document_id: str) -> str:
    """Complete analysis of a document by ID"""
    try:
        data = load_documents()
        
        if document_id not in data["documents"]:
            return json.dumps({"error": f"Document with ID '{document_id}' not found"})
        
        doc = data["documents"][document_id]
        content = doc.get("content", "")
        
        # Always perform fresh analysis
        sentiment_analysis = analyze_sentiment(content)
        keyword_analysis = extract_keywords_from_text(content, 15)
        readability_analysis = calculate_readability(content)
        basic_stats = doc.get("basic_stats", calculate_basic_stats(content))
        
        result = {
            "document_info": {
                "id": document_id,
                "title": doc.get("title", ""),
                "author": doc.get("author", ""),
                "category": doc.get("category", ""),
                "tags": doc.get("tags", []),
                "created_at": doc.get("created_at", ""),
                "source_file": doc.get("source_file")
            },
            "analysis": {
                "sentiment": sentiment_analysis,
                "keywords": keyword_analysis,
                "readability": readability_analysis,
                "basic_stats": basic_stats
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze document: {str(e)}"})

if __name__ == "__main__":
    mcp.run()