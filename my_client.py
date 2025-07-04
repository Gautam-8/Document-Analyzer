import asyncio
import base64
from fastmcp import Client

client = Client("main.py")

async def call_all_tools():
    async with client:
        print("🚀 Testing all Document Analyzer MCP tools")
        print("=" * 50)
        
        # Test 1: Sentiment Analysis
        print("\n📊 Test 1: Sentiment Analysis")
        result = await client.call_tool("get_sentiment", {
            "text": "I love this new TF-IDF implementation! It works amazingly well."
        })
        print('Sentiment Result:', result)
        
        # Test 2: Keyword Extraction  
        print("\n🔍 Test 2: Keyword Extraction")
        result = await client.call_tool("extract_keywords", {
            "text": "Machine learning and artificial intelligence are transforming technology. Neural networks and deep learning algorithms solve complex problems.",
            "limit": 8
        })
        print('Keywords Result:', result)
        
        # Test 3: Add Document
        print("\n📄 Test 3: Add Document")
        # Create base64 content
        content = "This is a test document about blockchain technology and cryptocurrency. Bitcoin and Ethereum are leading cryptocurrencies in the digital finance revolution."
        base64_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        result = await client.call_tool("add_document", {
            "title": "Blockchain Technology Overview",
            "file_content": base64_content,
            "file_type": "txt",
            "author": "Tech Writer",
            "category": "Technology",
            "tags": ["blockchain", "cryptocurrency", "bitcoin", "ethereum"]
        })
        print('Add Document Result:', result)
        
        # Test 4: Search Documents
        print("\n🔎 Test 4: Search Documents")
        result = await client.call_tool("search_documents", {
            "query": "artificial intelligence"
        })
        print('Search Result:', result)
        
        # Test 5: Analyze Document
        print("\n📈 Test 5: Analyze Document")
        result = await client.call_tool("analyze_document", {"document_id": "5"})
        print('Analysis Result:', result)
        
        print("\n✅ All tools tested successfully!")

asyncio.run(call_all_tools())