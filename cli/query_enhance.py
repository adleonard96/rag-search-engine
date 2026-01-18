import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")
client = genai.Client(api_key=api_key)

def call_model(query):
    client = genai.Client(api_key=api_key)
    return client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=query
    )
    
def spell_check(query):
    prompt = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query.
                Corrected:"""
    
    return call_model(prompt)

def rewrite(query):
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:"""
    return call_model(prompt)

def expand(query):
    prompt = f"""Expand this movie search query with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                This will be appended to the original query.

                Examples:

                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                Query: "{query}"
                """
    return call_model(prompt)

def rerank(query, title, doc):
    prompt = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {title} - {doc}
                
                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness
                
                Rate 0-10 (10 = perfect match).
                Give me ONLY the number in your response, no other text or explanation.
                
                Score:"""
                
    return call_model(prompt)

def batch(query, doc_list_str):
    promt = f"""Rank these movies by relevance to the search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

            [75, 12, 34, 2, 1]
            """
            
    return call_model(promt)

def eval(query, results):
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers out than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""
                
    return call_model(prompt)

def rrf_llm_response(query, docs):
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {docs}

                Provide a comprehensive answer that addresses the query:"""
    
    return call_model(prompt)

def summarize(query, search_results):
    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results:
                {search_results}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
                """
    return call_model(prompt)

def citations(query, search_results):
    prompt = f"""Answer the question or provide information based on the provided documents.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                Query: {query}

                Documents:
                {search_results}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources using [1], [2], etc. format when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""
    
    return call_model(prompt)

def question_the_llm(query, context):
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Question: {query}

                Documents:
                {context}

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation

                Answer:"""
                
    return call_model(prompt)