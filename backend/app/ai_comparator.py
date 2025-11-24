# backend/app/ai_comparator.py

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("PPLX_API_KEY") or os.getenv("API_KEY")

client = None
if API_KEY:
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.perplexity.ai"
    )
else:
    logger.warning("⚠️ No API Key found in .env file. AI comparison will fail.")

def compare_with_ai(standard_text: str, other_text: str) -> str:
    """
    Sends documents to Perplexity AI for expert comparison.
    Returns a concise Markdown formatted report.
    """
    if not client:
        return "❌ **Error:** API Key is missing. Please check your .env file."

    std_trunc = standard_text[:50000] 
    oth_trunc = other_text[:50000]

    # ✅ تحديث الـ Prompt لكي يكون التقرير مختصراً ومركزاً
    system_prompt = """
    You are an expert Legal Contract Analyst and Risk Consultant.
    Your task is to compare a "Standard/Master Document" against a "Second Document" and identify deviations.
    
    **CRITICAL: Keep the report CONCISE and FOCUSED.**
    
    **OUTPUT FORMAT RULES:**
    - Use these exact emojis for headers: ❌ (Missing), ➕ (Added), 🔄 (Modified).
    
    **Report Structure (MUST BE BRIEF):**
    
    1. **Executive Summary** (3-4 lines ONLY):
       - State the match percentage (e.g., 75%, 90%).
       - Highlight ONLY the most critical 1-2 risks.
       - End with a one-sentence recommendation.
    
    2. **Detailed Analysis** (Use bullet points, max 5 items per section):
       - **❌ MISSING CLAUSES**: List ONLY the most important missing items (max 5).
       - **➕ ADDED CLAUSES**: List ONLY significant additions (max 3).
       - **🔄 MODIFICATIONS**: List ONLY critical changes that affect enforceability or risk (max 5).
    
    **Important:**
    - Do NOT include long paragraphs or explanations.
    - Each bullet point should be ONE line (max 15 words).
    - Focus on HIGH-RISK items only (ignore minor formatting or typos unless they are critical).
    - Write in English for maximum accuracy.
    - Be direct and actionable.
    """

    user_prompt = f"""
    **Standard Document (The Benchmark):**
    ---
    {std_trunc}
    ---

    **Second Document (To be Reviewed):**
    ---
    {oth_trunc}
    ---

    Please provide a CONCISE expert comparison report following the structure above.
    """

    try:
        logger.info("🚀 Sending request to Perplexity AI...")
        
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )

        report = response.choices[0].message.content
        logger.info("✅ AI Analysis received successfully.")
        return report

    except Exception as e:
        logger.error(f"❌ AI Request Failed: {e}")
        return f"❌ **Error during AI comparison:**\n{str(e)}\n\nPlease check your API Key credits or connection."
