import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

const apiKey = process.env.GEMINI_API_KEY;
const model = "gemini-2.0-flash";
const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

export async function askGemini(contextText, question) {
  const prompt = `
You are an insurance policy expert AI.
Based on the following document content:

"${contextText.substring(0, 20000)}"

Answer this question:
"${question}"

Respond ONLY in structured JSON like:
{
  "answer": "Answer for the question in plain text"
}
Do NOT return an array. Only return a single string for the answer.
`;

  try {
    const response = await axios.post(endpoint, {
      contents: [{ parts: [{ text: prompt }] }]
    });

    const content = response.data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    const jsonStart = content.indexOf('{');
    const jsonEnd = content.lastIndexOf('}') + 1;

    const parsed = JSON.parse(content.slice(jsonStart, jsonEnd));

    // Convert array answers to flat strings if needed
    if (Array.isArray(parsed.answer)) {
      parsed.answer = parsed.answer.join(' ').replace(/\s+/g, ' ').trim();
    }

    return parsed;

  } catch (err) {
    console.error("Gemini Error:", err?.response?.data || err.message);
    return {
      answer: "Gemini API call failed or returned an unexpected format."
    };
  }
}
