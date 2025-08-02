import express from 'express';
import { fetchAndExtractPDF } from './utils.js';
import { askGemini } from './gemini.js';

const app = express();
app.use(express.json());

app.post("/hackrx/run", async (req, res) => {
  const { documents, questions } = req.body;

  try {
    const text = await fetchAndExtractPDF(documents);
    const answers = [];

    for (const question of questions) {
      const result = await askGemini(text, question);

      // Handle Gemini API errors gracefully
      if (result?.answer) {
        if (Array.isArray(result.answer)) {
          answers.push(result.answer.join(" ")); // flatten multiline answers
        } else {
          answers.push(result.answer.trim());
        }
      } else {
        answers.push("Answer not found or could not be extracted.");
      }
    }

    res.json({ answers });

  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ error: "Failed to process document", details: err.message });
  }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
