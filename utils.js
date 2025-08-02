import axios from 'axios';
import { getDocument } from 'pdfjs-dist';

export async function fetchAndExtractPDF(url) {
  const response = await axios.get(url, { responseType: "arraybuffer" });
  const data = new Uint8Array(response.data);

  const loadingTask = getDocument({ data });
  const pdf = await loadingTask.promise;

  let text = '';
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map((item) => item.str).join(' ');
    text += pageText + '\n';
  }

  return text;
}
