import axios from 'axios';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist/legacy/build/pdf.mjs';

// Optional: If you want to set workerSrc manually, though it's not always necessary in Node.js
// GlobalWorkerOptions.workerSrc = './node_modules/pdfjs-dist/legacy/build/pdf.worker.mjs';

export async function fetchAndExtractPDF(url) {
  const response = await axios.get(url, { responseType: "arraybuffer" });
  const data = new Uint8Array(response.data);

  const loadingTask = getDocument({
    data,
    standardFontDataUrl: 'node_modules/pdfjs-dist/standard_fonts/',
  });

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
