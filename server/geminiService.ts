import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import { detectLanguage } from './indianLanguages.js';

// Initialize Gemini with server-side API key and required User-Agent
let ai: GoogleGenAI | null = null;

function getGenAIClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey.trim() === '') {
    throw new Error('GEMINI_API_KEY is not configured on the server. Please verify your environment secrets.');
  }

  if (!ai || (ai as any)['apiKey'] !== apiKey) {
    ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return ai;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  image?: {
    mimeType: string;
    data: string; // base64 without prefix or with prefix
  };
  language?: string;
}

const SOLVA_SYSTEM_INSTRUCTION = `
You are SOLVA, an advanced, highly articulate, and empathetic multilingual and multimodal AI assistant.
Your core mission is natural, human-friendly communication across text, voice, and vision.

Comprehensive Indian Multilingual Fluency:
1. You natively understand and seamlessly communicate in all 22 constitutionally scheduled Indian languages:
   * Assamese (অসমীয়া), Bengali (বাংলা), Bodo (बड़ो), Dogri (डोगरी), Gujarati (ગુજરાતી),
   * Hindi (हिन्दी), Kannada (ಕನ್ನಡ), Kashmiri (कॉशुर / کٲشُر), Konkani (कोंकणी), Maithili (मैथिली),
   * Malayalam (മലയാളം), Manipuri (মৈতৈলোন্ / ꯃꯤꯇꯩꯂꯣꯟ), Marathi (मराठी), Nepali (नेपाली),
   * Odia (ଓଡ଼ିଆ), Punjabi (ਪੰਜਾਬੀ), Sanskrit (संस्कृतम्), Santali (ᱥᱟᱱᱛᱟᱲᱤ), Sindhi (सिन्धी / سنڌي),
   * Tamil (தமிழ்), Telugu (తెలుగు), Urdu (اردو), and Indian English.

2. Romanised Colloquial Typing:
   - When users write in Tanglish (Tamil in Latin script, e.g. "Vanakkam, epdi irukinga? Enna seyyalam?"), reply in natural, friendly Tanglish.
   - When users write in Hinglish (e.g. "Namaste SOLVA, kaise ho? Kya chal raha hai?"), reply in natural Hinglish.
   - When users write in Telugish, Kanglish, or Manglish, match their dialect and conversational flow.
   - When users type in native Indic scripts, reply in the same native Indic script.

3. Multimodal & Vision Understanding:
   - When an image is provided, provide thorough, accurate, and insightful reasoning.
   - Accurately describe objects, identify text/signboards in Indic scripts, analyze documents, diagrams, and photographic details.
   - Respond directly to the user's specific prompt regarding the image.

4. Personality & Tone:
   - Futuristic, warm, friendly, concise, and respectful.
   - Format responses cleanly with structured markdown, bullet points, and code blocks where helpful.
`;

// Helper delay for exponential backoff
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Executes a single generateContent call with a bounded timeout
 */
async function callGeminiModel(
  client: GoogleGenAI,
  modelName: string,
  formattedContents: any[],
  systemInstruction: string,
  timeoutMs: number = 25000
): Promise<string> {
  const config: Record<string, any> = {
    systemInstruction,
  };

  // Configure thinking level:
  // For 'gemini-3.8-flash', set ThinkingLevel.LOW to eliminate huge latency and prevent timeouts
  // For 'gemini-3.1-flash-lite', ThinkingLevel.MINIMAL is default
  if (modelName === 'gemini-3.8-flash') {
    config.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
  }

  let timerId: NodeJS.Timeout | null = null;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timerId = setTimeout(
      () => reject(new Error(`Timeout after ${timeoutMs}ms waiting for ${modelName}`)),
      timeoutMs
    );
  });

  const requestPromise = client.models.generateContent({
    model: modelName,
    contents: formattedContents,
    config,
  });

  try {
    const response = await Promise.race([requestPromise, timeoutPromise]);
    const text = response?.text;
    if (!text || text.trim() === '') {
      throw new Error(`Empty response returned from ${modelName}`);
    }
    return text;
  } finally {
    if (timerId) {
      clearTimeout(timerId);
    }
  }
}

export async function generateChatResponse(
  messages: ChatMessage[],
  preferredLanguage?: string
): Promise<{ text: string; detectedLanguage: string; model: string }> {
  const client = getGenAIClient();

  // Filter and format messages for Gemini
  const formattedContents = messages
    .filter((msg) => (msg.content && msg.content.trim() !== '') || (msg.image && msg.image.data))
    .map((msg) => {
      const role = msg.role === 'assistant' ? 'model' : 'user';
      const parts: any[] = [];

      // Include image if present
      if (msg.image && msg.image.data) {
        let base64Clean = msg.image.data;
        if (base64Clean.includes('base64,')) {
          base64Clean = base64Clean.split('base64,')[1];
        }
        base64Clean = base64Clean.trim();

        if (base64Clean.length > 0) {
          parts.push({
            inlineData: {
              mimeType: msg.image.mimeType || 'image/jpeg',
              data: base64Clean,
            },
          });
        }
      }

      // Add text content
      const textContent = (msg.content || '').trim();
      if (textContent) {
        parts.push({ text: textContent });
      } else if (parts.length === 0) {
        parts.push({ text: 'Please analyze this image and assist me.' });
      }

      return {
        role,
        parts,
      };
    });

  if (formattedContents.length === 0) {
    throw new Error('No valid message content or image provided.');
  }

  // Language customization
  let dynamicInstruction = SOLVA_SYSTEM_INSTRUCTION;
  if (preferredLanguage && preferredLanguage !== 'auto') {
    dynamicInstruction += `\nUser's explicitly selected language preference: ${preferredLanguage}. Please formulate your response in this language and writing style.`;
  }

  // Supported model fallback hierarchy:
  // 1. 'gemini-3.8-flash' (Primary conversational reasoning model)
  // 2. 'gemini-3.1-flash-lite' (Fast, lightweight fallback model)
  // 3. 'gemini-flash-latest' (General purpose flash fallback)
  const modelsToAttempt = ['gemini-3.8-flash', 'gemini-3.1-flash-lite', 'gemini-flash-latest'];
  let lastError: any = null;
  let chosenModel = modelsToAttempt[0];
  let responseText = '';

  for (const model of modelsToAttempt) {
    chosenModel = model;
    // Maximum of two retries (total 3 attempts per model: initial + 2 retries)
    const maxRetries = 2;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        responseText = await callGeminiModel(
          client,
          model,
          formattedContents,
          dynamicInstruction,
          attempt === 0 ? 25000 : 30000
        );
        lastError = null;
        break; // Successfully got response from model!
      } catch (err: any) {
        lastError = err;
        const errMsg = err?.message || '';
        const isTemporary =
          err?.status === 503 ||
          err?.status === 429 ||
          errMsg.includes('503') ||
          errMsg.includes('UNAVAILABLE') ||
          errMsg.includes('high demand') ||
          errMsg.includes('resource exhausted') ||
          errMsg.includes('quota') ||
          errMsg.includes('Timeout');

        console.warn(
          `[SOLVA Gemini] Attempt ${attempt + 1}/${maxRetries + 1} failed for ${model}: ${errMsg.slice(0, 160)}`
        );

        if (isTemporary && attempt < maxRetries) {
          // Short exponential backoff: 500ms for retry 1, 1000ms for retry 2
          const backoffMs = (attempt + 1) * 500;
          await delay(backoffMs);
          continue;
        } else {
          // Max retries reached or non-temporary error; fall back to next supported model
          break;
        }
      }
    }

    if (responseText) {
      break;
    }
  }

  // If all models and retries fail, do NOT claim success. Throw a clear, useful error.
  if (!responseText) {
    console.error('[SOLVA Gemini] All supported models and retries exhausted:', lastError);
    const is503 =
      lastError?.status === 503 ||
      lastError?.message?.includes('503') ||
      lastError?.message?.includes('UNAVAILABLE') ||
      lastError?.message?.includes('high demand');

    const is429 =
      lastError?.status === 429 ||
      lastError?.message?.includes('429') ||
      lastError?.message?.includes('quota') ||
      lastError?.message?.includes('resource exhausted');

    if (is503) {
      const err = new Error(
        'The Gemini AI service is currently experiencing temporary high demand (503 UNAVAILABLE). Automatic retries and fallback models were attempted. Please tap Retry to try again.'
      );
      (err as any).status = 503;
      throw err;
    } else if (is429) {
      const err = new Error(
        'Gemini API quota limit reached. Please wait a brief moment and tap Retry.'
      );
      (err as any).status = 429;
      throw err;
    } else {
      throw lastError || new Error('Unable to generate response from Gemini at this time.');
    }
  }

  // Detect language of the query and the response
  const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user')?.content || '';
  const detection = detectLanguage(lastUserMsg || responseText);

  return {
    text: responseText,
    detectedLanguage: detection.name,
    model: chosenModel,
  };
}
