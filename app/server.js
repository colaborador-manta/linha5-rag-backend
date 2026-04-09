const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const Database = require("better-sqlite3");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3001;

// --- Middleware ---
app.use(cors());
app.use(express.json({ limit: "50mb" }));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

// --- Database ---
const fs = require("fs");
const DB_PATH = process.env.DB_PATH || path.join(__dirname, "data", "rag.db");
fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

// Se DB existir mas estiver corrompido, deletar e recriar
let db;
try {
  db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
} catch (err) {
  console.warn("[DB] Banco corrompido, recriando:", err.message);
  try { fs.unlinkSync(DB_PATH); } catch (_) {}
  try { fs.unlinkSync(DB_PATH + "-wal"); } catch (_) {}
  try { fs.unlinkSync(DB_PATH + "-shm"); } catch (_) {}
  db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
}

db.exec(`
  CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT DEFAULT 'geral',
    size INTEGER,
    chunks_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    text_preview TEXT
  );
  CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    doc_type TEXT,
    filename TEXT,
    chunk_index INTEGER,
    text TEXT NOT NULL,
    embedding BLOB,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
  );
  CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
`);

// --- Embedding Engine ---
let pipeline = null;
let modelReady = false;
let modelLoading = false;

async function loadModel() {
  if (pipeline) return pipeline;
  if (modelLoading) {
    while (!modelReady) await new Promise(r => setTimeout(r, 500));
    return pipeline;
  }
  modelLoading = true;
  console.log("[EMBED] Carregando modelo all-MiniLM-L6-v2...");
  const start = Date.now();
  const { pipeline: createPipeline } = await import("@xenova/transformers");
  pipeline = await createPipeline("feature-extraction", "Xenova/multilingual-e5-small", {
    quantized: true,
  });
  modelReady = true;
  modelLoading = false;
  console.log(`[EMBED] Modelo carregado em ${((Date.now() - start) / 1000).toFixed(1)}s`);
  return pipeline;
}

// multilingual-e5 usa prefixos: "query: " para buscas, "passage: " para documentos
async function getEmbedding(text, isQuery = false) {
  const pipe = await loadModel();
  const prefix = isQuery ? "query: " : "passage: ";
  const output = await pipe((prefix + text).slice(0, 512), { pooling: "mean", normalize: true });
  return Buffer.from(new Float32Array(output.data).buffer);
}

function cosineSimBuf(a, b) {
  const va = new Float32Array(a.buffer, a.byteOffset, a.length / 4);
  const vb = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < va.length; i++) { dot += va[i] * vb[i]; na += va[i] * va[i]; nb += vb[i] * vb[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// --- Chunking ---
function chunkText(text, docId, docType, filename) {
  const chunks = [];
  const size = 2000, overlap = 400;
  // Normalizar: colapsar espacos multiplos em 1, manter newlines para paragrafos
  const clean = text.replace(/[^\S\n]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();

  // Tentar split por paragrafos (duplo newline)
  let paras = clean.split(/\n\n+/);
  // Se poucos paragrafos, splittar por newline simples
  if (paras.length < 5) paras = clean.split(/\n/);
  // Se ainda poucos, splittar por sentencas
  if (paras.length < 5) paras = clean.split(/(?<=[.!?])\s+/);

  let current = "", idx = 0;
  for (const para of paras) {
    const p = para.trim();
    if (!p) continue;
    if (current.length + p.length > size && current) {
      chunks.push({ id: `${docId}_${idx}`, doc_id: docId, doc_type: docType, text: current.trim(), filename, chunk_index: idx });
      current = current.slice(-overlap) + " " + p;
      idx++;
    } else {
      current = (current ? current + " " : "") + p;
    }
  }
  if (current.trim()) {
    chunks.push({ id: `${docId}_${idx}`, doc_id: docId, doc_type: docType, text: current.trim(), filename, chunk_index: idx });
  }

  // Fallback: se ficou muito grande ou sem chunks, chunkar por tamanho fixo
  if (chunks.length <= 1 && clean.length > size) {
    chunks.length = 0;
    idx = 0;
    for (let i = 0; i < clean.length; i += (size - overlap)) {
      const slice = clean.slice(i, i + size).trim();
      if (slice.length > 50) {
        chunks.push({ id: `${docId}_${idx}`, doc_id: docId, doc_type: docType, text: slice, filename, chunk_index: idx });
        idx++;
      }
    }
  }

  console.log(`[CHUNK] ${filename}: ${clean.length} chars -> ${chunks.length} chunks (avg ${chunks.length ? Math.round(clean.length / chunks.length) : 0} chars/chunk)`);
  return chunks;
}

// --- Routes ---

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", model_ready: modelReady, docs: db.prepare("SELECT COUNT(*) as c FROM documents").get().c, chunks: db.prepare("SELECT COUNT(*) as c FROM chunks").get().c });
});

// Pre-load model
app.post("/api/model/load", async (req, res) => {
  try {
    await loadModel();
    res.json({ status: "ready" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Upload document (PDF, TXT, MD, CSV)
app.post("/api/documents", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Nenhum arquivo enviado" });

    const file = req.file;
    const docType = req.body.type || "geral";
    const docId = `doc_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    // Extrair texto
    let text;
    const ext = file.originalname.toLowerCase().split(".").pop();
    if (ext === "pdf") {
      const result = await pdfParse(file.buffer);
      text = result.text;
    } else {
      text = file.buffer.toString("utf-8");
    }

    if (!text || text.trim().length < 10) {
      return res.status(400).json({ error: "Arquivo vazio ou sem texto extraivel" });
    }

    // Chunkar
    const chunks = chunkText(text, docId, docType, file.originalname);
    console.log(`[UPLOAD] ${file.originalname}: ${text.length} chars -> ${chunks.length} chunks`);

    // Salvar documento
    db.prepare("INSERT INTO documents (id, name, type, size, chunks_count, text_preview) VALUES (?, ?, ?, ?, ?, ?)").run(
      docId, file.originalname, docType, file.size, chunks.length, text.slice(0, 500)
    );

    // Gerar embeddings e salvar chunks
    const insertChunk = db.prepare("INSERT INTO chunks (id, doc_id, doc_type, filename, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)");
    const insertMany = db.transaction((chunks) => {
      for (const c of chunks) insertChunk.run(c.id, c.doc_id, c.doc_type, c.filename, c.chunk_index, c.text, c.embedding);
    });

    // Computar embeddings
    console.log(`[EMBED] Gerando embeddings para ${chunks.length} chunks...`);
    const start = Date.now();
    for (let i = 0; i < chunks.length; i++) {
      chunks[i].embedding = await getEmbedding(chunks[i].text);
      if ((i + 1) % 10 === 0) console.log(`[EMBED] ${i + 1}/${chunks.length}`);
    }
    console.log(`[EMBED] Concluido em ${((Date.now() - start) / 1000).toFixed(1)}s`);

    insertMany(chunks);

    res.json({
      id: docId,
      name: file.originalname,
      type: docType,
      chunks: chunks.length,
      embed_time_s: ((Date.now() - start) / 1000).toFixed(1)
    });
  } catch (err) {
    console.error("[UPLOAD] Erro:", err);
    res.status(500).json({ error: err.message });
  }
});

// Busca hibrida: vetorial (semantica) + keyword (BM25-like), fusao RRF
app.post("/api/search", async (req, res) => {
  try {
    const { query, top_k = 5, doc_type } = req.body;
    if (!query) return res.status(400).json({ error: "query required" });

    // Buscar chunks
    let sql = "SELECT id, doc_id, doc_type, filename, chunk_index, text, embedding FROM chunks WHERE embedding IS NOT NULL";
    const params = [];
    if (doc_type) { sql += " AND doc_type = ?"; params.push(doc_type); }
    const rows = db.prepare(sql).all(...params);

    // 1. Score vetorial (cosine similarity)
    const queryEmbedding = await getEmbedding(query, true);
    const vectorScores = rows.map(row => ({
      ...row,
      vectorScore: cosineSimBuf(queryEmbedding, row.embedding)
    }));

    // 2. Score keyword (BM25-like com bigramas)
    const norm = (s) => s.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\s+/g, " ").trim();
    const qNorm = norm(query);
    const words = qNorm.split(" ").filter(w => w.length > 2);
    const bigrams = [];
    for (let i = 0; i < words.length - 1; i++) bigrams.push(words[i] + " " + words[i + 1]);

    // IDF
    const df = {};
    words.forEach(w => { df[w] = 0; });
    rows.forEach(r => { const t = norm(r.text); words.forEach(w => { if (t.includes(w)) df[w]++; }); });
    const N = rows.length;
    const idf = {};
    words.forEach(w => { idf[w] = df[w] > 0 ? Math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1) : 0; });

    const keywordScores = vectorScores.map(row => {
      const t = norm(row.text);
      let kwScore = 0;
      // Frase exata
      if (t.includes(qNorm)) kwScore += 15;
      // Bigramas
      bigrams.forEach(bg => { if (t.includes(bg)) kwScore += 5; });
      // Palavras com IDF
      words.forEach(w => { if (t.includes(w)) kwScore += (idf[w] || 1); });
      // Proximidade
      if (words.length >= 2) {
        const positions = words.map(w => t.indexOf(w)).filter(p => p >= 0);
        if (positions.length >= 2) {
          positions.sort((a, b) => a - b);
          const span = positions[positions.length - 1] - positions[0];
          if (span < 200) kwScore += 3;
          if (span < 100) kwScore += 3;
        }
      }
      return { ...row, kwScore };
    });

    // 3. Fusao RRF com peso 2x para keyword (keywords sao mais precisas para termos exatos)
    const k = 60;
    const byVector = [...keywordScores].sort((a, b) => b.vectorScore - a.vectorScore);
    const byKeyword = [...keywordScores].sort((a, b) => b.kwScore - a.kwScore);
    const rrfScores = {};
    byVector.forEach((r, i) => { rrfScores[r.id] = (rrfScores[r.id] || 0) + 1 / (k + i + 1); });
    byKeyword.forEach((r, i) => { rrfScores[r.id] = (rrfScores[r.id] || 0) + 2 / (k + i + 1); }); // peso 2x keyword

    const results = keywordScores.map(row => ({
      id: row.id,
      doc_id: row.doc_id,
      doc_type: row.doc_type,
      filename: row.filename,
      chunk_index: row.chunk_index,
      text: row.text,
      score: rrfScores[row.id] || 0,
      vector_score: row.vectorScore,
      keyword_score: row.kwScore
    })).sort((a, b) => b.score - a.score).slice(0, top_k);

    // Remove embedding blob do response
    results.forEach(r => delete r.embedding);

    console.log(`[SEARCH] "${query.slice(0, 50)}" -> top: chunk ${results[0]?.chunk_index} (rrf=${results[0]?.score.toFixed(4)}, vec=${results[0]?.vector_score.toFixed(4)}, kw=${results[0]?.keyword_score})`);

    res.json({ query, results, total_chunks: rows.length, method: "hybrid_rrf" });
  } catch (err) {
    console.error("[SEARCH] Erro:", err);
    res.status(500).json({ error: err.message });
  }
});

// Listar documentos
app.get("/api/documents", (req, res) => {
  const docs = db.prepare("SELECT id, name, type, size, chunks_count, created_at FROM documents ORDER BY created_at DESC").all();
  res.json(docs);
});

// Deletar documento
app.delete("/api/documents/:id", (req, res) => {
  const { id } = req.params;
  db.prepare("DELETE FROM chunks WHERE doc_id = ?").run(id);
  const result = db.prepare("DELETE FROM documents WHERE id = ?").run(id);
  res.json({ deleted: result.changes > 0 });
});

// Deletar todos
app.delete("/api/documents", (req, res) => {
  db.prepare("DELETE FROM chunks").run();
  db.prepare("DELETE FROM documents").run();
  res.json({ deleted: true });
});

// Stats
app.get("/api/stats", (req, res) => {
  const docs = db.prepare("SELECT COUNT(*) as c FROM documents").get().c;
  const chunks = db.prepare("SELECT COUNT(*) as c FROM chunks").get().c;
  const embedded = db.prepare("SELECT COUNT(*) as c FROM chunks WHERE embedding IS NOT NULL").get().c;
  const types = db.prepare("SELECT doc_type, COUNT(*) as c FROM chunks GROUP BY doc_type").all();
  res.json({ documents: docs, chunks, embedded, model_ready: modelReady, types });
});

// --- Start ---
app.listen(PORT, () => {
  console.log(`\n🚀 RAG Backend rodando em http://localhost:${PORT}`);
  console.log(`   POST /api/documents     — Upload PDF/TXT (multipart/form-data)`);
  console.log(`   POST /api/search        — Busca vetorial { query, top_k }`);
  console.log(`   GET  /api/documents     — Listar documentos`);
  console.log(`   DELETE /api/documents/:id — Remover documento`);
  console.log(`   GET  /api/stats         — Estatisticas`);
  console.log(`   GET  /health            — Health check\n`);

  // Pre-load model em background
  loadModel().catch(err => console.error("[EMBED] Erro ao pre-carregar modelo:", err.message));
});
