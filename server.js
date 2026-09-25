const express = require("express");
const cors = require("cors");
const bcrypt = require("bcryptjs");
const sqlite3 = require("sqlite3").verbose();
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = Number(process.env.PORT) || 4000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "translator")));
app.use(express.static(path.join(__dirname, "public")));
// Serve HTML files sitting in the root folder automatically
app.use(express.static(__dirname));

// Route the base landing domain URL straight to your login page
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "login.html"));
});

// SQLite database file in project root
const db = new sqlite3.Database(path.join(__dirname, "users.db"));
const DATA_FILES = {
  videos: path.join(__dirname, "videos.json"),
  news: path.join(__dirname, "news.json"),
};

function readJsonArray(filePath) {
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    const data = JSON.parse(raw || "[]");
    return Array.isArray(data) ? data : [];
  } catch (e) {
    return [];
  }
}

function writeJsonArray(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf8");
}

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'user',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);
  db.run(`ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'`, (err) => {
    if (err && !String(err.message || "").includes("duplicate column")) {
      console.error("Failed to add role column:", err.message);
    }
  });
  db.run(`
    CREATE TABLE IF NOT EXISTS practice (
      user_id INTEGER PRIMARY KEY,
      progress_json TEXT NOT NULL,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id)
    )
  `);
});

function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function passwordIssues(password) {
  const issues = [];
  if (typeof password !== "string") return ["Password is required"];
  if (password.length < 8) issues.push("at least 8 characters");
  if (!/[a-z]/.test(password)) issues.push("a lowercase letter");
  if (!/[A-Z]/.test(password)) issues.push("an uppercase letter");
  if (!/[0-9]/.test(password)) issues.push("a number");
  if (!/[^\w\s]/.test(password)) issues.push("a special character");
  return issues;
}

// JSON list of users
app.get("/api/users", (req, res) => {
  db.all(
    "SELECT id, name, email, role, created_at FROM users ORDER BY id DESC",
    [],
    (err, rows) => {
      if (err) return res.status(500).json({ error: "Database error" });
      res.json(rows);
    }
  );
});

// Admin: update role by email (basic, relies on admin UI gate)
app.post("/api/admin/role", (req, res) => {
  const { email, role } = req.body || {};
  if (!email || !role) {
    return res.status(400).json({ error: "Email and role are required" });
  }
  const cleanRole = String(role).toLowerCase();
  if (!["admin", "user"].includes(cleanRole)) {
    return res.status(400).json({ error: "Invalid role" });
  }

  db.run(
    "UPDATE users SET role = ? WHERE email = ?",
    [cleanRole, email],
    function (err) {
      if (err) return res.status(500).json({ error: "Database error" });
      res.json({ updated: this.changes });
    }
  );
});

// Admin content: videos (stored in videos.json)
app.get("/api/admin/videos", (req, res) => {
  const videos = readJsonArray(DATA_FILES.videos);
  res.json(videos);
});

app.get("/api/admin/videos/:id", (req, res) => {
  const id = Number(req.params.id);
  const videos = readJsonArray(DATA_FILES.videos);
  const item = videos.find((v) => Number(v.id) === id);
  if (!item) return res.status(404).json({ error: "Not found" });
  res.json(item);
});

app.post("/api/admin/videos", (req, res) => {
  const { id, title, youtube } = req.body || {};
  const numId = Number(id);
  if (!numId || !title || !youtube) {
    return res
      .status(400)
      .json({ error: "id, title, youtube are required" });
  }
  const videos = readJsonArray(DATA_FILES.videos);
  if (videos.some((v) => Number(v.id) === numId)) {
    return res.status(400).json({ error: "ID already exists" });
  }
  videos.push({ id: numId, title, youtube });
  writeJsonArray(DATA_FILES.videos, videos);
  res.json({ ok: true });
});

app.put("/api/admin/videos/:id", (req, res) => {
  const id = Number(req.params.id);
  const { title, youtube } = req.body || {};
  if (!id || !title || !youtube) {
    return res
      .status(400)
      .json({ error: "title and youtube are required" });
  }
  const videos = readJsonArray(DATA_FILES.videos);
  const idx = videos.findIndex((v) => Number(v.id) === id);
  if (idx === -1) return res.status(404).json({ error: "Not found" });
  videos[idx] = { ...videos[idx], title, youtube };
  writeJsonArray(DATA_FILES.videos, videos);
  res.json({ ok: true });
});

app.delete("/api/admin/videos/:id", (req, res) => {
  const id = Number(req.params.id);
  const videos = readJsonArray(DATA_FILES.videos);
  const next = videos.filter((v) => Number(v.id) !== id);
  if (next.length === videos.length) {
    return res.status(404).json({ error: "Not found" });
  }
  writeJsonArray(DATA_FILES.videos, next);
  res.json({ ok: true });
});

// Admin content: news (stored in news.json)
app.get("/api/admin/news", (req, res) => {
  const news = readJsonArray(DATA_FILES.news);
  res.json(news);
});

app.get("/api/admin/news/:id", (req, res) => {
  const id = Number(req.params.id);
  const news = readJsonArray(DATA_FILES.news);
  const item = news.find((n) => Number(n.id) === id);
  if (!item) return res.status(404).json({ error: "Not found" });
  res.json(item);
});

app.post("/api/admin/news", (req, res) => {
  const { id, headline, description, category, date, level, video, learnSigns } =
    req.body || {};
  const numId = Number(id);
  if (!numId || !headline || !description) {
    return res
      .status(400)
      .json({ error: "id, headline, description are required" });
  }
  const news = readJsonArray(DATA_FILES.news);
  if (news.some((n) => Number(n.id) === numId)) {
    return res.status(400).json({ error: "ID already exists" });
  }
  news.push({
    id: numId,
    headline,
    description,
    category: category || "School",
    date: date || "Updated",
    level: level || "Easy",
    video: video || "",
    learnSigns: Array.isArray(learnSigns) ? learnSigns : [],
  });
  writeJsonArray(DATA_FILES.news, news);
  res.json({ ok: true });
});

app.put("/api/admin/news/:id", (req, res) => {
  const id = Number(req.params.id);
  const { headline, description, category, date, level, video, learnSigns } =
    req.body || {};
  if (!id || !headline || !description) {
    return res
      .status(400)
      .json({ error: "headline and description are required" });
  }
  const news = readJsonArray(DATA_FILES.news);
  const idx = news.findIndex((n) => Number(n.id) === id);
  if (idx === -1) return res.status(404).json({ error: "Not found" });
  news[idx] = {
    ...news[idx],
    headline,
    description,
    category: category || news[idx].category || "School",
    date: date || news[idx].date || "Updated",
    level: level || news[idx].level || "Easy",
    video: video || news[idx].video || "",
    learnSigns: Array.isArray(learnSigns)
      ? learnSigns
      : news[idx].learnSigns || [],
  };
  writeJsonArray(DATA_FILES.news, news);
  res.json({ ok: true });
});

app.delete("/api/admin/news/:id", (req, res) => {
  const id = Number(req.params.id);
  const news = readJsonArray(DATA_FILES.news);
  const next = news.filter((n) => Number(n.id) !== id);
  if (next.length === news.length) {
    return res.status(404).json({ error: "Not found" });
  }
  writeJsonArray(DATA_FILES.news, next);
  res.json({ ok: true });
});

// Public content feeds
app.get("/api/videos", (req, res) => {
  res.json(readJsonArray(DATA_FILES.videos));
});

app.get("/api/news", (req, res) => {
  res.json(readJsonArray(DATA_FILES.news));
});

// Register
app.post("/api/register", (req, res) => {
  const { name, email, password } = req.body;
  if (!name || !email || !password) {
    return res.status(400).json({ error: "All fields are required" });
  }

  const issues = passwordIssues(password);
  if (issues.length) {
    return res.status(400).json({
      error: `Password must include ${issues.join(", ")}.`,
    });
  }

  const hash = bcrypt.hashSync(password, 10);
  const stmt = db.prepare(
    "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)"
  );
  stmt.run(name, email, hash, function (err) {
    if (err) {
      if (err.message.includes("UNIQUE")) {
        return res.status(400).json({ error: "Email already registered" });
      }
      return res.status(500).json({ error: "Database error" });
    }
    res.json({ id: this.lastID, name, email });
  });
});

// Login
app.post("/api/login", (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ error: "Email and password required" });
  }

  db.get("SELECT * FROM users WHERE email = ?", [email], (err, user) => {
    if (err) return res.status(500).json({ error: "Database error" });
    if (!user) return res.status(401).json({ error: "Invalid credentials" });

    const ok = bcrypt.compareSync(password, user.password_hash);
    if (!ok) return res.status(401).json({ error: "Invalid credentials" });

    // In a real app you'd return a JWT; here we just send basic info
    res.json({
      id: user.id,
      name: user.name,
      email: user.email,
      role: user.role || "user",
    });
  });
});

// Get practice progress
app.get("/api/practice/:userId", (req, res) => {
  const userId = Number(req.params.userId);
  if (!userId) return res.status(400).json({ error: "Invalid user id" });
  db.get(
    "SELECT progress_json, updated_at FROM practice WHERE user_id = ?",
    [userId],
    (err, row) => {
      if (err) return res.status(500).json({ error: "Database error" });
      if (!row) return res.json({ progress: null });
      let progress = null;
      try {
        progress = JSON.parse(row.progress_json);
      } catch (e) {
        return res.status(500).json({ error: "Corrupt progress data" });
      }
      res.json({ progress, updated_at: row.updated_at });
    }
  );
});

// Save practice progress
app.post("/api/practice/:userId", (req, res) => {
  const userId = Number(req.params.userId);
  if (!userId) return res.status(400).json({ error: "Invalid user id" });
  const progress = req.body && req.body.progress;
  if (!progress) return res.status(400).json({ error: "Missing progress" });

  const progressJson = JSON.stringify(progress);
  db.run(
    `
    INSERT INTO practice (user_id, progress_json, updated_at)
    VALUES (?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(user_id) DO UPDATE SET
      progress_json = excluded.progress_json,
      updated_at = CURRENT_TIMESTAMP
    `,
    [userId, progressJson],
    function (err) {
      if (err) return res.status(500).json({ error: "Database error" });
      res.json({ ok: true });
    }
  );
});

// Explicit page routes keep navigation stable even if a URL is typed with minor variants.
app.get(["/detect", "/detect.html", "/detect.html."], (req, res) => {
  res.sendFile(path.join(__dirname, "translator", "detect.html"));
});

app.listen(PORT, () => {
  console.log(`Auth API running on http://localhost:${PORT}`);
});

