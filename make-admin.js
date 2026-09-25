const sqlite3 = require("sqlite3").verbose();

const email = process.argv[2];
if (!email) {
  console.error("Usage: node make-admin.js someone@email.com");
  process.exit(1);
}

const db = new sqlite3.Database("users.db");
db.run(
  "UPDATE users SET role='admin' WHERE email=?",
  [email],
  function (err) {
    if (err) {
      console.error(err);
    } else {
      console.log("Updated", this.changes);
    }
    db.close();
  }
);
