from flask import Flask, render_template, request, redirect, url_for
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_FILE = os.path.join(BASE_DIR, "data", "news.json")
VIDEOS_FILE = os.path.join(BASE_DIR, "data", "videos.json")


def load_json(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_new_id(items):
    if not items:
        return 1
    return max(item["id"] for item in items) + 1


@app.route("/")
def home():
    news = load_json(NEWS_FILE)
    videos = load_json(VIDEOS_FILE)
    return render_template("home.html", news=news, videos=videos)


@app.route("/videos")
def videos():
    videos_data = load_json(VIDEOS_FILE)
    return render_template("videos.html", videos=videos_data)


@app.route("/admin")
def admin_panel():
    news = load_json(NEWS_FILE)
    videos = load_json(VIDEOS_FILE)
    return render_template("admin_panel.html", news=news, videos=videos)


@app.route("/admin/add_news", methods=["POST"])
def add_news():
    news = load_json(NEWS_FILE)

    title = request.form["title"]
    desc = request.form["desc"]
    date = request.form["date"]

    news.append({
        "id": get_new_id(news),
        "title": title,
        "desc": desc,
        "date": date
    })

    save_json(NEWS_FILE, news)
    return redirect(url_for("admin_panel"))


@app.route("/admin/delete_news/<int:news_id>")
def delete_news(news_id):
    news = load_json(NEWS_FILE)
    news = [n for n in news if n["id"] != news_id]
    save_json(NEWS_FILE, news)
    return redirect(url_for("admin_panel"))


@app.route("/admin/edit_news/<int:news_id>", methods=["GET", "POST"])
def edit_news(news_id):
    news = load_json(NEWS_FILE)
    item = next((n for n in news if n["id"] == news_id), None)

    if request.method == "POST":
        item["title"] = request.form["title"]
        item["desc"] = request.form["desc"]
        item["date"] = request.form["date"]
        save_json(NEWS_FILE, news)
        return redirect(url_for("admin_panel"))

    return render_template("edit_news.html", item=item)


from flask import session

ADMIN_PASSWORD = "1234"   # change this

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    error = None

    if request.method == "POST":
        password = request.form["password"]

        if password == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin_panel"))
        else:
            error = "Wrong password ❌ Try again"

    return render_template("admin_login.html", error=error)
