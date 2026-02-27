import os
import json as _json
import time
import requests
from datetime import datetime, timezone
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "DrivetrainAi/tesseract")
WORKFLOW_FILE = os.environ.get("WORKFLOW_FILE", "regression-v2-tests.yml")
LD_API_KEY = os.environ.get("LD_API_KEY", "")
LD_PROJECT_KEY = os.environ.get("LD_PROJECT_KEY", "default")
APP_USERNAME = os.environ.get("APP_USERNAME", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gandalf.db")

# In-memory branch cache — refreshed at most once every 5 minutes
_branch_cache: dict = {"branches": [], "fetched_at": 0}
_BRANCH_CACHE_TTL = 300  # seconds

# ---------------------------------------------------------------------------
# Unified DB adapter — switches between PostgreSQL and SQLite automatically.
# When DATABASE_URL is set (e.g. on Render) Postgres is used; otherwise the
# local SQLite file is used so local development requires no extra setup.
# ---------------------------------------------------------------------------
if DATABASE_URL:
    import psycopg2
    import psycopg2.extras
    import psycopg2.errors

    def get_db():
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        return conn

    def db_cursor(conn):
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    P = "%s"  # query placeholder
    UniqueViolation = psycopg2.errors.UniqueViolation

    def _in_clause(ids):
        """Return (sql_fragment, params) for an IN query with a list of ids."""
        return "= ANY(%s)", (list(ids),)

    def _insert_ignore(table, cols, vals, conflict_col):
        """INSERT … ON CONFLICT DO NOTHING."""
        placeholders = ", ".join(["%s"] * len(cols))
        col_str = ", ".join(cols)
        return (
            f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
            vals,
        )

    def _upsert_run_name(conn):
        cur = db_cursor(conn)
        def upsert(run_id, run_name):
            cur.execute(
                "INSERT INTO run_names (github_run_id, run_name, saved_at) VALUES (%s, %s, NOW()) "
                "ON CONFLICT (github_run_id) DO UPDATE SET run_name = EXCLUDED.run_name, saved_at = NOW()",
                (run_id, run_name),
            )
        return upsert

    def executemany_insert_run_names(conn, mappings):
        if not mappings:
            return
        cur = db_cursor(conn)
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO run_names (github_run_id, run_name, saved_at) VALUES (%s, %s, NOW()) "
            "ON CONFLICT DO NOTHING",
            mappings,
        )

else:
    import sqlite3

    def get_db():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def db_cursor(conn):
        return conn.cursor()

    P = "?"
    UniqueViolation = sqlite3.IntegrityError

    def _in_clause(ids):
        placeholders = ", ".join(["?"] * len(ids))
        return f"IN ({placeholders})", list(ids)

    def _insert_ignore(table, cols, vals, conflict_col):
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        return (
            f"INSERT OR IGNORE INTO {table} ({col_str}) VALUES ({placeholders})",
            vals,
        )

    def _upsert_run_name(conn):
        def upsert(run_id, run_name):
            conn.execute(
                "INSERT OR REPLACE INTO run_names (github_run_id, run_name, saved_at) "
                "VALUES (?, ?, datetime('now'))",
                (run_id, run_name),
            )
        return upsert

    def executemany_insert_run_names(conn, mappings):
        if not mappings:
            return
        conn.executemany(
            "INSERT OR IGNORE INTO run_names (github_run_id, run_name) VALUES (?, ?)",
            mappings,
        )


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.is_json:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

WORKFLOW_INPUTS = [
    {
        "name": "runId", "label": "Run Name", "default": "", "required": True, "type": "text",
        "hint": "No spaces, commas, or quotes",
        "pattern": r"^[^\s,\'\"]+$",
        "pattern_error": "Must not contain spaces, commas, or quotes",
    },
    {
        "name": "maxEcsTasks", "label": "No. of ECS Workers for Parallel Run", "default": "1", "required": True, "type": "number",
        "hint": "Each worker runs tests independently in parallel. More workers = faster completion. Recommended: 5 for standard runs, 10–20 for large tenant sets.",
        "min": "1", "max": "20",
        "pattern": r"^(1[0-9]?|[2-9]|20)$",
        "pattern_error": "Must be a whole number between 1 and 20",
    },
    {
        "name": "tenants", "label": "Tenants List", "default": "*", "required": True, "type": "text",
        "hint": "Use * for all, or comma-separated IDs (e.g. 585,601). No spaces or quotes",
        "pattern": r"^[^\s\'\"]+$",
        "pattern_error": "Must not contain spaces or quotes",
    },
    {
        "name": "mainBranch", "label": "Main Branch", "default": "main-v3", "required": True, "type": "text",
        "hint": "No spaces, commas, or quotes (e.g. main-v3)",
        "pattern": r"^[^\s,\'\"]+$",
        "pattern_error": "Must not contain spaces, commas, or quotes",
    },
    {
        "name": "featBranch", "label": "Feature Branch", "default": "preprod-v3", "required": True, "type": "text",
        "hint": "No spaces, commas, or quotes (e.g. ENG-71221)",
        "pattern": r"^[^\s,\'\"]+$",
        "pattern_error": "Must not contain spaces, commas, or quotes",
    },
    {
        "name": "compareModels", "label": "Compare Models", "default": "true", "required": False,
        "type": "boolean",
        "hint": "Run regression comparison on models",
    },
    {
        "name": "compareReports", "label": "Compare Reports", "default": "true", "required": False,
        "type": "boolean",
        "hint": "Run regression comparison on reports",
    },
    {
        "name": "filters", "label": "Filters", "default": "", "required": False, "type": "text",
        "hint": "Optional — specific model or report name to filter",
    },
    {
        "name": "ldFlagsMainBranch", "label": "LD Flags (Main Branch)", "default": "", "required": False, "type": "text",
        "hint": "Optional — LaunchDarkly flags for the main branch",
    },
    {
        "name": "ldFlagsFeatBranch", "label": "LD Flags (Feature Branch)", "default": "", "required": False, "type": "text",
        "hint": "Optional — LaunchDarkly flags for the feature branch",
    },
]


def init_db():
    conn = get_db()
    cur = db_cursor(conn)
    if DATABASE_URL:
        statements = [
            """CREATE TABLE IF NOT EXISTS groups (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS group_runs (
                group_id INTEGER NOT NULL,
                run_id BIGINT NOT NULL,
                added_at TIMESTAMP NOT NULL DEFAULT NOW(),
                PRIMARY KEY (group_id, run_id),
                FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
            )""",
            """CREATE TABLE IF NOT EXISTS triggered_runs (
                id SERIAL PRIMARY KEY,
                run_id BIGINT,
                run_id_input TEXT,
                inputs_json TEXT,
                triggered_at TIMESTAMP NOT NULL DEFAULT NOW(),
                status TEXT DEFAULT 'dispatched'
            )""",
            """CREATE TABLE IF NOT EXISTS run_names (
                github_run_id BIGINT PRIMARY KEY,
                run_name TEXT NOT NULL,
                saved_at TIMESTAMP NOT NULL DEFAULT NOW()
            )""",
        ]
        for stmt in statements:
            cur.execute(stmt)
    else:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS group_runs (
                group_id INTEGER NOT NULL,
                run_id INTEGER NOT NULL,
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (group_id, run_id),
                FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS triggered_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                run_id_input TEXT,
                inputs_json TEXT,
                triggered_at TEXT NOT NULL DEFAULT (datetime('now')),
                status TEXT DEFAULT 'dispatched'
            );
            CREATE TABLE IF NOT EXISTS run_names (
                github_run_id INTEGER PRIMARY KEY,
                run_name TEXT NOT NULL,
                saved_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
    conn.commit()
    conn.close()


def github_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }


# --- Auth ---

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == APP_USERNAME and password == APP_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# --- Pages ---

@app.route("/")
@login_required
def index():
    return render_template("index.html", workflow_inputs=WORKFLOW_INPUTS, github_repo=GITHUB_REPO)


# --- API: Trigger ---

@app.route("/api/trigger", methods=["POST"])
@login_required
def trigger_workflow():
    if not GITHUB_TOKEN:
        return jsonify({"error": "GITHUB_TOKEN not configured"}), 500

    data = request.json or {}
    inputs = {}
    for inp in WORKFLOW_INPUTS:
        val = data.get(inp["name"], inp["default"])
        if inp["required"] and not val:
            return jsonify({"error": f"'{inp['label']}' is required"}), 400
        inputs[inp["name"]] = val

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    payload = {"ref": data.get("ref", "main-v3"), "inputs": inputs}

    resp = requests.post(url, json=payload, headers=github_headers(), timeout=30)

    if resp.status_code == 204:
        conn = get_db()
        cur = db_cursor(conn)
        cur.execute(
            f"INSERT INTO triggered_runs (run_id_input, inputs_json) VALUES ({P}, {P})",
            (inputs.get("runId", ""), _json.dumps(inputs)),
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Workflow dispatched successfully"})
    else:
        error_body = resp.text
        try:
            error_body = resp.json().get("message", resp.text)
        except Exception:
            pass
        return jsonify({"error": f"GitHub API error ({resp.status_code}): {error_body}"}), resp.status_code


# --- API: Runs ---

@app.route("/api/runs")
@login_required
def list_runs():
    if not GITHUB_TOKEN:
        return jsonify({"error": "GITHUB_TOKEN not configured"}), 500

    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/runs"
    params = {"page": page, "per_page": per_page}
    resp = requests.get(url, params=params, headers=github_headers(), timeout=30)

    if resp.status_code != 200:
        return jsonify({"error": f"GitHub API error ({resp.status_code})"}), resp.status_code

    data = resp.json()
    runs = []
    for run in data.get("workflow_runs", []):
        runs.append({
            "id": run["id"],
            "name": run.get("display_title", run.get("name", "")),
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "html_url": run["html_url"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "run_number": run["run_number"],
            "head_branch": run.get("head_branch", ""),
            "actor": run.get("actor", {}).get("login", ""),
            "run_name": "",
        })

    # Look up run names from local DB (saved at trigger time)
    conn = get_db()
    cur = db_cursor(conn)
    run_ids = [r["id"] for r in runs]
    if run_ids:
        in_sql, in_params = _in_clause(run_ids)
        cur.execute(
            f"SELECT github_run_id, run_name FROM run_names WHERE github_run_id {in_sql}",
            in_params,
        )
        name_rows = cur.fetchall()
        run_name_map = {row["github_run_id"]: row["run_name"] for row in name_rows}
    else:
        run_name_map = {}

    for run in runs:
        run["run_name"] = run_name_map.get(run["id"], "")

    # Timestamp-based fallback: match unresolved GitHub runs to triggered_runs records
    # by finding the closest triggered_at within a 2-minute window of created_at.
    unmatched = [r for r in runs if not run_name_map.get(r["id"])]
    if unmatched:
        cur.execute(
            "SELECT id, run_id_input, triggered_at FROM triggered_runs "
            "WHERE run_id_input IS NOT NULL AND run_id_input != '' "
            "ORDER BY triggered_at DESC"
        )
        triggered_rows = cur.fetchall()

        used_triggered_ids = set()
        new_mappings = []

        for run in sorted(unmatched, key=lambda r: r["created_at"]):
            gh_time = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
            best_tr, best_diff = None, None
            for tr in triggered_rows:
                if tr["id"] in used_triggered_ids:
                    continue
                tr_at = tr["triggered_at"]
                if isinstance(tr_at, str):
                    tr_at = tr_at.replace(" ", "T") + "+00:00"
                    tr_time = datetime.fromisoformat(tr_at)
                else:
                    tr_time = tr_at.replace(tzinfo=timezone.utc) if tr_at.tzinfo is None else tr_at
                diff = abs((gh_time - tr_time).total_seconds())
                if diff <= 120 and (best_diff is None or diff < best_diff):
                    best_tr, best_diff = tr, diff

            if best_tr:
                used_triggered_ids.add(best_tr["id"])
                run_name_map[run["id"]] = best_tr["run_id_input"]
                run["run_name"] = best_tr["run_id_input"]
                new_mappings.append((run["id"], best_tr["run_id_input"]))

        executemany_insert_run_names(conn, new_mappings)
        if new_mappings:
            conn.commit()

    cur.execute(
        "SELECT gr.run_id, g.id as group_id, g.name as group_name "
        "FROM group_runs gr JOIN groups g ON gr.group_id = g.id"
    )
    all_group_runs = cur.fetchall()

    # Build tenants map: run_name -> tenants value from inputs_json
    cur.execute(
        "SELECT run_id_input, inputs_json FROM triggered_runs "
        "WHERE run_id_input IS NOT NULL AND run_id_input != '' AND inputs_json IS NOT NULL"
    )
    triggered_inputs_rows = cur.fetchall()
    tenants_by_run_name = {}
    for row in triggered_inputs_rows:
        try:
            inp = _json.loads(row["inputs_json"])
            tenants_val = inp.get("tenants", "")
            if tenants_val and row["run_id_input"]:
                tenants_by_run_name[row["run_id_input"]] = tenants_val
        except Exception:
            pass

    conn.close()

    run_groups_map = {}
    for row in all_group_runs:
        rid = row["run_id"]
        if rid not in run_groups_map:
            run_groups_map[rid] = []
        run_groups_map[rid].append({"id": row["group_id"], "name": row["group_name"]})

    for run in runs:
        run["groups"] = run_groups_map.get(run["id"], [])
        run["tenants"] = tenants_by_run_name.get(run["run_name"], "")

    return jsonify({
        "runs": runs,
        "total_count": data.get("total_count", 0),
        "page": page,
        "per_page": per_page,
    })


@app.route("/api/branches")
@login_required
def list_branches():
    now = time.time()
    if _branch_cache["branches"] and now - _branch_cache["fetched_at"] < _BRANCH_CACHE_TTL:
        return jsonify({"branches": _branch_cache["branches"], "cached": True})

    branches = []
    page = 1
    while len(branches) < 500:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/branches",
            params={"per_page": 100, "page": page},
            headers=github_headers(),
            timeout=15,
        )
        if resp.status_code != 200:
            if branches:
                return jsonify({"branches": branches})
            return jsonify({"error": f"GitHub returned {resp.status_code}"}), resp.status_code
        data = resp.json()
        if not data:
            break
        branches.extend(b["name"] for b in data)
        if len(data) < 100:
            break
        page += 1

    _branch_cache["branches"] = branches
    _branch_cache["fetched_at"] = time.time()
    return jsonify({"branches": branches})


@app.route("/api/runs/<int:run_id>/cancel", methods=["POST"])
@login_required
def cancel_run(run_id):
    resp = requests.post(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}/cancel",
        headers=github_headers(),
        timeout=15,
    )
    if resp.status_code == 202:
        return jsonify({"success": True})
    return jsonify({"error": f"GitHub returned {resp.status_code}"}), resp.status_code


@app.route("/api/runs/<int:run_id>/name", methods=["POST"])
@login_required
def save_run_name(run_id):
    data = request.json or {}
    run_name = (data.get("run_name") or "").strip()
    if not run_name:
        return jsonify({"error": "run_name is required"}), 400
    conn = get_db()
    upsert = _upsert_run_name(conn)
    upsert(run_id, run_name)
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# --- API: LaunchDarkly ---

@app.route("/api/ld/flags")
@login_required
def search_ld_flags():
    if not LD_API_KEY:
        return jsonify({"error": "LD_API_KEY not configured"}), 500

    query = request.args.get("q", "").strip()
    limit = min(request.args.get("limit", 30, type=int), 100)

    url = f"https://app.launchdarkly.com/api/v2/flags/{LD_PROJECT_KEY}"
    params = {"limit": limit, "summary": "true"}
    if query:
        params["filter"] = f"query:{query}"

    resp = requests.get(url, params=params, headers={
        "Authorization": LD_API_KEY,
    }, timeout=15)

    if resp.status_code != 200:
        try:
            msg = resp.json().get("message", resp.text)
        except Exception:
            msg = resp.text
        return jsonify({"error": f"LaunchDarkly API error ({resp.status_code}): {msg}"}), resp.status_code

    data = resp.json()
    flags = []
    for item in data.get("items", []):
        variations = item.get("variations", [])
        flags.append({
            "key": item["key"],
            "name": item.get("name", item["key"]),
            "description": item.get("description", ""),
            "kind": item.get("kind", "boolean"),
            "variations": [{"value": str(v.get("value", "")).lower(), "name": v.get("name", str(v.get("value", "")))} for v in variations],
            "tags": item.get("tags", []),
        })

    return jsonify({"flags": flags, "total": data.get("totalCount", len(flags))})


# --- API: Groups ---

@app.route("/api/groups", methods=["GET"])
@login_required
def list_groups():
    conn = get_db()
    cur = db_cursor(conn)
    cur.execute("SELECT * FROM groups ORDER BY created_at DESC")
    groups = cur.fetchall()

    # Collect all unique run_ids across all groups
    raw_groups = []
    all_run_ids = set()
    for g in groups:
        cur.execute(
            f"SELECT run_id, added_at FROM group_runs WHERE group_id = {P} ORDER BY added_at DESC",
            (g["id"],),
        )
        run_rows = cur.fetchall()
        runs = [{"run_id": r["run_id"], "added_at": r["added_at"]} for r in run_rows]
        raw_groups.append({"id": g["id"], "name": g["name"], "created_at": str(g["created_at"]), "runs": runs})
        all_run_ids.update(r["run_id"] for r in run_rows)

    # Local DB lookup for run_names
    run_name_map = {}
    if all_run_ids:
        in_sql, in_params = _in_clause(all_run_ids)
        cur.execute(
            f"SELECT github_run_id, run_name FROM run_names WHERE github_run_id {in_sql}",
            in_params,
        )
        name_rows = cur.fetchall()
        run_name_map = {r["github_run_id"]: r["run_name"] for r in name_rows}
    conn.close()

    # Parallel GitHub fetch for status / branch / duration
    def _fetch_gh(run_id):
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}",
                headers=github_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                d = resp.json()
                duration = None
                try:
                    c = datetime.fromisoformat(d["created_at"].replace("Z", "+00:00"))
                    u = datetime.fromisoformat(d["updated_at"].replace("Z", "+00:00"))
                    duration = max(0, int((u - c).total_seconds()))
                except Exception:
                    pass
                return run_id, {
                    "status":      d.get("status"),
                    "conclusion":  d.get("conclusion"),
                    "head_branch": d.get("head_branch"),
                    "created_at":  d.get("created_at"),
                    "actor":       (d.get("actor") or {}).get("login", ""),
                    "duration":    duration,
                }
        except Exception:
            pass
        return run_id, {}

    gh_details = {}
    if all_run_ids:
        with ThreadPoolExecutor(max_workers=min(12, len(all_run_ids))) as pool:
            futures = {pool.submit(_fetch_gh, rid): rid for rid in all_run_ids}
            for future in as_completed(futures):
                run_id, details = future.result()
                if details:
                    gh_details[run_id] = details

    # Build enriched response
    result = []
    for g in raw_groups:
        enriched = []
        for r in g["runs"]:
            rid = r["run_id"]
            gh = gh_details.get(rid, {})
            enriched.append({
                "run_id":      rid,
                "added_at":    r["added_at"],
                "run_name":    run_name_map.get(rid, ""),
                "status":      gh.get("status"),
                "conclusion":  gh.get("conclusion"),
                "head_branch": gh.get("head_branch"),
                "created_at":  gh.get("created_at"),
                "actor":       gh.get("actor"),
                "duration":    gh.get("duration"),
            })
        result.append({"id": g["id"], "name": g["name"], "created_at": g["created_at"], "runs": enriched})

    return jsonify({"groups": result})


@app.route("/api/groups", methods=["POST"])
@login_required
def create_group():
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Group name is required"}), 400

    conn = get_db()
    cur = db_cursor(conn)
    try:
        cur.execute(f"INSERT INTO groups (name) VALUES ({P})", (name,))
        conn.commit()
    except UniqueViolation:
        conn.close()
        return jsonify({"error": f"Group '{name}' already exists"}), 409
    cur.execute(f"SELECT * FROM groups WHERE name = {P}", (name,))
    group = cur.fetchone()
    conn.close()
    return jsonify({"id": group["id"], "name": group["name"], "created_at": str(group["created_at"])}), 201


@app.route("/api/groups/<int:group_id>", methods=["DELETE"])
@login_required
def delete_group(group_id):
    conn = get_db()
    cur = db_cursor(conn)
    cur.execute(f"SELECT * FROM groups WHERE id = {P}", (group_id,))
    group = cur.fetchone()
    if not group:
        conn.close()
        return jsonify({"error": "Group not found"}), 404
    cur.execute(f"DELETE FROM groups WHERE id = {P}", (group_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/runs/search")
@login_required
def search_runs_by_name():
    """Search run_names by partial run name, return matches with GitHub run ID."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    conn = get_db()
    cur = db_cursor(conn)
    cur.execute(
        f"SELECT github_run_id, run_name FROM run_names "
        f"WHERE run_name LIKE {P} ORDER BY saved_at DESC LIMIT 20",
        (f"%{q}%",),
    )
    rows = cur.fetchall()
    conn.close()
    return jsonify([{"run_id": r["github_run_id"], "run_name": r["run_name"]} for r in rows])


@app.route("/api/groups/<int:group_id>/runs", methods=["POST"])
@login_required
def add_run_to_group(group_id):
    conn = get_db()
    cur = db_cursor(conn)
    cur.execute(f"SELECT * FROM groups WHERE id = {P}", (group_id,))
    group = cur.fetchone()
    if not group:
        conn.close()
        return jsonify({"error": "Group not found"}), 404

    data = request.json or {}
    run_id = data.get("run_id")
    if not run_id:
        conn.close()
        return jsonify({"error": "run_id is required"}), 400

    try:
        run_id = int(run_id)
    except (ValueError, TypeError):
        conn.close()
        return jsonify({"error": "run_id must be a number"}), 400

    try:
        cur.execute(
            f"INSERT INTO group_runs (group_id, run_id) VALUES ({P}, {P})",
            (group_id, run_id),
        )
        conn.commit()
    except UniqueViolation:
        conn.close()
        return jsonify({"error": "Run is already in this group"}), 409
    conn.close()
    return jsonify({"success": True}), 201


@app.route("/api/groups/<int:group_id>/runs/<int:run_id>", methods=["DELETE"])
@login_required
def remove_run_from_group(group_id, run_id):
    conn = get_db()
    cur = db_cursor(conn)
    cur.execute(
        f"DELETE FROM group_runs WHERE group_id = {P} AND run_id = {P}",
        (group_id, run_id),
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    if deleted == 0:
        return jsonify({"error": "Run not found in this group"}), 404
    return jsonify({"success": True})


@app.route("/api/groups/<int:group_id>/runs/batch", methods=["POST"])
@login_required
def batch_add_runs_to_group(group_id):
    """Add multiple runs to a group at once."""
    conn = get_db()
    cur = db_cursor(conn)
    cur.execute(f"SELECT * FROM groups WHERE id = {P}", (group_id,))
    group = cur.fetchone()
    if not group:
        conn.close()
        return jsonify({"error": "Group not found"}), 404

    data = request.json or {}
    run_ids = data.get("run_ids", [])
    if not run_ids:
        conn.close()
        return jsonify({"error": "run_ids array is required"}), 400

    added = 0
    skipped = 0
    for rid in run_ids:
        try:
            rid = int(rid)
            cur.execute(
                f"INSERT INTO group_runs (group_id, run_id) VALUES ({P}, {P})",
                (group_id, rid),
            )
            added += 1
        except (UniqueViolation, ValueError, TypeError):
            if DATABASE_URL:
                conn.rollback()
            skipped += 1
    conn.commit()
    conn.close()
    return jsonify({"added": added, "skipped": skipped}), 201


def _init_db_with_retry(retries=5, delay=3):
    """Try to initialise the DB, retrying on transient connection errors."""
    for attempt in range(1, retries + 1):
        try:
            init_db()
            return
        except Exception as exc:
            print(f"[gandalf] init_db attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(delay)
    print("[gandalf] WARNING: could not initialise DB after all retries — continuing anyway")

_init_db_with_retry()

if __name__ == "__main__":
    app.run(debug=True, port=5555)
