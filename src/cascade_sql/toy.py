"""Toy benchmark for the demo and tests: two small SQLite databases and ~600
template questions, each with a gold SQL and plausible wrong SQLs (used by the Mock
model). Test questions get two label versions: `orig` (with ~10% deliberately wrong
gold SQL, imitating benchmark annotation errors) and `corrected`.
"""
import os
import random
import sqlite3
from typing import Dict, List

from .io import write_jsonl

FIRST = ["Asha", "Ravi", "Meera", "Kiran", "Arjun", "Priya", "Vikram", "Neha", "Rohan", "Isha",
         "Dev", "Anika", "Kabir", "Tara", "Nikhil", "Sara", "Omar", "Lena", "Yusuf", "Mira"]
CITIES = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai"]
CATS = ["books", "electronics", "toys", "grocery", "sports"]
DEPTS = ["CS", "Math", "Physics", "Biology"]


def build_shop(path: str, seed: int = 0):
    rng = random.Random(seed)
    if os.path.exists(path):
        os.remove(path)
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
    CREATE TABLE products (id INTEGER PRIMARY KEY, title TEXT, category TEXT, price REAL);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
                         qty INTEGER, year INTEGER,
                         FOREIGN KEY (customer_id) REFERENCES customers(id),
                         FOREIGN KEY (product_id) REFERENCES products(id));""")
    for i, n in enumerate(FIRST, 1):
        c.execute("INSERT INTO customers VALUES (?,?,?)", (i, n, rng.choice(CITIES)))
    for i in range(1, 41):
        c.execute("INSERT INTO products VALUES (?,?,?,?)",
                  (i, f"item{i}", rng.choice(CATS), round(rng.uniform(5, 500), 2)))
    for i in range(1, 601):
        c.execute("INSERT INTO orders VALUES (?,?,?,?,?)",
                  (i, rng.randint(1, len(FIRST)), rng.randint(1, 40), rng.randint(1, 5),
                   rng.choice([2023, 2024, 2025])))
    c.commit()
    c.close()


def build_school(path: str, seed: int = 1):
    rng = random.Random(seed)
    if os.path.exists(path):
        os.remove(path)
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, year INTEGER);
    CREATE TABLE courses (id INTEGER PRIMARY KEY, code TEXT, dept TEXT, credits INTEGER);
    CREATE TABLE grades (student_id INTEGER, course_id INTEGER, score REAL,
                         FOREIGN KEY (student_id) REFERENCES students(id),
                         FOREIGN KEY (course_id) REFERENCES courses(id));""")
    for i in range(1, 61):
        c.execute("INSERT INTO students VALUES (?,?,?,?)",
                  (i, f"{rng.choice(FIRST)}{i}", rng.choice(DEPTS), rng.randint(1, 4)))
    for i in range(1, 21):
        c.execute("INSERT INTO courses VALUES (?,?,?,?)", (i, f"C{100 + i}", rng.choice(DEPTS), rng.choice([2, 3, 4])))
    for s in range(1, 61):
        for cid in rng.sample(range(1, 21), 5):
            c.execute("INSERT INTO grades VALUES (?,?,?)", (s, cid, round(rng.uniform(35, 100), 1)))
    c.commit()
    c.close()


def _shop_questions(rng) -> List[Dict]:
    qs = []
    for n in FIRST:
        qs.append(dict(q=f"How many orders did {n} place?", d=0.15,
                       g=f"SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.name = '{n}'",
                       w=[f"SELECT COUNT(*) FROM customers WHERE name = '{n}'",
                          f"SELECT SUM(o.qty) FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.name = '{n}'"]))
        qs.append(dict(q=f"What is the total amount {n} spent in 2024?", d=0.55,
                       g=f"SELECT SUM(o.qty * p.price) FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE c.name = '{n}' AND o.year = 2024",
                       w=[f"SELECT SUM(p.price) FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE c.name = '{n}' AND o.year = 2024",
                          f"SELECT SUM(o.qty * p.price) FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE c.name = '{n}'"]))
    for city in CITIES:
        qs.append(dict(q=f"List the names of customers who live in {city}.", d=0.05,
                       g=f"SELECT name FROM customers WHERE city = '{city}'",
                       w=[f"SELECT name, city FROM customers WHERE city = '{city}'",
                          f"SELECT name FROM customers WHERE city <> '{city}'"]))
        for y in (2023, 2024, 2025):
            qs.append(dict(q=f"Which product category sold the most units to customers in {city} in {y}?", d=0.8,
                           g=f"SELECT p.category FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE c.city = '{city}' AND o.year = {y} GROUP BY p.category ORDER BY SUM(o.qty) DESC LIMIT 1",
                           w=[f"SELECT p.category FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE c.city = '{city}' AND o.year = {y} GROUP BY p.category ORDER BY COUNT(*) DESC LIMIT 1",
                              f"SELECT p.category FROM orders o JOIN products p ON o.product_id = p.id WHERE o.year = {y} GROUP BY p.category ORDER BY SUM(o.qty) DESC LIMIT 1",
                              f"SELECT p.category FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE c.city = '{city}' GROUP BY p.category ORDER BY SUM(o.qty) DESC LIMIT 1"]))
    for cat in CATS:
        for th in (50, 100, 200, 300):
            qs.append(dict(q=f"How many {cat} products cost more than {th}?", d=0.2,
                           g=f"SELECT COUNT(*) FROM products WHERE category = '{cat}' AND price > {th}",
                           w=[f"SELECT COUNT(*) FROM products WHERE category = '{cat}' AND price >= {th}",
                              f"SELECT COUNT(*) FROM products WHERE price > {th}"]))
        qs.append(dict(q=f"What is the average price of {cat} products?", d=0.1,
                       g=f"SELECT AVG(price) FROM products WHERE category = '{cat}'",
                       w=[f"SELECT AVG(price) FROM products", f"SELECT SUM(price) FROM products WHERE category = '{cat}'"]))
    return qs


def _school_questions(rng) -> List[Dict]:
    qs = []
    for d in DEPTS:
        for y in (1, 2, 3, 4):
            qs.append(dict(q=f"How many year {y} students are in the {d} department?", d=0.1,
                           g=f"SELECT COUNT(*) FROM students WHERE dept = '{d}' AND year = {y}",
                           w=[f"SELECT COUNT(*) FROM students WHERE dept = '{d}'",
                              f"SELECT COUNT(*) FROM students WHERE year = {y}"]))
            qs.append(dict(q=f"What is the average score of year {y} {d} students in {d} courses?", d=0.7,
                           g=f"SELECT AVG(g.score) FROM grades g JOIN students s ON g.student_id = s.id JOIN courses c ON g.course_id = c.id WHERE s.dept = '{d}' AND s.year = {y} AND c.dept = '{d}'",
                           w=[f"SELECT AVG(g.score) FROM grades g JOIN students s ON g.student_id = s.id WHERE s.dept = '{d}' AND s.year = {y}",
                              f"SELECT AVG(g.score) FROM grades g JOIN students s ON g.student_id = s.id JOIN courses c ON g.course_id = c.id WHERE s.dept = '{d}' AND c.dept = '{d}'"]))
        for th in (50, 60, 70, 80, 90):
            qs.append(dict(q=f"How many {d} students have at least one score above {th}?", d=0.5,
                           g=f"SELECT COUNT(DISTINCT s.id) FROM students s JOIN grades g ON g.student_id = s.id WHERE s.dept = '{d}' AND g.score > {th}",
                           w=[f"SELECT COUNT(*) FROM students s JOIN grades g ON g.student_id = s.id WHERE s.dept = '{d}' AND g.score > {th}",
                              f"SELECT COUNT(DISTINCT s.id) FROM students s JOIN grades g ON g.student_id = s.id WHERE g.score > {th}"]))
        qs.append(dict(q=f"Which {d} course has the highest average score?", d=0.85,
                       g=f"SELECT c.code FROM courses c JOIN grades g ON g.course_id = c.id WHERE c.dept = '{d}' GROUP BY c.id ORDER BY AVG(g.score) DESC LIMIT 1",
                       w=[f"SELECT c.code FROM courses c JOIN grades g ON g.course_id = c.id WHERE c.dept = '{d}' GROUP BY c.id ORDER BY MAX(g.score) DESC LIMIT 1",
                          f"SELECT c.code FROM courses c JOIN grades g ON g.course_id = c.id GROUP BY c.id ORDER BY AVG(g.score) DESC LIMIT 1"]))
    return qs


def build(out_dir: str, seed: int = 0, label_error_rate: float = 0.10) -> Dict[str, int]:
    """Create toy DBs + manifests g_train / calib / test_b in out_dir."""
    rng = random.Random(seed)
    db_dir = os.path.join(out_dir, "databases")
    os.makedirs(db_dir, exist_ok=True)
    dbs = {"shop": os.path.join(db_dir, "shop.sqlite"), "school": os.path.join(db_dir, "school.sqlite")}
    build_shop(dbs["shop"], seed)
    build_school(dbs["school"], seed + 1)
    pool = [("shop", q) for q in _shop_questions(rng)] + [("school", q) for q in _school_questions(rng)]
    # paraphrase-replicate to get enough questions; difficulty jittered per copy
    items = []
    wraps = ["{q}", "Please tell me: {q}", "{q} Answer from the database.", "Quick question. {q}",
             "Using the data, {q}", "I need to know: {q}", "{q} Thanks!", "Could you check: {q}"]
    for rep, wrap in enumerate(wraps):
        for k, (db, q) in enumerate(pool):
            qtext = wrap.format(q=q["q"])
            diff = min(0.95, max(0.0, q["d"] + rng.gauss(0, 0.12)))
            items.append({"qid": f"toy_{rep}_{k}", "db_id": db, "db_path": dbs[db], "question": qtext,
                          "evidence": None, "difficulty": diff, "gold": {"orig": q["g"]},
                          "mock_meta": {"qid": f"toy_{rep}_{k}", "gold_sql": q["g"], "wrong_sqls": q["w"],
                                        "difficulty": diff, "some_table": "orders" if db == "shop" else "grades"}})
    rng.shuffle(items)
    n = len(items)
    g_train, calib, test = items[: int(0.45 * n)], items[int(0.45 * n): int(0.7 * n)], items[int(0.7 * n):]
    for it in g_train:
        it["split"] = "g_train"
    for it in calib:
        it["split"] = "calib"
    for it in test:
        it["split"] = "test_b"
        it["gold"] = {"orig": it["gold"]["orig"], "corrected": it["gold"]["orig"]}
        if rng.random() < label_error_rate:  # simulated annotation error in the original labels
            it["gold"]["orig"] = rng.choice(it["mock_meta"]["wrong_sqls"])
    write_jsonl(os.path.join(out_dir, "g_train.jsonl"), g_train)
    write_jsonl(os.path.join(out_dir, "calib.jsonl"), calib)
    write_jsonl(os.path.join(out_dir, "test_b.jsonl"), test)
    return {"g_train": len(g_train), "calib": len(calib), "test_b": len(test)}
