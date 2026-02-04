#!/usr/bin/env python3
import csv
import os
import random
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from decimal import Decimal

# ----------------------------
# Fixed dimensions (DO NOT expand)
# ----------------------------
COUNTRY_CITY = [
    ("Poland", "Warsaw"),
    ("United States", "New York"),
    ("Spain", "Madrid"),
    ("Italy", "Milan"),
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("United Kingdom", "London"),
    ("Portugal", "Lisbon"),
    ("Japan", "Tokyo"),
    ("China", "Shanghai"),
    ("India", "Bengaluru"),
    ("Brazil", "São Paulo"),
    ("Sweden", "Stockholm"),
    ("Netherlands", "Amsterdam"),
]

# Weighted (optional) – same 14 countries only
COUNTRY_WEIGHTS = {
    "United States": 5,
    "United Kingdom": 4,
    "Germany": 3,
    "France": 3,
    "Italy": 2,
    "Spain": 2,
    "Poland": 2,
    "India": 2,
    "Japan": 2,
    "Brazil": 2,
    "China": 2,
    "Portugal": 1,
    "Sweden": 1,
    "Netherlands": 1,
}

# Products fixed at 9 (DO NOT expand)
PRODUCTS = [
    (1, "SKU-1001", "Wireless Mouse",      "Electronics", Decimal("19.99"), True),
    (2, "SKU-1002", "Mechanical Keyboard", "Electronics", Decimal("79.00"), True),
    (3, "SKU-1003", "USB-C Hub",           "Electronics", Decimal("34.50"), True),
    (4, "SKU-2001", "Notebook A5",         "Stationery",  Decimal("4.25"),  True),
    (5, "SKU-2002", "Pen Set",             "Stationery",  Decimal("9.90"),  True),
    (6, "SKU-3001", "Coffee Beans 1kg",    "Grocery",     Decimal("24.00"), True),
    (7, "SKU-3002", "Green Tea 200g",      "Grocery",     Decimal("12.50"), True),
    (8, "SKU-4001", "T-Shirt",             "Apparel",     Decimal("18.00"), True),
    (9, "SKU-4002", "Hoodie",              "Apparel",     Decimal("45.00"), True),
]

# ----------------------------
# Better human-readable names
# ----------------------------
FIRST_NAMES = {
    "Poland": ["Jan", "Anna", "Piotr", "Katarzyna", "Michał", "Agnieszka", "Tomasz", "Zofia"],
    "United States": ["James", "Emily", "Michael", "Olivia", "David", "Sophia", "Daniel", "Emma"],
    "Spain": ["Carlos", "Lucía", "Javier", "María", "Miguel", "Sofía", "Pablo", "Carmen"],
    "Italy": ["Luca", "Giulia", "Marco", "Francesca", "Matteo", "Chiara", "Alessandro", "Sara"],
    "France": ["Pierre", "Camille", "Lucas", "Chloé", "Louis", "Emma", "Jules", "Manon"],
    "Germany": ["Max", "Anna", "Leon", "Lena", "Felix", "Mia", "Paul", "Laura"],
    "United Kingdom": ["Oliver", "Amelia", "George", "Isla", "Harry", "Ava", "Jack", "Mia"],
    "Portugal": ["João", "Inês", "Tiago", "Beatriz", "Miguel", "Ana", "Rafael", "Sofia"],
    "Japan": ["Haruto", "Yui", "Sota", "Aoi", "Ren", "Hina", "Yuto", "Sakura"],
    "China": ["Wei", "Li", "Jun", "Fang", "Ming", "Ling", "Hao", "Mei"],
    "India": ["Arjun", "Priya", "Rahul", "Ananya", "Vikram", "Sneha", "Amit", "Neha"],
    "Brazil": ["Lucas", "Ana", "Gabriel", "Beatriz", "Pedro", "Mariana", "Rafael", "Juliana"],
    "Sweden": ["Erik", "Sofia", "Liam", "Ella", "Hugo", "Alva", "Noah", "Maja"],
    "Netherlands": ["Daan", "Emma", "Lars", "Sophie", "Milan", "Julia", "Bram", "Tess"],
}

LAST_NAMES = {
    "Poland": ["Kowalski", "Nowak", "Wiśniewski", "Wójcik", "Kamiński", "Lewandowski"],
    "United States": ["Smith", "Johnson", "Brown", "Taylor", "Miller", "Davis"],
    "Spain": ["García", "Fernández", "López", "Martínez", "Sánchez", "Pérez"],
    "Italy": ["Rossi", "Russo", "Ferrari", "Bianchi", "Romano", "Gallo"],
    "France": ["Dubois", "Moreau", "Laurent", "Simon", "Lefèvre", "Roux"],
    "Germany": ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer"],
    "United Kingdom": ["Smith", "Jones", "Taylor", "Brown", "Wilson", "Davies"],
    "Portugal": ["Silva", "Santos", "Ferreira", "Costa", "Oliveira", "Pereira"],
    "Japan": ["Sato", "Suzuki", "Tanaka", "Yamamoto", "Watanabe", "Ito"],
    "China": ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang"],
    "India": ["Sharma", "Patel", "Singh", "Gupta", "Kumar", "Mehta"],
    "Brazil": ["Silva", "Souza", "Costa", "Oliveira", "Pereira", "Almeida"],
    "Sweden": ["Andersson", "Johansson", "Karlsson", "Nilsson", "Larsson", "Persson"],
    "Netherlands": ["de Jong", "Jansen", "Bakker", "Visser", "Smit", "de Vries"],
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def random_ts(days_back: int = 365) -> datetime:
    now = utc_now()
    return now - timedelta(seconds=random.randint(0, days_back * 24 * 3600))


def pick_country_city(mode: str) -> tuple[str, str]:
    mode = (mode or "weighted").strip().lower()
    if mode == "uniform":
        return random.choice(COUNTRY_CITY)

    names = [c for c, _ in COUNTRY_CITY]
    weights = [COUNTRY_WEIGHTS.get(c, 1) for c in names]
    picked = random.choices(names, weights=weights, k=1)[0]
    for c, city in COUNTRY_CITY:
        if c == picked:
            return c, city
    return random.choice(COUNTRY_CITY)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def generate_name(country: str) -> str:
    first = random.choice(FIRST_NAMES.get(country, ["Alex"]))
    last = random.choice(LAST_NAMES.get(country, ["Smith"]))
    return f"{first} {last}"


def slugify_email(name: str) -> str:
    """
    Convert "Łukasz Wiśniewski" -> "lukasz.wisniewski"
    Keeps only [a-z0-9.] in the final result.
    """
    # normalize accents (Müller -> Muller, Wiśniewski -> Wisniewski)
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    s = ascii_only.lower().strip().replace(" ", ".")
    s = re.sub(r"[^a-z0-9.]+", "", s)
    s = re.sub(r"\.+", ".", s).strip(".")
    return s or "user"


def generate_customers(n_customers: int, country_mode: str) -> list[dict]:
    base = [
        (1,  "Anna Kowalska", "anna.kowalska@example.com", "Poland",         "Warsaw"),
        (2,  "John Smith",    "john.smith@example.com",    "United States",  "New York"),
        (3,  "María García",  "maria.garcia@example.com",  "Spain",          "Madrid"),
        (4,  "Luca Bianchi",  "luca.bianchi@example.com",  "Italy",          "Milan"),
        (5,  "Sophie Martin", "sophie.martin@example.com", "France",         "Paris"),
        (6,  "Max Müller",    "max.mueller@example.com",   "Germany",        "Berlin"),
        (7,  "Olivia Brown",  "olivia.brown@example.com",  "United Kingdom", "London"),
        (8,  "João Silva",    "joao.silva@example.com",    "Portugal",       "Lisbon"),
        (9,  "Aiko Tanaka",   "aiko.tanaka@example.com",   "Japan",          "Tokyo"),
        (10, "Chen Wei",      "chen.wei@example.com",      "China",          "Shanghai"),
        (11, "Priya Sharma",  "priya.sharma@example.com",  "India",          "Bengaluru"),
        (12, "Lucas Souza",   "lucas.souza@example.com",   "Brazil",         "São Paulo"),
        (13, "Emma Svensson", "emma.svensson@example.com", "Sweden",         "Stockholm"),
        (14, "Noah van Dijk", "noah.vandijk@example.com",  "Netherlands",    "Amsterdam"),
    ]

    rows = []
    for cid, name, email, country, city in base:
        rows.append({
            "customer_id": cid,
            "full_name": name,
            "email": email,
            "country": country,
            "city": city,
            "created_at": iso(random_ts()),
        })

    target = max(n_customers, len(base))
    next_id = len(base) + 1

    for i in range(next_id, target + 1):
        country, city = pick_country_city(country_mode)
        name = generate_name(country)
        email_local = slugify_email(name)
        # make unique by appending id
        email = f"{email_local}{i}@example.com"
        rows.append({
            "customer_id": i,
            "full_name": name,
            "email": email,
            "country": country,
            "city": city,
            "created_at": iso(random_ts()),
        })

    return rows


def generate_products() -> list[dict]:
    rows = []
    now = utc_now()
    for pid, sku, name, cat, price, active in PRODUCTS:
        rows.append({
            "product_id": pid,
            "sku": sku,
            "product_name": name,
            "category": cat,
            "unit_price": f"{price:.2f}",
            "active": "true" if active else "false",
            "created_at": iso(now),
        })
    return rows


def generate_orders(n_orders: int, n_customers: int) -> list[dict]:
    statuses = ["paid"] * 8 + ["pending"] + ["refunded"] + ["cancelled"]
    currency = "USD"
    product_choices = [(pid, price) for pid, *_rest, price, _active in PRODUCTS]

    rows = []
    for oid in range(1, n_orders + 1):
        customer_id = random.randint(1, n_customers)
        product_id, unit_price = random.choice(product_choices)
        quantity = random.randint(1, 5)
        status = random.choice(statuses)
        rows.append({
            "order_id": oid,
            "order_ts": iso(random_ts()),
            "customer_id": customer_id,
            "product_id": product_id,
            "quantity": quantity,
            "unit_price": f"{unit_price:.2f}",
            "currency": currency,
            "status": status,
        })
    return rows


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    out_dir = os.getenv("OUT_DIR", "./csv")
    n_customers = int(os.getenv("N_CUSTOMERS", "1000"))
    n_orders = int(os.getenv("N_ORDERS", "50000"))
    country_mode = os.getenv("COUNTRY_MODE", "weighted")  # weighted | uniform

    ensure_dir(out_dir)

    customers = generate_customers(n_customers, country_mode)
    products = generate_products()
    orders = generate_orders(n_orders, max(n_customers, 14))

    write_csv(
        os.path.join(out_dir, "customers.csv"),
        ["customer_id", "full_name", "email", "country", "city", "created_at"],
        customers,
    )
    write_csv(
        os.path.join(out_dir, "products.csv"),
        ["product_id", "sku", "product_name", "category", "unit_price", "active", "created_at"],
        products,
    )
    write_csv(
        os.path.join(out_dir, "orders.csv"),
        ["order_id", "order_ts", "customer_id", "product_id", "quantity", "unit_price", "currency", "status"],
        orders,
    )

    print(f"Generated CSVs in: {out_dir}")
    print(f"customers={len(customers)}, products={len(products)}, orders={len(orders)}, mode={country_mode}")


if __name__ == "__main__":
    main()
