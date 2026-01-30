# 🛒 Sales Demo Database (PostgreSQL + Docker)

A fully reproducible **e-commerce demo dataset** for testing.

The environment includes:

✅ PostgreSQL in Docker
✅ Realistic customers from 14 countries
✅ Human-readable names
✅ Products catalog
✅ Large orders table
✅ Read-only DB user
✅ Data generated via Python → loaded via CSV

## 🚀 Quick Start

```bash
chmod +x setup_sales_db.sh
./setup_sales_db.sh
```

Optional scale:

```bash
N_CUSTOMERS=5000 N_ORDERS=200000 COUNTRY_MODE=uniform ./setup_sales_db.sh
```

## 🔌 Connection Info

After setup, the script prints:

| Role              | Purpose                    |
| ----------------- | -------------------------- |
| **postgres**      | admin access               |
| **readonly_user** | safe analytics / AI access |

Example:

```bash
psql "postgresql://readonly_user:readonly_password@localhost:5432/salesdb"
```

## 📦 Tables Description

### 👤 `customers`

Customer master data.

| Column        | Type         | Description                                |
| ------------- | ------------ | ------------------------------------------ |
| `customer_id` | BIGSERIAL PK | Unique customer identifier                 |
| `full_name`   | TEXT         | Realistic human name generated per country |
| `email`       | TEXT UNIQUE  | Synthetic but natural-looking email        |
| `country`     | TEXT         | One of 14 supported countries              |
| `city`        | TEXT         | Capital or major city                      |
| `created_at`  | TIMESTAMPTZ  | Account creation timestamp                 |

---

### 🛍 `products`

Product catalog (fixed set of 9 items).

| Column         | Type          | Description                         |
| -------------- | ------------- | ----------------------------------- |
| `product_id`   | BIGSERIAL PK  | Product ID                          |
| `sku`          | TEXT UNIQUE   | Product SKU                         |
| `product_name` | TEXT          | Display name                        |
| `category`     | TEXT          | Electronics, Grocery, Apparel, etc. |
| `unit_price`   | NUMERIC(12,2) | Price per item                      |
| `active`       | BOOLEAN       | Availability flag                   |
| `created_at`   | TIMESTAMPTZ   | Product creation timestamp          |

---

### 🧾 `orders`

Transactional sales table.

| Column        | Type          | Description                        |
| ------------- | ------------- | ---------------------------------- |
| `order_id`    | BIGSERIAL PK  | Order ID                           |
| `order_ts`    | TIMESTAMPTZ   | Order timestamp                    |
| `customer_id` | BIGINT FK     | → customers.customer_id            |
| `product_id`  | BIGINT FK     | → products.product_id              |
| `quantity`    | INT           | Units purchased                    |
| `unit_price`  | NUMERIC(12,2) | Price at time of order             |
| `currency`    | TEXT          | Always USD                         |
| `status`      | TEXT          | paid, pending, refunded, cancelled |

Indexes exist on:

```
orders.customer_id
orders.product_id
orders.order_ts
```

---

## 🎲 Data Generation Logic

Data is created in Python:

* Customers get **country-specific names**
* Distribution can be:

  * `weighted` (realistic market sizes)
  * `uniform` (even distribution)
* Orders:

  * Random timestamps (last 365 days)
  * Quantities 1–5
  * Mostly `paid`, some `pending/refunded/cancelled`

---

## 🗂 File Structure

```
setup_sales_db.sh      # Docker + DB + CSV load
generate_sales_csv.py  # Data generator
csv/                   # Generated CSV files
```

---

## 🔐 Read-Only User

Safe for analytics & AI agents.

Permissions:

* CONNECT to DB
* SELECT on all tables
* No INSERT / UPDATE / DELETE

---

## 📊 Example Queries

Top countries by customers:

```sql
SELECT country, COUNT(*)
FROM customers
GROUP BY country
ORDER BY 2 DESC;
```

Revenue by product:

```sql
SELECT p.product_name,
       SUM(o.quantity * o.unit_price) AS revenue
FROM orders o
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'paid'
GROUP BY 1
ORDER BY revenue DESC;
```

Orders per day:

```sql
SELECT date(order_ts), COUNT(*)
FROM orders
GROUP BY 1
ORDER BY 1;
```

