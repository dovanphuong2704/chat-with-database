#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="sales-postgres"
VOLUME_NAME="sales-postgres-data"
POSTGRES_IMAGE="postgres:16"

DB_NAME="salesdb"

ADMIN_USER="postgres"
ADMIN_PASSWORD="postgres"

RO_USER="readonly_user"
RO_PASSWORD="readonly_password"

PREFERRED_PORT="5432"
FALLBACK_PORT="55432"

# CSV generation knobs (override via env)
N_CUSTOMERS="${N_CUSTOMERS:-1000}"
N_ORDERS="${N_ORDERS:-50000}"
COUNTRY_MODE="${COUNTRY_MODE:-weighted}"     # weighted | uniform
CSV_DIR="${CSV_DIR:-./csv}"

echo_step() { printf "\n✅ %s\n" "$1"; }

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}\$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -q ":${port} "
  else
    (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1
  fi
}

HOST_PORT="${PREFERRED_PORT}"
if port_in_use "${PREFERRED_PORT}"; then
  HOST_PORT="${FALLBACK_PORT}"
fi

echo_step "Reset: removing existing container/volume (if any)"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
if docker volume ls --format '{{.Name}}' | grep -q "^${VOLUME_NAME}\$"; then
  docker volume rm "${VOLUME_NAME}" >/dev/null
fi

echo_step "Starting PostgreSQL (${POSTGRES_IMAGE}) on host port ${HOST_PORT}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -e POSTGRES_USER="${ADMIN_USER}" \
  -e POSTGRES_PASSWORD="${ADMIN_PASSWORD}" \
  -p "${HOST_PORT}:5432" \
  -v "${VOLUME_NAME}:/var/lib/postgresql/data" \
  "${POSTGRES_IMAGE}" >/dev/null

echo_step "Waiting for Postgres to become ready..."
until docker exec "${CONTAINER_NAME}" pg_isready -U "${ADMIN_USER}" >/dev/null 2>&1; do
  sleep 1
done

echo_step "Creating database ${DB_NAME} (if not exists)"
docker exec -i "${CONTAINER_NAME}" psql -U "${ADMIN_USER}" -d postgres -v ON_ERROR_STOP=1 <<SQL
SELECT 'CREATE DATABASE ${DB_NAME}'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}')\\gexec
SQL

echo_step "Creating tables (DDL only)"
docker exec -i "${CONTAINER_NAME}" psql -U "${ADMIN_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<'SQL'
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
  customer_id  BIGSERIAL PRIMARY KEY,
  full_name    TEXT NOT NULL,
  email        TEXT UNIQUE NOT NULL,
  country      TEXT NOT NULL,
  city         TEXT NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE products (
  product_id   BIGSERIAL PRIMARY KEY,
  sku          TEXT UNIQUE NOT NULL,
  product_name TEXT NOT NULL,
  category     TEXT NOT NULL,
  unit_price   NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
  active       BOOLEAN NOT NULL DEFAULT TRUE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE orders (
  order_id     BIGSERIAL PRIMARY KEY,
  order_ts     TIMESTAMPTZ NOT NULL DEFAULT now(),
  customer_id  BIGINT NOT NULL REFERENCES customers(customer_id),
  product_id   BIGINT NOT NULL REFERENCES products(product_id),
  quantity     INTEGER NOT NULL CHECK (quantity > 0),
  unit_price   NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
  currency     TEXT NOT NULL DEFAULT 'USD',
  status       TEXT NOT NULL DEFAULT 'paid'
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_product  ON orders(product_id);
CREATE INDEX idx_orders_order_ts ON orders(order_ts);
SQL

echo_step "Generating CSV files with Python -> ${CSV_DIR}"
OUT_DIR="${CSV_DIR}" N_CUSTOMERS="${N_CUSTOMERS}" N_ORDERS="${N_ORDERS}" COUNTRY_MODE="${COUNTRY_MODE}" \
  python3 generate_sales_csv.py

echo_step "Copying CSV files into container"
docker exec "${CONTAINER_NAME}" bash -lc "rm -f /tmp/customers.csv /tmp/products.csv /tmp/orders.csv"
docker cp "${CSV_DIR}/customers.csv" "${CONTAINER_NAME}:/tmp/customers.csv"
docker cp "${CSV_DIR}/products.csv"  "${CONTAINER_NAME}:/tmp/products.csv"
docker cp "${CSV_DIR}/orders.csv"    "${CONTAINER_NAME}:/tmp/orders.csv"

echo_step "Loading CSVs using COPY"
docker exec -i "${CONTAINER_NAME}" psql -U "${ADMIN_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<'SQL'
-- Load with explicit IDs so we can reference them in orders
COPY customers(customer_id, full_name, email, country, city, created_at)
FROM '/tmp/customers.csv'
WITH (FORMAT csv, HEADER true);

COPY products(product_id, sku, product_name, category, unit_price, active, created_at)
FROM '/tmp/products.csv'
WITH (FORMAT csv, HEADER true);

COPY orders(order_id, order_ts, customer_id, product_id, quantity, unit_price, currency, status)
FROM '/tmp/orders.csv'
WITH (FORMAT csv, HEADER true);

-- Fix sequences to continue after max IDs
SELECT setval(pg_get_serial_sequence('customers','customer_id'), COALESCE((SELECT max(customer_id) FROM customers), 1), true);
SELECT setval(pg_get_serial_sequence('products','product_id'),   COALESCE((SELECT max(product_id) FROM products), 1), true);
SELECT setval(pg_get_serial_sequence('orders','order_id'),       COALESCE((SELECT max(order_id) FROM orders), 1), true);
SQL

echo_step "Creating read-only user + grants"
docker exec -i "${CONTAINER_NAME}" psql -U "${ADMIN_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${RO_USER}') THEN
    EXECUTE format('CREATE USER %I WITH PASSWORD %L', '${RO_USER}', '${RO_PASSWORD}');
  ELSE
    EXECUTE format('ALTER USER %I WITH PASSWORD %L', '${RO_USER}', '${RO_PASSWORD}');
  END IF;
END
\$\$;

GRANT CONNECT ON DATABASE ${DB_NAME} TO ${RO_USER};
GRANT USAGE ON SCHEMA public TO ${RO_USER};
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${RO_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ${RO_USER};
SQL

echo_step "All done."

cat <<INFO

Connection info:
  Host: localhost
  Port: ${HOST_PORT}
  Database: ${DB_NAME}

Admin:
  User: ${ADMIN_USER}
  Password: ${ADMIN_PASSWORD}
  Conn: postgresql://${ADMIN_USER}:${ADMIN_PASSWORD}@localhost:${HOST_PORT}/${DB_NAME}

Read-only:
  User: ${RO_USER}
  Password: ${RO_PASSWORD}
  Conn: postgresql://${RO_USER}:${RO_PASSWORD}@localhost:${HOST_PORT}/${DB_NAME}

Generated data:
  customers=${N_CUSTOMERS}
  orders=${N_ORDERS}
  country_mode=${COUNTRY_MODE}  (set COUNTRY_MODE=uniform for pure random)
  csv_dir=${CSV_DIR}

Test:
  psql "postgresql://${RO_USER}:${RO_PASSWORD}@localhost:${HOST_PORT}/${DB_NAME}" \\
    -c "SELECT country, count(*) FROM customers GROUP BY country ORDER BY 2 DESC;"

INFO
