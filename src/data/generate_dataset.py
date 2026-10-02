"""
Generate a reproducible synthetic e-commerce dataset for Agentic Analytics Runtime.

The dataset is intentionally designed with realistic relationships and
controlled patterns so it can be used for:
- natural-language analytics
- SQL generation
- DuckDB execution
- agent evaluation
- visualization
- evidence validation
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_ORDERS = 60_000

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

CSV_PATH = RAW_DIR / "ecommerce_orders.csv"
PARQUET_PATH = PARQUET_DIR / "ecommerce_orders.parquet"


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

REGION_DATA = {
    "North America": {
        "countries": {
            "United States": ["New York", "Chicago", "Los Angeles", "Houston"],
            "Canada": ["Toronto", "Vancouver", "Montreal"],
        },
        "sales_multiplier": 1.18,
    },
    "Europe": {
        "countries": {
            "United Kingdom": ["London", "Manchester", "Birmingham"],
            "Germany": ["Berlin", "Munich", "Hamburg"],
            "France": ["Paris", "Lyon", "Marseille"],
        },
        "sales_multiplier": 1.08,
    },
    "Asia Pacific": {
        "countries": {
            "India": ["Mumbai", "Delhi", "Bengaluru", "Hyderabad"],
            "Singapore": ["Singapore"],
            "Australia": ["Sydney", "Melbourne", "Brisbane"],
        },
        "sales_multiplier": 0.96,
    },
    "Latin America": {
        "countries": {
            "Brazil": ["Sao Paulo", "Rio de Janeiro", "Brasilia"],
            "Mexico": ["Mexico City", "Guadalajara", "Monterrey"],
        },
        "sales_multiplier": 0.86,
    },
}

CUSTOMER_SEGMENTS = [
    "Consumer",
    "Corporate",
    "Home Office",
]

SEGMENT_PROBABILITIES = [
    0.58,
    0.27,
    0.15,
]

PAYMENT_METHODS = [
    "Credit Card",
    "Debit Card",
    "UPI",
    "PayPal",
    "Bank Transfer",
]

PAYMENT_PROBABILITIES = [
    0.31,
    0.18,
    0.24,
    0.17,
    0.10,
]

CATEGORY_DATA = {
    "Technology": {
        "subcategories": {
            "Laptops": [
                "ProBook 14",
                "UltraBook X",
                "WorkMate 15",
                "PowerStation 17",
            ],
            "Phones": [
                "Nova X",
                "Pixel Pro",
                "Galaxy Max",
                "OnePlus Edge",
            ],
            "Accessories": [
                "Wireless Mouse",
                "Mechanical Keyboard",
                "USB-C Hub",
                "Webcam HD",
            ],
        },
        "base_price": 420,
        "margin": 0.17,
        "discount": 0.10,
    },
    "Office Supplies": {
        "subcategories": {
            "Paper": [
                "Premium A4 Pack",
                "Recycled Paper Box",
                "Color Paper Pack",
            ],
            "Stationery": [
                "Executive Notebook",
                "Gel Pen Set",
                "Desk Organizer",
            ],
            "Storage": [
                "Archive Box",
                "Document Folder",
                "Storage Binder",
            ],
        },
        "base_price": 55,
        "margin": 0.23,
        "discount": 0.08,
    },
    "Furniture": {
        "subcategories": {
            "Chairs": [
                "Ergonomic Chair",
                "Executive Chair",
                "Mesh Office Chair",
            ],
            "Desks": [
                "Standing Desk",
                "Executive Desk",
                "Compact Work Desk",
            ],
            "Storage": [
                "Filing Cabinet",
                "Office Shelf",
                "Storage Unit",
            ],
        },
        "base_price": 280,
        "margin": 0.20,
        "discount": 0.13,
    },
    "Home & Lifestyle": {
        "subcategories": {
            "Kitchen": [
                "Air Fryer",
                "Coffee Maker",
                "Blender Pro",
            ],
            "Fitness": [
                "Yoga Mat",
                "Resistance Kit",
                "Fitness Tracker",
            ],
            "Decor": [
                "Desk Lamp",
                "Wall Clock",
                "Decor Set",
            ],
        },
        "base_price": 120,
        "margin": 0.26,
        "discount": 0.09,
    },
}

CATEGORIES = list(CATEGORY_DATA.keys())

CATEGORY_PROBABILITIES = [
    0.31,
    0.27,
    0.21,
    0.21,
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_location_lookup():
    """Create a flat lookup table for region/country/city sampling."""

    rows = []

    for region, region_info in REGION_DATA.items():
        for country, cities in region_info["countries"].items():
            for city in cities:
                rows.append(
                    {
                        "region": region,
                        "country": country,
                        "city": city,
                    }
                )

    return pd.DataFrame(rows)


def generate_dates(rng, n):
    """Generate order dates with mild seasonal weighting."""

    dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )

    # Weight November and December more heavily to create realistic
    # year-end shopping seasonality.
    weights = np.ones(len(dates), dtype=float)

    for i, date in enumerate(dates):
        if date.month == 11:
            weights[i] *= 1.45
        elif date.month == 12:
            weights[i] *= 1.65
        elif date.month in (3, 4):
            weights[i] *= 1.05

    weights /= weights.sum()

    return pd.to_datetime(
        rng.choice(
            dates,
            size=n,
            replace=True,
            p=weights,
        )
    )


def generate_locations(rng, n):
    """Generate region, country and city for each order."""

    location_lookup = build_location_lookup()

    # Weight regions according to their relative order volume.
    region_weights = {
        "North America": 0.34,
        "Europe": 0.29,
        "Asia Pacific": 0.25,
        "Latin America": 0.12,
    }

    regions = rng.choice(
        list(region_weights.keys()),
        size=n,
        p=list(region_weights.values()),
    )

    countries = []
    cities = []

    for region in regions:
        region_info = REGION_DATA[region]

        country_names = list(region_info["countries"].keys())
        country = rng.choice(country_names)

        city = rng.choice(region_info["countries"][country])

        countries.append(country)
        cities.append(city)

    return regions, np.array(countries), np.array(cities)


def generate_products(rng, n):
    """Generate category, subcategory and product."""

    categories = rng.choice(
        CATEGORIES,
        size=n,
        p=CATEGORY_PROBABILITIES,
    )

    subcategories = []
    products = []

    for category in categories:
        subcategory_names = list(
            CATEGORY_DATA[category]["subcategories"].keys()
        )

        subcategory = rng.choice(subcategory_names)

        product = rng.choice(
            CATEGORY_DATA[category]["subcategories"][subcategory]
        )

        subcategories.append(subcategory)
        products.append(product)

    return (
        np.array(categories),
        np.array(subcategories),
        np.array(products),
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset():
    """Generate the complete synthetic e-commerce dataset."""

    rng = np.random.default_rng(SEED)

    print("Generating Agentic Analytics Runtime dataset...")
    print(f"Rows: {N_ORDERS:,}")
    print(f"Date range: {START_DATE} → {END_DATE}")
    print(f"Random seed: {SEED}")

    # -----------------------------------------------------------------------
    # Core identifiers
    # -----------------------------------------------------------------------

    order_ids = np.array(
        [f"ORD-{i:07d}" for i in range(1, N_ORDERS + 1)]
    )

    customer_pool_size = 12_000

    customer_ids = rng.choice(
        [
            f"CUST-{i:05d}"
            for i in range(1, customer_pool_size + 1)
        ],
        size=N_ORDERS,
        replace=True,
    )

    customer_segments = rng.choice(
        CUSTOMER_SEGMENTS,
        size=N_ORDERS,
        p=SEGMENT_PROBABILITIES,
    )

    # -----------------------------------------------------------------------
    # Dates and geography
    # -----------------------------------------------------------------------

    order_dates = generate_dates(rng, N_ORDERS)

    regions, countries, cities = generate_locations(
        rng,
        N_ORDERS,
    )

    # -----------------------------------------------------------------------
    # Products
    # -----------------------------------------------------------------------

    categories, subcategories, products = generate_products(
        rng,
        N_ORDERS,
    )

    # -----------------------------------------------------------------------
    # Order quantities
    # -----------------------------------------------------------------------

    quantity_lambda = np.select(
        [
            customer_segments == "Corporate",
            customer_segments == "Home Office",
            customer_segments == "Consumer",
        ],
        [
            3.1,
            2.3,
            1.8,
        ],
        default=2.0,
    )

    quantities = np.maximum(
        1,
        rng.poisson(quantity_lambda),
    )

    # -----------------------------------------------------------------------
    # Discounts
    # -----------------------------------------------------------------------

    base_discounts = np.array(
        [
            CATEGORY_DATA[category]["discount"]
            for category in categories
        ]
    )

    discount_noise = rng.normal(
        loc=0.0,
        scale=0.035,
        size=N_ORDERS,
    )

    discounts = np.clip(
        base_discounts + discount_noise,
        0.0,
        0.35,
    )

    # -----------------------------------------------------------------------
    # Base prices
    # -----------------------------------------------------------------------

    base_prices = np.array(
        [
            CATEGORY_DATA[category]["base_price"]
            for category in categories
        ]
    )

    price_noise = rng.lognormal(
        mean=0.0,
        sigma=0.28,
        size=N_ORDERS,
    )

    unit_prices = base_prices * price_noise

    # -----------------------------------------------------------------------
    # Sales / revenue
    # -----------------------------------------------------------------------

    region_multipliers = np.array(
        [
            REGION_DATA[region]["sales_multiplier"]
            for region in regions
        ]
    )

    segment_multipliers = np.select(
        [
            customer_segments == "Corporate",
            customer_segments == "Home Office",
            customer_segments == "Consumer",
        ],
        [
            1.22,
            1.08,
            1.00,
        ],
        default=1.0,
    )

    # Year-end seasonal effect.
    month_multipliers = np.select(
        [
            order_dates.month == 11,
            order_dates.month == 12,
            order_dates.month.isin([1, 2]),
        ],
        [
            1.18,
            1.30,
            0.92,
        ],
        default=1.0,
    )

    gross_sales = (
        unit_prices
        * quantities
        * region_multipliers
        * segment_multipliers
        * month_multipliers
    )

    sales = gross_sales * (1.0 - discounts)

    # Small amount of transaction noise.
    sales *= rng.normal(
        loc=1.0,
        scale=0.06,
        size=N_ORDERS,
    )

    sales = np.maximum(
        sales,
        5.0,
    )

    # -----------------------------------------------------------------------
    # Shipping cost
    # -----------------------------------------------------------------------

    shipping_cost = (
        8
        + quantities * rng.uniform(2.5, 7.5, size=N_ORDERS)
        + np.where(
            categories == "Furniture",
            rng.uniform(12, 28, size=N_ORDERS),
            0,
        )
    )

    shipping_cost *= region_multipliers ** -0.15

    # -----------------------------------------------------------------------
    # Profit
    # -----------------------------------------------------------------------

    base_margins = np.array(
        [
            CATEGORY_DATA[category]["margin"]
            for category in categories
        ]
    )

    # Higher discounts reduce effective margin.
    effective_margin = (
        base_margins
        - discounts * 0.55
    )

    profit = (
        sales * effective_margin
        - shipping_cost
    )

    profit += rng.normal(
        loc=0.0,
        scale=np.maximum(sales * 0.025, 1.0),
        size=N_ORDERS,
    )

    # -----------------------------------------------------------------------
    # Payment method
    # -----------------------------------------------------------------------

    payment_methods = rng.choice(
        PAYMENT_METHODS,
        size=N_ORDERS,
        p=PAYMENT_PROBABILITIES,
    )

    # -----------------------------------------------------------------------
    # Order status
    # -----------------------------------------------------------------------

    # Base cancellation probability.
    cancellation_probability = np.full(
        N_ORDERS,
        0.045,
    )

    cancellation_probability += np.where(
        discounts > 0.20,
        0.015,
        0.0,
    )

    cancellation_probability += np.where(
        categories == "Technology",
        0.008,
        0.0,
    )

    cancelled = (
        rng.random(N_ORDERS)
        < cancellation_probability
    )

    return_probability = np.full(
        N_ORDERS,
        0.035,
    )

    return_probability += np.where(
        discounts > 0.20,
        0.012,
        0.0,
    )

    return_probability += np.where(
        categories == "Technology",
        0.010,
        0.0,
    )

    returned = (
        ~cancelled
        & (
            rng.random(N_ORDERS)
            < return_probability
        )
    )

    order_status = np.full(
        N_ORDERS,
        "Completed",
        dtype=object,
    )

    order_status[cancelled] = "Cancelled"
    order_status[returned] = "Returned"

    # -----------------------------------------------------------------------
    # Build dataframe
    # -----------------------------------------------------------------------

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "order_date": order_dates,
            "customer_id": customer_ids,
            "customer_segment": customer_segments,
            "region": regions,
            "country": countries,
            "city": cities,
            "category": categories,
            "subcategory": subcategories,
            "product": products,
            "quantity": quantities.astype(int),
            "sales": np.round(sales, 2),
            "discount": np.round(discounts, 4),
            "profit": np.round(profit, 2),
            "shipping_cost": np.round(shipping_cost, 2),
            "payment_method": payment_methods,
            "order_status": order_status,
        }
    )

    # Sort chronologically to make the dataset easier to inspect.
    df = df.sort_values(
        ["order_date", "order_id"]
    ).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARQUET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        CSV_PATH,
        index=False,
    )

    df.to_parquet(
        PARQUET_PATH,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("Dataset generated successfully.")
    print(f"CSV:     {CSV_PATH}")
    print(f"Parquet: {PARQUET_PATH}")
    print()
    print("Shape:")
    print(df.shape)
    print()
    print("Columns:")
    print(list(df.columns))
    print()
    print("Date range:")
    print(
        f"{df['order_date'].min().date()} → "
        f"{df['order_date'].max().date()}"
    )
    print()
    print("Order status:")
    print(
        df["order_status"]
        .value_counts()
        .to_string()
    )
    print()
    print("Category distribution:")
    print(
        df["category"]
        .value_counts()
        .to_string()
    )
    print()
    print("Region distribution:")
    print(
        df["region"]
        .value_counts()
        .to_string()
    )
    print()
    print("Sales:")
    print(
        f"Total: ${df['sales'].sum():,.2f}"
    )
    print(
        f"Average order: ${df['sales'].mean():,.2f}"
    )
    print()
    print("Profit:")
    print(
        f"Total: ${df['profit'].sum():,.2f}"
    )
    print(
        f"Average order: ${df['profit'].mean():,.2f}"
    )


if __name__ == "__main__":
    generate_dataset()