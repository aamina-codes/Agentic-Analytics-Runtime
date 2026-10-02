import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "ecommerce_orders.csv"
OUTPUT_PATH = PROJECT_DIR / "evaluation" / "expected_results.json"


df = pd.read_csv(DATA_PATH)
df["order_date"] = pd.to_datetime(df["order_date"])

results = {}


# Q01 — Highest total sales by region
region_sales = (
    df.groupby("region")["sales"]
    .sum()
    .sort_values(ascending=False)
)

results["q01"] = {
    "answer": region_sales.index[0],
    "value": round(float(region_sales.iloc[0]), 2),
    "metric": "total_sales",
}


# Q02 — Highest total profit by category
category_profit = (
    df.groupby("category")["profit"]
    .sum()
    .sort_values(ascending=False)
)

results["q02"] = {
    "answer": category_profit.index[0],
    "value": round(float(category_profit.iloc[0]), 2),
    "metric": "total_profit",
}


# Q03 — Highest average sales per order by customer segment
segment_avg_sales = (
    df.groupby("customer_segment")["sales"]
    .mean()
    .sort_values(ascending=False)
)

results["q03"] = {
    "answer": segment_avg_sales.index[0],
    "value": round(float(segment_avg_sales.iloc[0]), 2),
    "metric": "average_sales",
}


# Q04 — Total sales in 2024
sales_2024 = df.loc[df["order_date"].dt.year == 2024, "sales"].sum()

results["q04"] = {
    "answer": round(float(sales_2024), 2),
    "metric": "total_sales_2024",
}


# Q05 — Total sales in 2025
sales_2025 = df.loc[df["order_date"].dt.year == 2025, "sales"].sum()

results["q05"] = {
    "answer": round(float(sales_2025), 2),
    "metric": "total_sales_2025",
}


# Q06 — Sales change between 2024 and 2025
sales_change = sales_2025 - sales_2024
sales_change_pct = (sales_change / sales_2024) * 100

results["q06"] = {
    "absolute_change": round(float(sales_change), 2),
    "percentage_change": round(float(sales_change_pct), 4),
    "from_year": 2024,
    "to_year": 2025,
}


# Q07 — Highest average sales by region
region_avg_sales = (
    df.groupby("region")["sales"]
    .mean()
    .sort_values(ascending=False)
)

results["q07"] = {
    "answer": region_avg_sales.index[0],
    "value": round(float(region_avg_sales.iloc[0]), 2),
    "metric": "average_sales",
}


# Q08 — Total profit from completed orders
completed_profit = df.loc[
    df["order_status"] == "Completed",
    "profit",
].sum()

results["q08"] = {
    "answer": round(float(completed_profit), 2),
    "metric": "completed_order_profit",
}


# Q09 — Highest average profit per order by category
category_avg_profit = (
    df.groupby("category")["profit"]
    .mean()
    .sort_values(ascending=False)
)

results["q09"] = {
    "answer": category_avg_profit.index[0],
    "value": round(float(category_avg_profit.iloc[0]), 2),
    "metric": "average_profit",
}


# Q10 — Percentage of orders that were cancelled
cancelled_count = (df["order_status"] == "Cancelled").sum()
total_orders = len(df)

cancelled_pct = (cancelled_count / total_orders) * 100

results["q10"] = {
    "cancelled_orders": int(cancelled_count),
    "total_orders": int(total_orders),
    "percentage": round(float(cancelled_pct), 4),
}


# Q11 — Most frequently used payment method
payment_counts = (
    df["payment_method"]
    .value_counts()
)

results["q11"] = {
    "answer": payment_counts.index[0],
    "order_count": int(payment_counts.iloc[0]),
}


# Q12 — Highest total sales by country
country_sales = (
    df.groupby("country")["sales"]
    .sum()
    .sort_values(ascending=False)
)

results["q12"] = {
    "answer": country_sales.index[0],
    "value": round(float(country_sales.iloc[0]), 2),
    "metric": "total_sales",
}


# Q13 — Sales and profit by region
region_summary = (
    df.groupby("region")
    .agg(
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
    )
    .sort_values("total_sales", ascending=False)
)

results["q13"] = {
    "rows": [
        {
            "region": index,
            "total_sales": round(float(row["total_sales"]), 2),
            "total_profit": round(float(row["total_profit"]), 2),
        }
        for index, row in region_summary.iterrows()
    ]
}


# Q14 — Customer segment with highest total profit
segment_profit = (
    df.groupby("customer_segment")["profit"]
    .sum()
    .sort_values(ascending=False)
)

results["q14"] = {
    "answer": segment_profit.index[0],
    "value": round(float(segment_profit.iloc[0]), 2),
    "metric": "total_profit",
}


# Q15 — Highest sales category among completed orders
completed = df[df["order_status"] == "Completed"]

completed_category_sales = (
    completed.groupby("category")["sales"]
    .sum()
    .sort_values(ascending=False)
)

results["q15"] = {
    "answer": completed_category_sales.index[0],
    "value": round(float(completed_category_sales.iloc[0]), 2),
    "metric": "completed_order_sales",
}


# Q16 — Highest profit margin by region
region_margin = (
    df.groupby("region")
    .agg(
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
    )
)

region_margin["profit_margin"] = (
    region_margin["total_profit"]
    / region_margin["total_sales"]
    * 100
)

region_margin = region_margin.sort_values(
    "profit_margin",
    ascending=False,
)

results["q16"] = {
    "answer": region_margin.index[0],
    "profit_margin": round(
        float(region_margin.iloc[0]["profit_margin"]),
        4,
    ),
}


# Q17 — Difference in total sales between the two segments
# with the highest average order value
segment_stats = (
    df.groupby("customer_segment")
    .agg(
        average_sales=("sales", "mean"),
        total_sales=("sales", "sum"),
    )
    .sort_values("average_sales", ascending=False)
)

top_two_segments = segment_stats.head(2)

segment_1 = top_two_segments.index[0]
segment_2 = top_two_segments.index[1]

segment_1_sales = top_two_segments.iloc[0]["total_sales"]
segment_2_sales = top_two_segments.iloc[1]["total_sales"]

results["q17"] = {
    "segment_1": segment_1,
    "segment_1_total_sales": round(float(segment_1_sales), 2),
    "segment_2": segment_2,
    "segment_2_total_sales": round(float(segment_2_sales), 2),
    "difference": round(
        float(abs(segment_1_sales - segment_2_sales)),
        2,
    ),
}


# Q18 — Highest average discount by category
category_discount = (
    df.groupby("category")
    .agg(
        average_discount=("discount", "mean"),
        average_profit=("profit", "mean"),
    )
    .sort_values("average_discount", ascending=False)
)

top_discount_category = category_discount.index[0]

results["q18"] = {
    "category": top_discount_category,
    "average_discount": round(
        float(
            category_discount.loc[
                top_discount_category,
                "average_discount",
            ]
        ),
        4,
    ),
    "average_profit": round(
        float(
            category_discount.loc[
                top_discount_category,
                "average_profit",
            ]
        ),
        2,
    ),
}


# Q19 — Region with highest sales for each year
year_region = (
    df.assign(year=df["order_date"].dt.year)
    .groupby(["year", "region"])["sales"]
    .sum()
    .reset_index()
)

year_region_results = []

for year in sorted(year_region["year"].unique()):
    year_data = (
        year_region[year_region["year"] == year]
        .sort_values("sales", ascending=False)
    )

    top = year_data.iloc[0]

    year_region_results.append(
        {
            "year": int(year),
            "region": top["region"],
            "sales": round(float(top["sales"]), 2),
        }
    )

results["q19"] = {
    "rows": year_region_results
}


# Q20 — Highest-profit category among orders
# with discount >= 20%
discounted = df[df["discount"] >= 0.20]

discount_category_profit = (
    discounted.groupby("category")["profit"]
    .sum()
    .sort_values(ascending=False)
)

results["q20"] = {
    "answer": discount_category_profit.index[0],
    "value": round(
        float(discount_category_profit.iloc[0]),
        2,
    ),
    "metric": "profit_with_discount_at_least_20_percent",
}


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


print("Expected results generated successfully.")
print(f"Questions evaluated: {len(results)}")
print(f"Output: {OUTPUT_PATH}")