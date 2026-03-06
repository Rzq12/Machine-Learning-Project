import pandas as pd
import time
import os
from google_play_scraper import Sort, reviews, app

APP_ID       = "com.gojek.app"
LANG         = "id"          
COUNTRY      = "id"         
TARGET_COUNT = 12000         
BATCH_SIZE   = 200           
OUTPUT_CSV   = "dataset_gojek.csv"

try:
    app_info = app(APP_ID, lang=LANG, country=COUNTRY)
    print(f"Nama Aplikasi : {app_info['title']}")
    print(f"Developer     : {app_info['developer']}")
    print(f"Rating        : {app_info['score']:.2f}")
    print(f"Total Ulasan  : {app_info['ratings']:,}")
except Exception as e:
    print(f"Gagal mengambil info aplikasi: {e}")

all_reviews = []
continuation_token = None

print(f"\nMemulai scraping {TARGET_COUNT:,} ulasan...\n")

for sort_method in [Sort.NEWEST, Sort.MOST_RELEVANT]:
    continuation_token = None
    fetch_count = TARGET_COUNT // 2  # bagi rata antara dua metode sort

    while len(all_reviews) < TARGET_COUNT:
        remaining = TARGET_COUNT - len(all_reviews)
        batch = min(BATCH_SIZE, remaining)

        try:
            result, continuation_token = reviews(
                APP_ID,
                lang=LANG,
                country=COUNTRY,
                sort=sort_method,
                count=batch,
                continuation_token=continuation_token,
            )

            if not result:
                print("Tidak ada ulasan lagi untuk metode sort ini.")
                break

            all_reviews.extend(result)
            print(f"[{sort_method.name}] Terkumpul: {len(all_reviews):,} ulasan", end="\r")

            if continuation_token is None:
                break

            time.sleep(0.5)  

        except Exception as e:
            print(f"\nError saat scraping: {e}")
            time.sleep(5)
            continue

    print()  

    if len(all_reviews) >= TARGET_COUNT:
        break

print(f"\nUlasan sebelum dedup : {len(all_reviews):,}")
seen_ids = set()
unique_reviews = []
for r in all_reviews:
    if r["reviewId"] not in seen_ids:
        seen_ids.add(r["reviewId"])
        unique_reviews.append(r)

print(f"Ulasan setelah dedup : {len(unique_reviews):,}")

df = pd.DataFrame(unique_reviews)[
    ["reviewId", "userName", "content", "score", "thumbsUpCount", "at"]
]

df.rename(columns={
    "reviewId"     : "review_id",
    "userName"     : "user_name",
    "content"      : "review",
    "score"        : "rating",
    "thumbsUpCount": "thumbs_up",
    "at"           : "date",
}, inplace=True)

# Hapus ulasan kosong
df.dropna(subset=["review"], inplace=True)
df = df[df["review"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

def label_sentiment(rating: int) -> str:
    if rating <= 3:
        return "negatif"
    else:
        return "positif"

df["sentiment"] = df["rating"].apply(label_sentiment)


print("\nDistribusi Sentimen:")
print(df["sentiment"].value_counts())
print(f"\nTotal ulasan final: {len(df):,}")

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nDataset disimpan ke:")
print(f"  - {OUTPUT_CSV}")
print("\nScraping selesai!")
