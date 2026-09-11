import pandas as pd
import re
from collections import Counter

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

FILE_PATH = "data/raw/twcs.csv"
BRAND = "AppleSupport"

# --------------------------------------------------
# 2. LOAD REQUIRED COLUMNS
# --------------------------------------------------

print("Loading dataset...")

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id"
]

df = pd.read_csv(
    FILE_PATH,
    usecols=columns,
    low_memory=False
)

print("Rows loaded:", len(df))

# --------------------------------------------------
# 3. PREPARE TWEET LOOKUP
# --------------------------------------------------

df["tweet_id"] = df["tweet_id"].astype(str)

tweet_lookup = df.set_index("tweet_id")[
    ["author_id", "inbound", "text"]
]

# --------------------------------------------------
# 4. FIND APPLESUPPORT REPLIES
# --------------------------------------------------

brand_tweets = df[
    (df["author_id"] == BRAND) &
    (df["inbound"] == False)
].copy()

brand_tweets["response_tweet_id"] = (
    brand_tweets["response_tweet_id"]
    .fillna("")
    .astype(str)
)

links = brand_tweets[
    ["tweet_id", "text", "response_tweet_id"]
].copy()

# A brand tweet can have multiple response IDs
links["response_tweet_id"] = (
    links["response_tweet_id"].str.split(",")
)

links = links.explode("response_tweet_id")

links["response_tweet_id"] = (
    links["response_tweet_id"].str.strip()
)

links = links[
    links["response_tweet_id"].ne("")
]

# --------------------------------------------------
# 5. CONNECT BRAND REPLIES TO CUSTOMER MESSAGES
# --------------------------------------------------

customer_info = tweet_lookup.reindex(
    links["response_tweet_id"]
)

links["customer_inbound"] = (
    customer_info["inbound"].values
)

links["customer_text"] = (
    customer_info["text"].values
)

pairs = links[
    (links["customer_inbound"] == True) &
    (links["customer_text"].notna())
].copy()

print("\nUsable AppleSupport customer messages:", len(pairs))

# --------------------------------------------------
# 6. CLEAN CUSTOMER TEXT
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


pairs["clean_text"] = pairs["customer_text"].apply(clean_text)

# --------------------------------------------------
# 7. BASIC STOPWORDS
# --------------------------------------------------

stopwords = {
    "the", "a", "an", "and", "or", "but", "if", "to",
    "of", "in", "on", "for", "with", "is", "it", "this",
    "that", "my", "me", "i", "you", "your", "we", "they",
    "he", "she", "was", "are", "be", "have", "has", "had",
    "do", "does", "did", "can", "could", "would", "should",
    "will", "just", "not", "so", "from", "at", "as", "about",
    "what", "when", "where", "why", "how", "please", "help",
    "hi", "hello", "thanks", "thank"
}

# --------------------------------------------------
# 8. FIND COMMON WORDS
# --------------------------------------------------

word_counter = Counter()

for text in pairs["clean_text"]:
    words = text.split()

    for word in words:
        if word not in stopwords and len(word) > 2:
            word_counter[word] += 1

print("\n" + "=" * 60)
print("TOP 50 CUSTOMER WORDS")
print("=" * 60)

for word, count in word_counter.most_common(50):
    print(f"{word:25} {count}")

# --------------------------------------------------
# 9. FIND COMMON TWO-WORD PHRASES
# --------------------------------------------------

bigram_counter = Counter()

for text in pairs["clean_text"]:
    words = [
        word
        for word in text.split()
        if word not in stopwords and len(word) > 2
    ]

    for i in range(len(words) - 1):
        phrase = words[i] + " " + words[i + 1]
        bigram_counter[phrase] += 1

print("\n" + "=" * 60)
print("TOP 50 TWO-WORD PHRASES")
print("=" * 60)

for phrase, count in bigram_counter.most_common(50):
    print(f"{phrase:30} {count}")

# --------------------------------------------------
# 10. REPRESENTATIVE EXAMPLES
# --------------------------------------------------

important_terms = [
    word
    for word, count in word_counter.most_common(20)
]

print("\n" + "=" * 60)
print("REPRESENTATIVE CUSTOMER MESSAGES")
print("=" * 60)

for term in important_terms:

    matches = pairs[
        pairs["clean_text"].str.contains(
            rf"\b{re.escape(term)}\b",
            regex=True,
            na=False
        )
    ]

    print(f"\n--- {term.upper()} ---")

    for _, row in matches.head(3).iterrows():
        print(row["customer_text"])