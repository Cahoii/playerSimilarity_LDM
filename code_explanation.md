# Phân tích & Giải thích Mã Nguồn Mô hình Đề xuất Cầu thủ (Player Similarity)

Tài liệu này giải thích chi tiết từng dòng code và luồng hoạt động trong tập lệnh `main.ipynb`. Mục tiêu của dự án này là **Tìm kiếm những cầu thủ tương đồng** bằng cách sử dụng các kỹ thuật Xử lý Ngôn ngữ Tự nhiên (NLP).

Cụ thể, dữ liệu dạng bảng (tabular data) của cầu thủ sẽ được chuyển đổi thành dạng "câu" (bag of tokens). Sau đó, mỗi cầu thủ được biểu diễn bằng một vector đếm tần suất xuất hiện của các token (Bag of Words), từ đó đo lường sự tương đồng (Cosine Similarity) giữa các vector đại diện cho mỗi cầu thủ.

---

## 1. Import các thư viện cần thiết

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import time
```

**Công việc & Ý nghĩa:**

- `pandas` (`pd`): Thư viện cốt lõi để đọc và thao tác dữ liệu dạng bảng (Dataframes).
- `numpy` (`np`): Thư viện tính toán toán học và xử lý mảng tốc độ cao.
- `matplotlib.pyplot` và `seaborn`: Dùng để vẽ các biểu đồ trực quan hóa dữ liệu.

---

## 2. Đọc và khám phá dữ liệu

```python
file_path = "./data/dataset.csv"

df = pd.read_csv(file_path)
df = df.drop(columns=["Value"], errors="ignore")

print("First 5 records:\n", df.head())
df.info()
```

**Công việc & Ý nghĩa:**

- `pd.read_csv(file_path)`: Đọc file dữ liệu cầu thủ ở định dạng `.csv` vào bộ nhớ dưới dạng bảng (`DataFrame`).
- `df.drop(columns=["Value"])`: Loại bỏ cột `Value` vì dữ liệu này không cần thiết hoặc không được sử dụng trong phiên bản tính toán sự tương đồng hiện tại.
- `df.head()`: Lấy ra 5 dòng đầu tiên để xem qua cấu trúc dữ liệu.
- `df.info()`: Hiển thị thông tin tổng quan, bao gồm tên cột, số lượng giá trị không bị thiếu (non-null), và kiểu dữ liệu (chuỗi `str` hay số `int64`). Điều này rất quan trọng để quyết định phương pháp xử lý dữ liệu tiếp theo.

---

## 3. Lọc bỏ các cầu thủ thủ môn (GK)

```python
rows_before_gk_filter = len(df)

df = df.loc[
    df["Best position"].astype("string").str.strip().str.upper().ne("GK")
].copy()

print(f"Removed {rows_before_gk_filter - len(df):,} GK players.")
print(f"Remaining players: {len(df):,}")
```

**Công việc & Ý nghĩa:**

- Mục tiêu ở đây là loại bỏ những cầu thủ chơi ở vị trí thủ môn (`GK`). Thủ môn có hệ thống chỉ số hoàn toàn khác với các cầu thủ thi đấu trên sân (ví dụ: cản phá, phát bóng), do đó việc để chung có thể làm nhiễu mô hình đo lường sự tương đồng.
- `astype("string").str.strip().str.upper()`: Chuẩn hóa chuỗi bằng cách xóa khoảng trắng thừa 2 đầu và viết hoa để đảm bảo không bị sót nếu chữ viết thường.
- `.ne("GK")`: Giữ lại những dòng có vị trí không bằng (Not Equal) `"GK"`.

---

## 4. Xóa dữ liệu trùng lặp (Drop duplicates)

```python
# Kiểm tra và loại bỏ các hàng dữ liệu bị lặp
duplicates = df.duplicated().sum()
print(f"Số lượng hàng bị lặp: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates(keep='first')
    print(f"Số lượng hàng còn lại: {len(df):,}")
else:
    print("Không có hàng nào bị lặp.")
```

**Công việc & Ý nghĩa:**

- Kiểm tra xem có dòng dữ liệu nào giống nhau hoàn toàn hay không bằng hàm `df.duplicated().sum()`.
- Nếu có, gọi hàm `df.drop_duplicates(keep='first')` để loại bỏ các hàng lặp lại (chỉ giữ lại bản ghi xuất hiện đầu tiên). Việc này giúp làm sạch dữ liệu, đảm bảo mỗi cầu thủ (hoặc bản ghi) chỉ được đại diện một lần, tránh làm sai lệch mô hình.

---

## 4.5 Lọc bỏ các cầu thủ có tên chứa chữ số

```python
# Lọc bỏ các cầu thủ có chứa chữ số trong tên (ví dụ: "14J. Putupu")
rows_before_name_filter = len(df)

df = df[~df["Name"].str.contains(r"\d", na=False)].copy()

print(f"Removed {rows_before_name_filter - len(df):,} players with numbers in their name.")
print(f"Remaining players: {len(df):,}")
```
**Công việc & Ý nghĩa:**
- Dùng `str.contains(r"\d")` để tìm các cầu thủ mà cột tên (`Name`) có chứa bất kỳ chữ số nào (như `14J. Putupu`).
- Dấu ngã `~` có tác dụng đảo ngược điều kiện (tức là chỉ giữ lại những cầu thủ **không** chứa chữ số trong tên). Việc này nhằm loại bỏ các dữ liệu rác/lỗi do quá trình thu thập thông tin.

---

## 5. Xử lý & Chuẩn hóa dữ liệu (Data Cleaning)

Dữ liệu thô thường chứa các ký tự không cần thiết, ví dụ "183cm 6'0"" ở cột chiều cao, hoặc "SC Freiburg2024 ~ 2030" ở cột đội bóng. Các hàm dưới đây sinh ra để làm sạch chúng.

### 5.1 Chuẩn hóa Đội bóng & Hợp đồng

```python
def normalize_team(series):
    return (
        series.astype("string")
        .str.replace(r"(?:\d{4}.*|[A-Z][a-z]{2} \d{1,2}, \d{4}.*|Free)$", "", regex=True)
        .str.strip()
    )
```

- Sử dụng biểu thức chính quy (`Regex`) để tìm và xóa năm hợp đồng ở cuối tên đội bóng. Ví dụ: `SC Freiburg2024 ~ 2030` sẽ trở thành `SC Freiburg`. Việc này giúp mô hình coi tất cả cầu thủ của cùng 1 đội là giống nhau.

### 5.2 Chuẩn hóa Cân nặng / Chiều cao

```python
def normalize_measure(series, unit):
    values = series.astype("string").str.extract(
        rf"(\d+(?:\.\d+)?)\s*{unit}", expand=False
    )
    return pd.to_numeric(values, errors="coerce").astype("Float64")
```

- Hàm này trích xuất duy nhất phần con số nằm ngay trước đơn vị (ví dụ `cm` hoặc `kg`). Bỏ qua hệ đo lường của Mỹ (feet/lbs) thường được đính kèm.
- Sau đó chuyển chuỗi thành kiểu số thực (`Float64`).

### 5.3 Chuẩn hóa Body Type

```python
def normalize_body_type(series):
    return series.astype("string").str.replace(
        r"\s*\([^)]*\)", "", regex=True
    ).str.strip()
```

- Loại bỏ phần mô tả cân nặng trong dấu ngoặc của Body Type. Ví dụ: `Normal (170-185)` trở thành `Normal`. Rút gọn giúp giảm số lượng token dư thừa, tăng cường hiệu quả gom cụm cầu thủ.

### 5.4 Áp dụng hàm & Lưu dữ liệu

```python
# Keep only the requested unit or text portion in each field.
df["Team & Contract"] = normalize_team(df["Team & Contract"])
df["Height"] = normalize_measure(df["Height"], "cm")
df["Weight"] = normalize_measure(df["Weight"], "kg")
df["Body type"] = normalize_body_type(df["Body type"])

output_path = "./data/dataset_cleaned.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
```

- Gọi các hàm chuẩn hóa tương ứng, và lưu ra file `.csv` mới để chuẩn bị cho quá trình huấn luyện.

---

## 6. Cấp ID duy nhất và phân loại kiểu dữ liệu

```python
if "ID" in df.columns:
    df = df.drop(columns="ID")
df.insert(0, "ID", range(1, len(df) + 1))

numeric_columns = [
    "Age",
    "Overall rating",
    "Height",
    "Weight",
    "Total attacking",
    "Total skill",
    "Total movement",
    "Total power",
    "Total mentality",
    "Total defending",
]
categorical_columns = [
    "Name",
    "Nationality",
    "Team & Contract",
    "foot",
    "Best position",
    "Body type",
]
```

**Công việc & Ý nghĩa:**

- Cấp 1 cột `ID` tăng dần từ 1 đến N ở vị trí đầu tiên để sau này truy xuất cầu thủ một cách dễ dàng và nhanh chóng (thay vì dựa vào Tên - vì tên có thể bị trùng).
- Khai báo rõ đâu là các biến có giá trị số (Numeric - đo lường được) và đâu là các biến phân loại (Categorical - đặc tính).

---

## 6.5 Phân tích tương quan (Correlation Heatmap) giữa các đặc trưng số

```python
# Trực quan hóa Mối quan hệ (Correlation Heatmap) giữa các đặc trưng dạng số
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].apply(pd.to_numeric, errors="coerce").corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Training Features", fontsize=16)
plt.show()
```

**Công việc & Ý nghĩa:**
- Tính toán ma trận tương quan (Correlation Matrix) giữa tất cả các đặc trưng dạng số (`numeric_columns`).
- Dùng thư viện `seaborn` (`sns.heatmap`) để vẽ bản đồ nhiệt trực quan hóa các hệ số tương quan này. Bản đồ sẽ giúp bạn nhận diện được những chỉ số nào thường đồng biến (tăng cùng tăng) hoặc nghịch biến (tăng cùng giảm) với nhau, qua đó hiểu rõ hơn cấu trúc bên trong dữ liệu trước khi đưa vào mô hình đo sự tương đồng.

---

## 7. Mã hóa Token (Tokenization) cho dữ liệu dạng bảng

NLP dựa trên văn bản, nhưng dữ liệu của chúng ta là dạng bảng tính. Để huấn luyện được, chúng ta cần biến mỗi giá trị ô dữ liệu thành 1 "Từ" (Token) duy nhất không nhầm lẫn.

```python
categorical_prefixes = {"Nationality": "Na", "Team & Contract": "Tc", "foot": "F", ...}
numeric_prefixes = {"Age": "a", "Overall rating": "o", "Height": "h", ...}

tokenized_df = df[["ID", *categorical_columns]].copy()

for column, prefix in categorical_prefixes.items():
    tokenized_df[column] = (
        tokenized_df[column].astype("string")
        .map(lambda value: f"{prefix}_{value}" if pd.notna(value) else "NA")
    )
```

**Công việc & Ý nghĩa với dữ liệu Categorical:**

- Thêm các tiền tố vào giá trị. Giả sử quốc tịch là Bồ Đào Nha `Portugal`, giá trị trở thành `Na_Portugal`. Tại sao? Nếu chỉ ghi `Portugal`, mô hình có thể không phân biệt được đó là Quốc tịch, hay là Tên một câu lạc bộ mang chữ đó. Tiền tố giữ ngữ cảnh cho token.

```python
for column, prefix in numeric_prefixes.items():
    values = pd.to_numeric(df[column], errors="coerce")
    bin_numbers = pd.qcut(values, q=5, labels=False, duplicates="drop")
  
    tokenized_df[column] = (
        bin_numbers.add(1).astype("Int64").astype("string").radd(prefix).fillna("NA")
    )
```

**Công việc & Ý nghĩa với dữ liệu Numeric:**

- Dữ liệu số liên tục rất khó biến thành "từ". Ví dụ cầu thủ cao `183.1cm` và `183.2cm` nên được coi là như nhau. Do đó, tác giả sử dụng `pd.qcut` để chia toàn bộ số liệu cột đó thành **5 giỏ (5 bins) có số lượng bằng nhau**.
- Một cầu thủ thuộc khoảng chiều cao cao nhất (bin thứ 5) sẽ được gán chuỗi `h5`. Tiền tố đứng trước số của bin. Token hóa các dữ liệu số theo bucket (khoảng) là một giải pháp cực hay khi kết hợp NLP.

---

## 8. Xây dựng "Bag of Words" và "Từ điển"

```python
player_bags = tokenized_df[training_columns].apply(
    lambda row: list(row.dropna()),
    axis=1,
)
```

- Ghép toàn bộ Token của một cầu thủ (như quốc tịch, nhóm chiều cao, nhóm cân nặng, vị trí...) lại với nhau thành một list chứa các "từ" (Bag of Tokens). Mỗi cầu thủ giống như "1 câu văn".

```python
token_counter = Counter()
for bag in player_bags:
    token_counter.update(bag)

vocabulary = sorted(token_counter.keys())

token_to_id = {token: index for index, token in enumerate(vocabulary)}
id_to_token = {index: token for token, index in token_to_id.items()}

encoded_player_bags = player_bags.apply(
    lambda bag: [token_to_id[token] for token in bag]
)
```

- Duyệt qua tất cả các tokens của tất cả cầu thủ, thu thập lại thành một từ điển duy nhất (Vocabulary).
- Chuyển (Encode) từng token dạng chữ về một con số ID tương ứng trong từ điển để dễ dàng ánh xạ thành vector số học.

---

## 9. Biểu diễn Cầu thủ bằng Vector Bag of Words (BoW) và Độ tương đồng

Thay vì sử dụng các mô hình học sâu phức tạp để huấn luyện Embedding, chúng ta biểu diễn trực tiếp mỗi cầu thủ dưới dạng một vector tần suất từ (Bag of Words vector).

```python
vocab_size = len(vocabulary)

def bag_to_vector(bag, vocab_size):
    vec = np.zeros(vocab_size)
    for token_id in bag:
        vec[token_id] += 1
    return vec

player_embeddings = np.array([bag_to_vector(bag, vocab_size) for bag in encoded_player_bags])
```

- Mỗi cầu thủ được ánh xạ thành một mảng `numpy` có độ dài bằng tổng số lượng từ vựng (`vocab_size`).
- Tại vị trí `token_id` tương ứng với thuộc tính mà cầu thủ sở hữu, giá trị sẽ được cộng thêm 1. Từ đó, mỗi cầu thủ trở thành một vector số học đại diện.

### Hàm tìm kiếm

```python
def find_similar_players(player_id, top_n=10):
    ...
    similarity_scores = cosine_similarity(
        player_embeddings[player_index:player_index + 1],
        player_embeddings,
    ).ravel()
    ...
```

- Sử dụng độ đo **Cosine Similarity** (Đo góc giữa 2 vector) được cung cấp bởi `sklearn.metrics.pairwise`.
- Cosine càng gần 1, nghĩa là vector của 2 cầu thủ càng có cùng hướng (sở hữu chung nhiều token đặc tính), đồng nghĩa với việc sự tương đồng đặc tính càng lớn.

---

## 10. Triển khai Query nhanh bằng SQLite Database

Thay vì mỗi lần tính toán phải duyệt nguyên mảng Numpy lớn bằng Python, tác giả đã lưu Embeddings dưới dạng chuỗi JSON vào CSDL SQLite.

```python
# 1. Đăng ký hàm nội bộ bằng Python vào SQLite
def cosine_from_json(left_json, right_json):
    left = np.asarray(json.loads(left_json), dtype=float)
    ...
connection.create_function("cosine_similarity", 2, cosine_from_json)

# Thay vì SELECT từng cột, ta có thể SELECT * để lấy toàn bộ thông tin
similarity_result = pd.read_sql_query(
    """
    SELECT *,
           cosine_similarity(embedding, ?) AS "Cos Similarity"
    FROM player_embeddings
    WHERE ID <> ?
    ORDER BY "Cos Similarity" DESC
    LIMIT 10
    """,
    connection,
    params=(target_embedding, query_id),
)
```

- SQLite cho phép viết 1 function Python và đăng ký nó dưới dạng 1 command của SQL. Bằng cách này, ta có thể viết Query SQL trực tiếp để tìm Top 10 người tương đồng nhất một cách gọn gàng và dễ dàng đóng gói phục vụ xây dựng Server Backend sau này.

---

## 11. Cơ chế đánh giá kết quả (Evaluation Score) thủ công

Cuối cùng, người viết xây dựng một hàm tính điểm Heuristic (Dựa trên logic tự nhiên) để kiểm tra chéo xem kết quả sinh ra bởi Machine Learning có tốt không.

```python
def calculate_evaluation_score(query_id, similar_df):
    ...
        # 1. Tính điểm cho Categorical features
        for col in cat_features:
            if str(query_row[col]) == str(candidate_row[col]):
                S += 1.0
              
        # 2. Tính điểm cho Numeric features
        for col in num_features:
            q_val = float(query_row[col])
            c_val = float(candidate_row[col])
            if max(q_val, c_val) != 0:
                S += min(q_val, c_val) / max(q_val, c_val)
    ...
```

**Ý nghĩa cách chấm điểm:**

- Các cột Categorical (Quốc tịch, Đội bóng, v.v.): Nếu giống nhau y hệt -> Điểm được cộng thêm 1.
- Các cột Numeric (Tốc độ, Chuyền, Thể lực, v.v.): So sánh tỷ lệ. `min/max` sẽ luôn trả về 1 con số nhỏ hơn hoặc bằng 1. Hai người chỉ số càng gần nhau, phân số sẽ càng gần 1. (Ví dụ 80 và 82 -> Điểm = 80/82 = 0.97).
- Tổng toàn bộ điểm lại rồi chia cho tổng số lượng cột (Total Features) để ra tỷ lệ % độ giống nhau (Accuracy).

Đây là một phương pháp rất hay để xác thực mô hình AI (Validation) thay vì chỉ tin vào con số toán học khô khan (Cosine).
