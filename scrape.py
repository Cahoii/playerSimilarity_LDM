import csv
import time
import os
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

base_url = "https://sofifa.com/players?showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=wi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=bp&showCol%5B%5D=ta&showCol%5B%5D=ts&showCol%5B%5D=to&showCol%5B%5D=tp&showCol%5B%5D=te&showCol%5B%5D=td&showCol%5B%5D=tg&showCol%5B%5D=bt"

def scrape_sofifa(max_pages=2):
    # Khởi tạo trình duyệt chống bot
    options = uc.ChromeOptions()
    # Nếu muốn ẩn trình duyệt chạy ngầm (không bật cửa sổ), bỏ comment dòng dưới:
    # options.add_argument('--headless')

    print("Đang khởi tạo trình duyệt Chrome ẩn danh...")
    driver = uc.Chrome(options=options, version_main=151)

    all_players = []
    header_labels = []
    headers_extracted = False

    try:
        for page in range(max_pages):
            offset = page * 60
            url = f"{base_url}&offset={offset}"
            print(f"Đang cào dữ liệu trang {page + 1}: {url}")

            driver.get(url)

            # Đợi tối đa 10 giây cho đến khi bảng dữ liệu xuất hiện
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            except Exception:
                print(
                    f"Trang {page+1} tải quá lâu hoặc bị Cloudflare chặn thử thách (CAPTCHA)."
                )
                break

            # Nghỉ thêm một chút để dữ liệu load hoàn toàn
            time.sleep(3)

            # Lấy HTML nguồn và parse bằng BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")
            table = soup.find("table")

            if not table:
                print("Không tìm thấy bảng dữ liệu.")
                break

            # 1. Trích xuất tiêu đề cột (chỉ làm ở trang đầu)
            if not headers_extracted:
                thead = table.find("thead")
                if thead:
                    for th in thead.find_all("th"):
                        text = th.get_text(strip=True)
                        if not text and th.find("a"):
                            text = th.find("a").get_text(strip=True)
                        header_labels.append(text if text else "N/A")

                    # CHÈN THÊM TIÊU ĐỀ QUỐC TỊCH VÀO SAU CỘT TÊN CẦU THỦ
                    header_labels.insert(2, "Nationality")
                    headers_extracted = True

            # 2. Trích xuất dữ liệu cầu thủ
            tbody = table.find("tbody")
            if not tbody:
                continue

            rows = tbody.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                player_data = []
                for cell in cells:
                    # Ưu tiên lấy tên từ thẻ chứa link profile cầu thủ
                    player_link = cell.find(
                        "a", href=lambda href: href and "player/" in href
                    )
                    if player_link:
                        player_data.append(player_link.get_text(strip=True))

                        # CHÈN THÊM TÍNH NĂNG TRÍCH XUẤT QUỐC TỊCH TỪ LÁ CỜ
                        flag_img = cell.find("img", class_="flag")
                        if not flag_img:
                            flag_img = cell.find("img", title=True)
                        nationality = flag_img["title"] if flag_img and flag_img.has_attr(
                            "title") else "Unknown"
                        player_data.append(nationality)
                    else:
                        player_data.append(cell.get_text(strip=True))

                if player_data:
                    all_players.append(player_data)

    finally:
        # Luôn đóng trình duyệt khi kết thúc hoặc lỗi
        driver.quit()

    if not all_players:
        print("Không có dữ liệu nào được cào.")
        return

    # Chuẩn hóa dữ liệu và xuất file CSV
    max_cols = max(len(row) for row in all_players)
    if len(header_labels) < max_cols:
        header_labels += [
            f"Col_{i}" for i in range(len(header_labels), max_cols)
        ]
    header_labels[0] = "ID/Avatar"

    filename = "data/dataset.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header_labels[:max_cols])
        for row in all_players:
            actual_row = row + [""] * (max_cols - len(row))
            writer.writerow(actual_row[:max_cols])

    print(
        f"\nThành công! Đã cào được {len(all_players)} cầu thủ và lưu vào '{filename}'"
    )


if __name__ == "__main__":
    # 60000 cầu thủ, 1000 trang, mỗi trang 60 cầu thủ
    scrape_sofifa(max_pages=1000)
