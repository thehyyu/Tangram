import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, NavigableString

BASE_URL = "https://archive.tw"
SPEECHES_URL = f"{BASE_URL}/speeches"
OUTPUT_PATH = Path("data/raw_speeches.json")
DELAY = 1.0  # 每頁請求間隔 1 秒，避免對伺服器造成壓力

# 遇到 502/503/504 自動重試 3 次，每次等待間隔加倍（1s → 2s → 4s）
_session = requests.Session()
_retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))

# 非 Q&A 格式的標題關鍵字，爬到就跳過
SKIP_PATTERNS = ["商周專欄"]


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def should_skip(title: str) -> bool:
    return any(p in title for p in SKIP_PATTERNS)


def get_speech_urls() -> list[str]:
    """從演講列表頁取得所有中文演講的 URL。"""
    resp = _session.get(SPEECHES_URL, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    seen = set()
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # 只取符合 /YYYY-MM-DD-... 格式的連結
        if not re.match(r"^/\d{4}-\d{2}-\d{2}-", href):
            continue
        title_decoded = unquote(href)
        if not has_chinese(title_decoded):
            continue  # 跳過英文演講
        if should_skip(title_decoded):
            continue  # 跳過商周專欄等非 Q&A 格式
        full_url = urljoin(BASE_URL, href)
        if full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)
    return urls


def parse_li(li) -> tuple[str | None, str | None]:
    """從單一 <li> 取出發言者名稱和發言內容。"""
    skip_texts = {"前後文", "連結", "Context", "Link"}

    speaker_a = li.find("a", href=lambda h: h and "/speaker/" in h)
    if not speaker_a:
        return None, None

    speaker = speaker_a.get_text(strip=True)

    text_parts = []
    for child in li.children:
        if child is speaker_a or getattr(child, "name", None) == "img":
            continue
        if getattr(child, "name", None) == "a":
            if child.get_text(strip=True) in skip_texts:
                continue
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                text_parts.append(t)
        elif getattr(child, "name", None) in ("p", "span", "div"):
            t = child.get_text(strip=True)
            if t and t not in skip_texts:
                text_parts.append(t)

    return speaker, " ".join(text_parts).strip()


def parse_speech(url: str) -> dict | None:
    """爬取單篇演講，回傳結構化資料；若無 Q&A 結構則回傳 None。"""
    resp = _session.get(url, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    h1 = soup.find("h1")
    if not h1:
        return None

    full_title = h1.get_text(strip=True)
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})\s*(.*)", full_title)
    date = date_match.group(1) if date_match else ""
    title = date_match.group(2) if date_match else full_title

    # 收集所有發言輪次
    turns = []
    for li in soup.find_all("li"):
        speaker, text = parse_li(li)
        if not speaker or not text:
            continue
        turns.append({"speaker": speaker, "is_tang": "唐鳳" in speaker, "text": text})

    # 確認唐鳳有出現
    if not any(t["is_tang"] for t in turns):
        return None

    # 建立 Q&A pair：連續非唐鳳發言 → 連續唐鳳發言
    qa_pairs = []
    i = 0
    while i < len(turns):
        if not turns[i]["is_tang"]:
            question_parts = []
            while i < len(turns) and not turns[i]["is_tang"]:
                question_parts.append(turns[i]["text"])
                i += 1
            answer_parts = []
            while i < len(turns) and turns[i]["is_tang"]:
                answer_parts.append(turns[i]["text"])
                i += 1
            if answer_parts:
                qa_pairs.append(
                    {
                        "question": " ".join(question_parts),
                        "answer": " ".join(answer_parts),
                    }
                )
        else:
            i += 1

    if not qa_pairs:
        return None

    return {"url": url, "title": title, "date": date, "qa_pairs": qa_pairs}


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print("取得演講列表...")
    urls = get_speech_urls()
    print(f"找到 {len(urls)} 篇中文演講（已過濾英文與商周專欄）\n")

    results = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            speech = parse_speech(url)
            if speech:
                results.append(speech)
                print(f"  ✓ {speech['title']} — {len(speech['qa_pairs'])} 組 Q&A")
            else:
                print("  ✗ 跳過（無 Q&A 結構或唐鳳未出現）")
        except Exception as e:
            print(f"  ✗ 錯誤：{e}")
        time.sleep(DELAY)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_qa = sum(len(s["qa_pairs"]) for s in results)
    print(f"\n完成！{len(results)} 篇演講，共 {total_qa} 組 Q&A")
    print(f"已存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
