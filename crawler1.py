import requests
import logging
import re
import json
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

BASE_URL = 'https://www.xigushi.com/'

html1 = """
<ul>
<li><a href="/ymgs/"><span>幽默故事</span></a></li>
<li><a href="/thgs/"><span>儿童故事</span></a></li>
<li><a href="/aqgs/"><span>爱情故事</span></a></li>
<li><a href="/jcgs/"><span>职场故事</span></a></li>
<li><a href="/lzgs/"><span>励志故事</span></a></li>
<li><a href="/zlgs/"><span>哲理故事</span></a></li>
<li><a href="/xygs/"><span>校园故事</span></a></li>
<li><a href="/rsgs/"><span>人生故事</span></a></li>
<li><a href="/yygs/"><span>寓言故事</span></a></li>
<li><a href="/mrgs/"><span>名人故事</span></a></li>
<li><a href="/qqgs/"><span>亲情故事</span></a></li>
<li><a href="/yqgs/"><span>友情故事</span></a></li>
</ul>
"""

s = requests.Session()

def extract_links(html1):
    pattern = re.compile('<li><a href="(.*?)"><span>(.*?)</span></a></li>')
    items = re.findall(pattern, html1)
    return {item[1]: item[0] for item in items}  # Create a dictionary with link text as keys and href as values


def scrape_page(url):
    logging.info('scraping %s...', url)
    try:
        response = s.get(url)
        response.encoding = 'utf-8'  # manually set encoding to utf-8
        if response.status_code == 200:
            return response.text
        logging.error('get invalid status code %s while scraping %s',
                      response.status_code, url)
    except requests.RequestException:
        logging.error('error occurred while scraping %s', url, exc_info=True)


def scrape_index(page, BASE_URL, cati_num):
    index_url = f'{BASE_URL}/list_{cati_num}_{page}.html'
    return scrape_page(index_url)


def displaying_parse_index(html, BASE_URL):
    pattern = re.compile(r'<a.*?href="(.*?\.html)".*?>(.*?)</a>')
    items = re.findall(pattern, html)
    if not items:
        return []
    for item in items:
        detail_url = urljoin(BASE_URL, item[0])  # Note that item is now a tuple
        logging.info('get detail url %s', detail_url)
        yield detail_url


def scrape_detail(url):
    return scrape_page(url)


def parse_detail(html, category):
    title_pattern = re.compile('<h1>(.*?)</h1>')
    content_pattern = re.compile('<p>(.*?)</p>', re.S)

    title_search = re.search(title_pattern, html)
    title = title_search.group(1).strip() if title_search else None

    content_search = re.search(content_pattern, html)
    content = content_search.group(1).strip() if content_search else None

    if content:
        content = re.sub('[^\u4e00-\u9fa5，。？\n]', '', content)

    words = len(content) if content else 0

    return {
        'category': category,
        'title': title,
        'content': content,
        'words': words
    }


def main():
    links = extract_links(html1)
    cor_total_page = [58, 41, 53, 52, 50, 42, 43, 56, 46, 43, 52, 32]
    cati_nums = [3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]

    with ThreadPoolExecutor(max_workers=10) as executor, open('output.json', 'w') as f:
        results = []
        for (category, relative_url), total_page, cati_num in zip(links.items(), cor_total_page, cati_nums):
            BASE_URL = urljoin('https://www.xigushi.com', relative_url)

            futures = {executor.submit(scrape_index, page, BASE_URL, cati_num): page for page in range(1, total_page + 1)}

            for future in as_completed(futures):
                index_html = future.result()
                detail_urls = displaying_parse_index(index_html, BASE_URL)
                for detail_url in detail_urls:
                    detail_html = scrape_detail(detail_url)
                    data = parse_detail(detail_html, category)
                    results.append(data)  # Add data to list
                    logging.info('get detail data %s', data)
        json.dump(results, f, ensure_ascii=False)  # Write data to JSON file


if __name__ == '__main__':
    main()
