import json
import glob
import os

# img_download
meta_path = 'meta/*.json'

meta_info_paths = glob.glob(meta_path)

img_urls = []

# img url parsing
for meta_info_path in meta_info_paths:
    with open(meta_info_path) as json_file:
        json_data = json.load(json_file)

    img_url = json_data['url']
    img_urls.append(img_url)

# download img sync
# 코루틴으로 이미지 다운로드 받기
# https://sjquant.tistory.com/14
# await : 병목이 발생해서 다른 작업을 통제권을 넘겨줄 필요가 있는 부분에서는 await을 써줌
# await 뒤에 오는 함수도 코루틴으로 작성되어야 함.

import asyncio
import aiohttp
from PIL import Image
from io import BytesIO

dst_folder_path = 'img'

async def download_imgs(img_urls, dst_folder_path):

    async def fetch(sess, url, dst_folder_path):
        try:
            src = url
            img_name = src.split('/')[-1]
            dst = os.path.join(dst_folder_path, img_name)

            async with sess.get(src) as response:
                if response.status != 200:
                    response.raise_for_status()

                byte_data = await response.read()
                im_src = Image.open(BytesIO(byte_data))
                im_src.save(dst)

        except:
            pass

    async def fetch_all(sess, url_list, dst_folder_path):
        # await : 코루틴을 실행하는 예약어.
        await asyncio.gather(*[asyncio.create_task(fetch(sess, url, dst_folder_path)) for url in url_list])

    async with aiohttp.ClientSession() as sess:
        await fetch_all(sess, img_urls, dst_folder_path)

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_imgs(img_urls, dst_folder_path))
    loop.close()

if __name__ == '__main__':
    main()