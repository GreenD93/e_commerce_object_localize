import argparse
import multiprocessing

import numpy as np
import json
import glob
import os

import asyncio
import aiohttp
from PIL import Image
from io import BytesIO

META_PATH = 'meta/train/*.json'
DST_FOLDER_PATH = 'test/'

# download img sync
# 코루틴으로 이미지 다운로드 받기
# https://sjquant.tistory.com/14
# await : 병목이 발생해서 다른 작업을 통제권을 넘겨줄 필요가 있는 부분에서는 await을 써줌
# await 뒤에 오는 함수도 코루틴으로 작성되어야 함.

#입력 파라메타 파싱
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_worker', type=int)
    return parser.parse_args()

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

def main(img_urls):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_imgs(img_urls, DST_FOLDER_PATH))
    loop.close()


if __name__ == '__main__':

    # params
    args = get_args()
    n_worker = args.n_worker

    meta_info_paths = glob.glob(META_PATH)

    srcs = []
    # img url parsing
    for meta_info_path in meta_info_paths:
        with open(meta_info_path) as json_file:
            json_data = json.load(json_file)

        img_url = json_data['url']
        srcs.append(img_url)

    # split arr with n chunks
    src_chunks = np.array_split(srcs, n_worker)

    # multi-processing
    pool = multiprocessing.Pool(processes=n_worker)
    pool.starmap(main, zip(src_chunks,))

    pool.close()
    pool.join()

    # 7961 img download
    # 1 : 56초
    # 5 : 16초
    # 10 : 10초