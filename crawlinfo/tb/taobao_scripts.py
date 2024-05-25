import csv
import json
import os
import re

"""
该模块用于拦截浏览器接收api返回的请求，从而获得商品评论信息
"""
def save_to_csv(products, path, fieldnames):
    """将商品数据保存到 CSV 文件"""
    # 判断路径是否存在
    isExists = os.path.exists(path)

    with open(path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 如果文件不存在，则写入表头
        if not isExists:
            writer.writeheader()
        # 写入商品数据
        writer.writerows(products)


"""
mitmdump -s taobao_scripts.py
"""


# 抓取淘宝商品评价信息
def response(flow):
    url = 'https://h5api.m.tmall.com/h5/mtop.alibaba.review.list.for.new.pc.detail/1.0/'
    if flow.request.url.startswith(url):
        text = flow.response.text
        json_data = json.loads(text)
        # print(json_data)
        # 提取评论内容并添加到列表中
        reviews = json_data.get("data", {}).get("module", {}).get("reviewVOList", [])
        pattern = re.compile(r'\d+天后追评')
        for review in reviews:
            comment = review.get("reviewWordContent", "")
            print(comment)
            # for comment in comments:
            if comment and comment != "此用户没有填写评价。":
                comment = pattern.sub('', comment)
                # print(comment)
                # comments.append(comment)
                save_to_csv(products=[{"comment": comment}],
                    path="F:\\pythonProject\\Product_Review_Analysis" + "\\output" + "\\6923450656181" + '\\commentsxx.csv',
                    fieldnames=['comment'])
