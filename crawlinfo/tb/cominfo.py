# coding=utf-8
"""
开发日志：
:issue: 登录界面的各种变化，无法实现全自动解放双手
:issue: 网页版只能查看部分评论，能否通过开发者工具中模拟安卓页面进行爬取
:issue: 爬取到一定情况下会弹出一个互动验证

:suggest: 这个模块可以封装的更好

:bug: liushen-6901294179165.jpg的爬取中，爬取了48个商品，有三个广告商品
:speculate:
:solve:
:bug: yida-6923450656181.jpg的爬取中，在45个商品数据当中爬取了46个商品，有一个多出来的商品并没存在于网页当中，但是商品信息与目标查商品信息匹配度为百分之30~40
:speculate:
:solve:
"""
import csv
import os
import pathlib
import re
from time import sleep

import pandas as pd
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import TimeoutException
from urllib.parse import quote
from pyquery import PyQuery as pq


class ComInfo:
    """
    先让Chrome浏览器进入调试状态的终端命令：
    chrome.exe --remote-debugging-port=9222 --incognito
    查询端口：
    netstat -ano|findstr "9222"
    """

    def __init__(self, barCode="6923450656181", scriptDirectory="F:\\pythonProject\\Product_Review_Analysis",
                 KEYWORD='益达木糖醇薄荷40粒瓶装56g 瓶装', cur_page=1, max_page=5, index_page_time=3):
        self.barCode = barCode
        # 获取项目位置
        self.scriptDirectory = scriptDirectory
        # 此处端口保持和命令行启动的端口一致
        self.chrome_options = Options()
        # 隐身模式
        self.chrome_options.add_argument("--incognito")
        # 设置的端口要和命令行启动的端口一致
        self.chrome_options.add_experimental_option("debuggerAddress", "localhost:9222")
        # 参数：--user-data-dir='F:/pythonProject/Product_Review_Analysis/data/userinfo/tb'
        self.chrome_options.add_argument(
            f"user-data-dir={self.scriptDirectory}\\data\\userinfo\\tb")  # 实际脚本的目录下的userdata
        # 让Selenium接管调试状态的Chrome浏览器
        self.driver = Chrome(options=self.chrome_options)
        # 等待特定条件的出现
        self.wait = WebDriverWait(self.driver, 10)
        """
        益达木糖醇薄荷40粒瓶装56g 瓶装
        六神驱蚊花露水95ml 95ml
        """
        self.KEYWORD = KEYWORD
        # 爬取的页数
        self.cur_page = cur_page
        self.max_page = max_page
        self.index_page_time = index_page_time

    def mkdir(self, path):
        # 去除首位空格
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")
        # 判断路径是否存在
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            os.makedirs(path)
            print(path + ' 创建成功')
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print(path + ' 目录已存在')
            return False

    def selenium_tb(self, option_fun):
        isLoging = False
        isLoging_ans = 0
        while isLoging is False and isLoging_ans < 3:
            isLoging = self.login_taobao()
        if isLoging:
            sleep(self.index_page_time)
            print('已经登录')
            # 新建目录
            self.mkdir("output" + '\\' + self.barCode)
            # 清空爬取商品信息的文件
            products_path = os.path.join(self.scriptDirectory, "output", self.barCode, 'products.csv')
            comments_path = os.path.join(self.scriptDirectory, "output", self.barCode, 'comments.csv')
            # print(products_path)
            # print(comments_path)
            if option_fun == 0:
                # 判断路径是否存在
                products_isExists = os.path.exists(products_path)
                if products_isExists:
                    os.remove(products_path)
                url = 'https://s.taobao.com/search?page=1&q=' + quote(self.KEYWORD) + '&tab=all'
                self.index_page(url, self.cur_page, self.max_page)

                comments_isExists = os.path.exists(comments_path)
                if comments_isExists:
                    os.remove(comments_path)
                self.get_prod_comments()
            else:
                comments_isExists = os.path.exists(comments_path)
                if comments_isExists:
                    os.remove(comments_path)
                self.get_prod_comments()
            print("爬取结束")
        else:
            print("登录失败")

    # 模拟淘宝登录
    def login_taobao(self):
        """
        登录账号
        <input name="fm-login-id" type="text" class="fm-text" id="fm-login-id" tabindex="1" aria-label="账号名/邮箱/手机号" placeholder="账号名/邮箱/手机号" autocapitalize="off" data-spm-anchor-id="a2107.1.0.i0.50c011d9cNS402"></div>
        密码
        <input name="fm-login-password" type="password" class="fm-text" id="fm-login-password" tabindex="2" aria-label="请输入登录密码" placeholder="请输入登录密码" maxlength="40" autocapitalize="off" data-spm-anchor-id="a2107.1.0.i2.50c011d9cNS402"><div class="password-look-btn"><i class="iconfont  icon-eye-close"></i></div></div>
        按钮
        <button type="submit" tabindex="3" class="fm-button fm-submit password-login">登录</button></div>

        :return: is_loging -> bool
        """
        print('开始登录...')
        login_url = 'https://login.taobao.com/member/login.jhtml'
        try:
            self.driver.get(login_url)
            # 切换到用户密码模式登录
            # change_type = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.other-account-login-link')))
            # change_type.click()
            # 根据元素ID获取输入框，并且输入数据
            input_login_id = self.wait.until(EC.presence_of_element_located((By.ID, 'fm-login-id')))
            input_login_password = self.wait.until(EC.presence_of_element_located((By.ID, 'fm-login-password')))
            # 清空输入框
            input_login_id.clear()
            input_login_password.clear()
            input_login_id.send_keys('tb160xxxxxx')  # 用你自己的淘宝账号替换
            input_login_password.send_keys('gdufe212xxx')  # 用你自己的密码替换
            submit = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.fm-button.fm-submit.password-login')))
            submit.click()
            # is_loging = wait.until(EC.url_changes(login_url))
            # return is_loging
        except TimeoutException:
            print('login_taobao TimeoutException!!!!!!!')
            # 减少ID条件尝试
            # submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.fm-button.fm-submit')))
            # submit.click()
        finally:
            isLoging = self.wait.until(EC.url_changes(login_url))
            if isLoging is False:
                isLoging = self.login_taobao()
            return isLoging

    # 保存获取的商品到CSV文件
    def save_to_csv(self, products, path, fieldnames):
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

    # 解析获取商品信息
    def get_products(self):
        """提取商品数据"""
        # 获取全部源码
        html = self.driver.page_source
        # 转化为pyquery对象，解析html
        doc = pq(html)
        # 每个商品含有的属性
        items = doc('.Card--doubleCardWrapper--L2XFE73').items()
        # print(len(doc('.Card--doubleCardWrapper--L2XFE73')))

        # 过滤掉含有 'SalesPoint--iconPic--cVEOTPF' 类的广告元素
        filtered_items = [item for item in items if not item.find('.SalesPoint--iconPic--cVEOTPF')]
        # print(len(filtered_items))

        products = []
        for item in filtered_items:
            product = {
                # 'idx': index,
                'url': item.attr('href'),
                # 'price': item.find('.Price--priceInt--ZlsSi_M').text(),
                # 'realsales': item.find('.Price--realSales--FhTZc7U-cnt').text(),
                'title': item.find('.Title--title--jCOPvpf').text(),
                # 'shop': item.find('.ShopInfo--TextAndPic--yH0AZfx').text(),
                # 'location': item.find('.Price--procity--_7Vt3mX').text()
            }
            products.append(product)
            # print(product)

        # 新建目录
        self.mkdir("output" + '\\' + self.barCode)
        self.save_to_csv(products, os.path.join(self.scriptDirectory, "output", self.barCode, 'products.csv'),
                         fieldnames=['url', 'title'])

    # 爬取商品评价
    def get_prod_comments(self):
        # 新建目录
        self.mkdir("output" + '\\' + self.barCode)
        products = pd.read_csv(os.path.join(self.scriptDirectory, "output", self.barCode, 'products.csv'),
                               encoding="utf-8")
        for item_href in products["url"]:
            try:
                # item_href = product.url  # 得到商品的详情访问页面
                if item_href.find('https:') >= 0:
                    item_url = item_href
                    print(item_url)
                else:
                    item_url = "https:" + item_href
                    # 爬取商品评价
                    # self.get_prod_comments(item_url)
                    sleep(self.index_page_time)  # 3秒延迟

                self.driver.get(item_url)
                print('跳转至详情页.......' + item_url)
                ele = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='Tabs--title--1Ov7S5f ']/span")))
                sleep(self.index_page_time)
                # 向下滚动至目标元素可见
                js = "arguments[0].scrollIntoView();"
                self.driver.execute_script(js, ele)
                print('向下滚动至-宝贝评价-元素可见.......')
                self.driver.execute_script("arguments[0].click();", ele)
                print('点击-宝贝评价.......')
                sleep(self.index_page_time)
                ele_comments = self.driver.find_elements(By.CSS_SELECTOR, ".Comment--content--15w7fKj")
                print('提取宝贝评价信息.......')
                # 定义正则表达式模式，用于匹配评论中是否包含 x天后追评 这种格式
                pattern = re.compile(r'\d+天后追评')
                for ele_comment in ele_comments:
                    comment_text = ele_comment.text.strip()
                    if comment_text != "此用户没有填写评价。":
                        comment_text = pattern.sub('', comment_text)
                        # print(comment_text)
                        self.save_to_csv([{"comment": comment_text}],
                                         os.path.join(self.scriptDirectory, "output", self.barCode, 'comments.csv'),
                                         fieldnames=['comment'])
            except Exception as e:
                print(e)

    # 自动获取商品信息并自动翻页
    def index_page(self, url, cur_page, max_page):
        # 爬取每一页的信息
        print(' 正在爬取：' + url)
        try:
            self.driver.get(url)
            sleep(self.index_page_time)  # 3秒延迟
            # 获取当前page的商品信息
            self.get_products()
            try:
                # 识别到下一页按钮并且等待按钮可以点击
                # self.driver.get(url)
                sleep(self.index_page_time)  # 3秒延迟
                next_page_btn = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//button/span[contains(text(),"下一页")]')))
                sleep(self.index_page_time)  # 3秒延迟
                next_page_btn.click()
                sleep(self.index_page_time)  # 3秒延迟
                do_change = self.wait.until(EC.url_changes(url))

                # 更新参数
                if do_change and cur_page < max_page:
                    new_url = self.driver.current_url
                    cur_page = cur_page + 1
                    self.index_page(new_url, cur_page, max_page)
            except Exception as e:
                print('---Error---{}'.format(e))
        except TimeoutException:
            print('---index_page TimeoutException---')


"""
先让Chrome浏览器进入调试状态的终端命令：
chrome.exe --remote-debugging-port=9222 --incognito
查询端口：
netstat -ano|findstr "9222"
"""
"""
益达木糖醇薄荷40粒瓶装56g 瓶装
六神驱蚊花露水95ml 95ml
"""
# scriptDirectory = "F:\\pythonProject\\Product_Review_Analysis"
# barCode = "6923450656181"
# KEYWORD = '益达木糖醇薄荷40粒瓶装56g 瓶装'
# cur_page = 1
# max_page = 5
# index_page_time = 3
# selenium_tb_tor = ComInfo(barCode=barCode, scriptDirectory=scriptDirectory, KEYWORD=KEYWORD, cur_page=cur_page,
#                           max_page=max_page, index_page_time=index_page_time)
