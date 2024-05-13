# coding=utf-8

import pathlib
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import TimeoutException

"""
先让Chrome浏览器进入调试状态的终端命令：
chrome.exe --remote-debugging-port=9222
查询端口：
netstat -ano|findstr "9222"
"""
# 获取项目位置
scriptDirectory = pathlib.Path().absolute()

# 此处端口保持和命令行启动的端口一致
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "localhost:9222")
# 参数：--user-data-dir='F:/pythonProject/Product_Review_Analysis/data/userinfo/tb'
chrome_options.add_argument(f"user-data-dir={scriptDirectory}\\data\\userinfo\\tb")  # 实际脚本的目录下的userdata
# 让Selenium接管调试状态的Chrome浏览器
driver = Chrome(options=chrome_options)
# 等待特定条件的出现
wait = WebDriverWait(driver, 10)


# 模拟淘宝登录
def login_taobao():
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
        driver.get(login_url)
        # 根据元素ID获取输入框，并且输入数据
        input_login_id = wait.until(EC.presence_of_element_located((By.ID, 'fm-login-id')))
        input_login_password = wait.until(EC.presence_of_element_located((By.ID, 'fm-login-password')))
        # 清空输入框
        input_login_id.clear()
        input_login_password.clear()
        input_login_id.send_keys('tb16xxxxx')  # 用你自己的淘宝账号替换
        input_login_password.send_keys('gdufexxxxx')  # 用你自己的密码替换
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.fm-button.fm-submit.password-login')))
        submit.click()
        # is_loging = wait.until(EC.url_changes(login_url))
        # return is_loging
    except TimeoutException:
        print('login_taobao TimeoutException!!!!!!!')
        # 减少ID条件尝试
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.fm-button.fm-submit')))
        submit.click()
    finally:
        isLoging = wait.until(EC.url_changes(login_url))
        if isLoging is False:
            isLoging = login_taobao()
        return isLoging


if __name__ == '__main__':
    isLoging1 = login_taobao()
    if isLoging1:
        print('已经登录')
