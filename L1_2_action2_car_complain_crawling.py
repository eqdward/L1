# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:24:23 2020

@author: yy
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent

def get_one_page_content(url):
    """获取url地址网页的内容
       输入：url，网页地址
       输出：soup，返回BeatifulSoup解析后的结果
    """
    ua = UserAgent()
    ua_random = ua.random
    headers = {"User-Agent": ua_random} 
    
    html = requests.get(url, headers=headers, timeout=1000).text   
    soup = BeautifulSoup(html, 'html.parser', 
                       #from_encoding='utf-8'
                       )
    return soup


def get_one_page_complaints(soup):
    """爬取一页中的投诉信息，信息包含字段：投诉编号，投诉品牌，投诉车系，投诉车型，问题简述，典型问题，投诉时间，投诉状态
       输入：soup，get_one_page_content(url)的返回结果
       输出：list格式的投诉信息，一条投诉记录为一个list元素
    """
    table = soup.find('div', class_="tslb_b")

    complaints_list = []
    tr_list = table.find_all('tr')
    for tr in tr_list:
        one_compltaint = []
        td_list = tr.find_all('td')
        for td in td_list:
            one_compltaint.append(td.text)
            #print(td.text)
        complaints_list.append(one_compltaint)
        #print("--"*20)
    complaints_list = complaints_list[1:]

    return complaints_list


def get_pages_complaints(url, page_num=10):
    """爬取若干页的投诉信息
       输入：start_url，起始页
       输入：page_num，指定爬取的页数，取值为int或None，int表示爬取页数，默认为10；None表示爬取所有页
       输出：result，DateFrame格式的爬取结果
    """
    n = 1   # 页面计数器
    result = []   # 存储每一页爬取结果的列表
    while True:
        print("正在爬取第 %d 页" % n)
        soup = get_one_page_content(url)
        result.extend(get_one_page_complaints(soup))
      
        next_page = soup.find('a', text="下一页")   # 检查是否有“下一页”
    
        if page_num is None and next_page is not None:   # 没有限制爬取页数且存在“下一页”时，继续爬取
            url = "http://www.12365auto.com/zlts/" + next_page['href']
            n += 1
        elif page_num is not None and n<page_num:   # 有限制爬取页数但尚未达到页数限制时，继续爬取
            url = "http://www.12365auto.com/zlts/" + next_page['href']
            n += 1
        else:
            break
        
    print("爬取完成！")
    result = pd.DataFrame(data = result, columns = ['id', 'brand', 'car_model', 'type', 'desc', 'problem', 'datetime', 'status'])
    return result


# 网站首页
url = 'http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-1.shtml'
df = get_pages_complaints(url, page_num=20)

df.to_csv(r'C:\Users\lenovo\Desktop\car_complain.csv', index=False,encoding="utf_8_sig")
