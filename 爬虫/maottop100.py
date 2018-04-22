import re
from multiprocessing import Pool   #多线程
import requests
from requests.exceptions import RequestException  #添加异常

def get_one_page(url):
    try:
        response=requests.get(url)
        if response.status_code==200:  #判断状态码  如果是200则请求成功
            return response.text  #返回文档信息
        return None
    except RequestException:   #捕捉异常
        return None

def prse_one_page(html):
    rember=[]
    pattern= re.compile('<dd>.*?<i class="board-index board-index-.*?<a href=".*?title="(.*?)"'
                        ' class="image-link" data-act="boarditem-click".*?<p class="star">(.*?)</p>.*?<p class="releasetime">(.*?)</p>',re.S)
    items=re.findall(pattern,html)
    '''
    列表的添加方法
    for intem in items:
        rember.append((intem[0],intem[1].strip()[3:],intem[2]))
    return rember
    '''
    for item in items:
        yield{   #这是一个函数，具体百度
            '电影名：':item[0],
            '主演：':item[1].strip()[3:],
            '上映时间：':item[2].strip()[5:]
        }
def write_to_file(content):
    with open('top100.txt','a',encoding='utf-8') as f:  #规定编码格式 utf-8否则乱码
        f.write(str(content)+'\n')      #字典强制转换为字符串，再加上换行符 \n  换行

def main(offset):
    url="http://maoyan.com/board/4?offset="+str(offset)
    html=get_one_page(url)
    for item in prse_one_page(html):
        print(item)
        write_to_file(item)

if __name__=="__main__":

    pool=Pool()     #多线程
    pool.map(main, [i*10 for i in range(10)])

    '''  #正常方法
     for i in range(10):
        main(i*10)
     '''