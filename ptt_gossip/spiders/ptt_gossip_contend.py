# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import scrapy
from datetime import datetime
from ptt_gossip.items import PostItem
from scrapy.http import FormRequest
from bs4 import BeautifulSoup
import bs4
import sys
from utils import notification
import re

def loopUntilA(text, firstElement,nextATag,flag=True):
    if flag:
        firstElement = firstElement.next
        
    if type(firstElement.next) ==  bs4.element.NavigableString:
        text += firstElement.next
        return loopUntilA(text, firstElement.next,nextATag,flag=False)

    elif type(firstElement.next) ==  bs4.element.Tag:
        if (firstElement.next == nextATag):             
            return text
        else:
        #Using double next to skip the string nodes themselves
            return loopUntilA(text, firstElement.next,nextATag,flag=False)
        


def findNextTag(soup):
    return soup(text=re.compile(u'批踢踢實業坊\(ptt.cc\)'))[0].parent


def getContend(soup):

    firstBigTag = soup.find_all('span', {'class' : 'article-meta-value'})[-1]
    #firstBigTag = soup.find('div', id='main-content')
    nextATag = findNextTag(soup)
    targetString = loopUntilA('', firstBigTag,nextATag)
    return targetString

reload(sys)
sys.setdefaultencoding('utf-8')

class PTTSpiderContend(scrapy.Spider):
    
    name = 'ptt_gossip_contend'
    allowed_domains = ['ptt.cc']
    start_urls = ('https://www.ptt.cc/bbs/Gossiping/index.html', )

    _retries = 0
    MAX_RETRY = 10

    _pages = 18920
    MAX_PAGES = 18900

    _pages_ = _pages
    def parse(self, response):

        domain = 'https://www.ptt.cc/bbs/Gossiping/'
        if len(response.xpath('//div[@class="over18-notice"]')) > 0:
            if self._retries < PTTSpiderContend.MAX_RETRY:
                self._retries += 1
                logging.warning('retry {} times...'.format(self._retries))
                yield FormRequest.from_response(response,
                                                formdata={'yes': 'yes'},
                                                callback=self.parse)
            else:
                logging.warning('you cannot pass')

        else:
            logging.info('now finished' + str(float(self.MAX_PAGES- self._pages)/float(self._pages_- self.MAX_PAGES)*100) + '%')

            self._pages -= 1

            for href in response.css('.r-ent > div.title > a::attr(href)'):
                url = response.urljoin(href.extract())
                yield scrapy.Request(url, callback=self.parse_post)

            if self._pages > self.MAX_PAGES+1:
    
                yield scrapy.Request(domain+ 'index' + str(self._pages) + '.html', callback=self.parse)

            else:
                logging.warning('max pages reached')
                notification.notification(self)

    def parse_post(self, response):

        res = BeautifulSoup(response.body.encode('utf-8'))
        item = PostItem()
        author = u'作者'
        time = u'時間'
        item['title'] = response.xpath(
            '//meta[@property="og:title"]/@content')[0].extract()

  #      qq = res.findAll("span", { "class" : "article-meta-value" })[0].text
        item['author'] = res.findAll("span", { "class" : "article-meta-value" })[0].text
        datetime_str = res.findAll("span", { "class" : "article-meta-value" })[3].text
        item['date'] = datetime.strptime(datetime_str, '%a %b %d %H:%M:%S %Y')

        item['content'] = getContend(res)
        item['url'] = response.url

        yield item
