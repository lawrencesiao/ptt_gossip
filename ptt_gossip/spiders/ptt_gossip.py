# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import scrapy
from datetime import datetime
from ptt_gossip.items import PostItem
from scrapy.http import FormRequest
from bs4 import BeautifulSoup

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class PTTSpider(scrapy.Spider):
    
    name = 'ptt_gossip'
    allowed_domains = ['ptt.cc']
    start_urls = ('https://www.ptt.cc/bbs/Gossiping/index.html', )

    _retries = 0
    MAX_RETRY = 3

    _pages = 21326
    MAX_PAGES = 21000

    def parse(self, response):

        domain = 'https://www.ptt.cc/bbs/Gossiping/'
        if len(response.xpath('//div[@class="over18-notice"]')) > 0:
            if self._retries < PTTSpider.MAX_RETRY:
                self._retries += 1
                logging.warning('retry {} times...'.format(self._retries))
                yield FormRequest.from_response(response,
                                                formdata={'yes': 'yes'},
                                                callback=self.parse)
            else:
                logging.warning('you cannot pass')

        else:
            self._pages -= 1
            print "next pageggggg"
            for href in response.css('.r-ent > div.title > a::attr(href)'):
                url = response.urljoin(href.extract())
                yield scrapy.Request(url, callback=self.parse_post)

            if self._pages > self.MAX_PAGES:
    
                yield scrapy.Request(domain+ 'index' + str(self._pages) + '.html', callback=self.parse)

            else:
                logging.warning('max pages reached')

    def parse_post(self, response):

        res = BeautifulSoup(response.body.encode('utf-8'))
        item = PostItem()
        author = u'作者'
        time = u'時間'
        print "here"
        item['title'] = response.xpath(
            '//meta[@property="og:title"]/@content')[0].extract()

  #      qq = res.findAll("span", { "class" : "article-meta-value" })[0].text
        item['author'] = res.findAll("span", { "class" : "article-meta-value" })[0].text
        datetime_str = res.findAll("span", { "class" : "article-meta-value" })[3].text
        item['date'] = datetime.strptime(datetime_str, '%a %b %d %H:%M:%S %Y')

        item['content'] = response.xpath('//div[@id="main-content"]/text()')[
            0].extract()

        comments = []
        total_score = 0
        for comment in response.xpath('//div[@class="push"]'):
            push_tag = comment.css('span.push-tag::text')[0].extract()
            push_user = comment.css('span.push-userid::text')[0].extract()
            push_content = comment.css('span.push-content::text')[0].extract()

            if u'推' in push_tag:
                score = 1
            elif u'噓' in push_tag:
                score = -1
            else:
                score = 0

            total_score += score

            comments.append({'user': push_user,
                             'content': push_content,
                             'score': score})

        item['comments'] = comments
        item['score'] = total_score
        item['url'] = response.url

        yield item