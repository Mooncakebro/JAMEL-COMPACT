# App Split

JAMEL uses the paper split over 96 ScaleWoB browser apps:

- `train86`: 86 apps used for data collection and training.
- `test10`: 10 held-out apps used for the main evaluation.

The machine-readable split is in `configs/benchmark_apps.json`.

## test10

```text
vipshop alibaba expedia taobao pinduoduo dongchedi youku keep meituan temu
```

## train86

```text
12306 BestBuy Booking CNN Canva Feishu_D Gmail Notion_D Pinterest Quora
Spotify_D WhatsApp X agoda airbnb airchina alipay amap baicizhan bilibili
cainiao ccb chinamobile crunchyroll csdn damai dewu dianping dingtalk douban
douyin ebay fanqie feishu googlebooks googlefit googlemaps googlenews hemafresh
huazhu instagram iqiyi jd kfc kuwomusic mangotv mcdonalds mijia mishop mubu
neteasemusic qishui qq qqmail qqmusic quanminkge reddit shein slack snapchat
spotify sunlogin telegram tencentdocs tencentmeeting tencentvideo tieba toutiao
trip twitch uber umeit walmart wechat weibo wikipedia wps xianyu xiaoheihe
xiaohongshu ximalaya yangshipin youdao youtubemusic zhihu zol
```

