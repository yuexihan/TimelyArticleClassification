import requests
import json

def parse(s):
    d = json.loads(s)
    result = d['result']
    retObj = result['retObj']
    realdoc = retObj['realdoc']
    return [x['docid'] for x in realdoc]


if __name__ == '__main__':
    data = {
        'host': '10.185.11.212',
        'key': 'new_timely_ysz',
        'cacheId': 0,
        'pageSize': 100,
    }

    url = 'http://kandian.qq.com/qz_kandian_ext/kandian_ext/GetTimelyDocJson'

    f = open('positive.txt', 'w')
    i = 1
    while True:
        data['pageNum'] = i
        resp = requests.get(url, params=data)
        try:
            innerIds = parse(resp.content)
        except Exception:
            print i, resp.content
        if not innerIds:
            break
        for id in innerIds:
            f.write(id + '\n')
        print i, 'ok'
        i += 1
