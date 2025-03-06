import os

PROXIES = {
    'HTTP_PROXY': 'http://httpproxy-tcop.vip.ebay.com:80',
    'HTTPS_PROXY': 'http://httpproxy-tcop.vip.ebay.com:80',
    'http_proxy': 'http://httpproxy-tcop.vip.ebay.com:80',
    'https_proxy': 'http://httpproxy-tcop.vip.ebay.com:80',
    'NO_PROXY': 'krylov,ams,ems,mms,localhost,127.0.0.1,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.krylov-prod.svc,oauth.stratus.ebay.com',
    'no_proxy': 'krylov,ams,ems,mms,localhost,127.0.0.1,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.krylov-prod.svc,oauth.stratus.ebay.com'
}


def export_proxies():
    for key, value in PROXIES.items():
        os.environ[key] = value
