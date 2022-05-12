import os
import json
import urllib


for i in range(3):
    os.system('sudo sysctl -w net.ipv4.ip_forward=1')

    ip_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
    ip = str(ip_data['ip'])

    print(ip)