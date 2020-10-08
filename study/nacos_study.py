import nacos

SERVER_ADDRESSES = "192.168.1.102:8848"
NAMESPACE = "***"

# no auth mode
client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, username="nacos", password="nacos")
# auth mode
# client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, username="nacos", password="nacos")

# get config
data_id = "config.nacos"
group = "group"
print(client.get_config(data_id, group))
