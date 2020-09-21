import pymysql
from DBUtils.PooledDB import PooledDB
import configs.db_config as Config
from utils.log_utils import get_logger


class MySQLConnectionPool(object):
    __pool = None
    __instance = None

    def __init__(self):
        self.logger = get_logger()

    def __enter__(self):
        self.conn = self.__get_connection()
        self.cursor = self.conn.cursor()
        self.logger("connect to mysql {}:{}\ndatabase:{}".format(Config.DB_TEST_HOST, Config.DB_TEST_PORT,
                                                                 Config.DB_TEST_DBNAME))
        return self

    def __exit__(self):
        """
        close connect
        """
        self.close()

    def get_conn(self):
        """
        从线程池取出一个连接
        :return:cursor, conn
        """
        conn = self.__get_connection()
        cursor = conn.cursor()
        return cursor, conn

    def close(self):
        self.cursor.close()
        self.conn.close()

    @classmethod
    def __get_connection(cls):
        if cls.__pool is None:
            cls.__pool = PooledDB(creator=pymysql, mincached=Config.DB_MIN_CACHED, maxcached=Config.DB_MAX_CACHED,
                                  maxshared=Config.DB_MAX_SHARED, maxconnections=Config.DB_MAX_CONNECYIONS,
                                  blocking=Config.DB_BLOCKING, maxusage=Config.DB_MAX_USAGE,
                                  setsession=Config.DB_SET_SESSION,
                                  host=Config.DB_TEST_HOST, port=Config.DB_TEST_PORT,
                                  user=Config.DB_TEST_USER, passwd=Config.DB_TEST_PASSWORD,
                                  db=Config.DB_TEST_DBNAME, use_unicode=False, charset=Config.DB_CHARSET)
            print("==================================1")
        return cls.__pool.connection()

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = MySQLConnectionPool()
        return cls.__instance


def get_connection():
    return MySQLConnectionPool.get_instance()
