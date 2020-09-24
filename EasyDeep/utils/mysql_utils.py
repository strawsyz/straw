from utils.db_utils import get_connection
from utils.log_utils import get_logger


class MysqlUtil(object):
    db_utils = None
    tag_2_db_uilts = {}

    def __init__(self):
        self.db = get_connection()
        self.logger = get_logger()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MysqlUtil, cls).__new__(cls, *args, **kwargs)
        return cls.instance

    def close(self, cursor, conn):
        cursor.close()
        conn.close()

    def get_one(self, sql, param=()):
        """
        return a tuple
        if can't get data, then return None
        """
        try:
            cursor, conn = self.execute(sql, param)
            res = cursor.fetchone()
            return res
        except Exception as e:
            self.logger.debug(sql)
            raise e
        finally:
            self.close(cursor, conn)

    def get_all(self, sql='', param=()):
        try:
            cursor, conn = self.execute(sql, param)
            res = cursor.fetchall()
            return res
        except Exception as e:
            self.logger.debug(sql)
            raise e
        finally:
            self.close(cursor, conn)

    def insert(self, sql='', param=()):
        conn = None
        cursor = None
        try:
            cursor, conn = self.execute(sql, param)
            _id = cursor.lastrowid
            conn.commit()
            if _id == 0:
                return True
            return _id
        except Exception as e:
            self.logger.debug(sql)
            self.logger.debug('insert except  {}'.format(e.args))
            if conn is not None:
                conn.rollback()
            raise e
        finally:
            self.close(cursor, conn)

    def insert__multi(self, sql='', param=()):
        cursor, conn = self.db.get_conn()
        try:
            cursor.executemany(sql, param)
            conn.commit()
            self.close(cursor, conn)
            return True
        except Exception as e:
            self.logger.debug(sql)
            self.logger.debug('insert many except   ', e.args)
            conn.rollback()
            self.close(cursor, conn)
            raise e

    def delete(self, sql='', param=()):
        try:
            cursor, conn = self.execute(sql, param)
            self.close(cursor, conn)
            return True
        except Exception as e:
            self.logger.debug(sql)
            self.logger.debug("Exception when delete {}".format(e.args))
            conn.rollback()
            self.close(cursor, conn)
            raise e

    def update(self, sql='', param=()):
        try:
            cursor, conn = self.execute(sql, param)
            self.close(cursor, conn)
            return True
        except Exception as e:
            self.logger.debug(sql)
            self.logger.debug('sql is {}\nparam is {}'.format(sql, param))
            self.logger.debug("Exception when update {}".format(e.args))
            conn.rollback()
            self.close(cursor, conn)
            raise e

    @classmethod
    def get_instance(self):
        if MysqlUtil.mysql is None:
            MysqlUtil.mysql = MysqlUtil()
        return MysqlUtil.mysql

    def execute(self, sql='', param=(), autoclose=False):
        cursor, conn = self.db.get_conn()
        try:
            if param:
                cursor.execute(sql, param)
            else:
                cursor.execute(sql)
            conn.commit()
            if autoclose:
                self.close(cursor, conn)
        except Exception as e:
            self.logger.debug(sql)
            self.logger.debug(param)
            self.logger.debug(e.args)
            raise e
        return cursor, conn

    def execute_multi(self, list=[]):
        cursor, conn = self.db.get_conn()
        try:
            for order in list:
                sql = order['sql']
                param = order['param']
                if param:
                    cursor.execute(sql, param)
                else:
                    cursor.execute(sql)
            conn.commit()
            return True
        except Exception as e:
            self.logger.debug('execute failed========', e.args)
            self.logger.debug(list)
            conn.rollback()
            raise e
        finally:
            self.close(cursor, conn)



def get_db_utils():
    return MysqlUtil()


def get_info_from_sql():
    mysql = MysqlUtil()
    res = mysql.get_all('select name from theme WHERE  state = 0')
    print(res)

