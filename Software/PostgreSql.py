import psycopg2
import logging as log
import datetime
import numpy as np
import os

DEFINE_LOG = 2  # disable log : 0, show log : 1, write log to file : 2


class PostgreSql():
    def __init__(self):
        self.ConfigLog()
        log.info("PostgreSql Class Constructor")
        self.mDataBase = "postgres"
        self.mUser = "postgres"
        self.mPassword = "postgres"
        self.mHost = "10.0.8.40"
        self.mPort = "54321"
        self.mConn = None
        self.OpenDB()

    def __del__(self):
        log.info("PostgreSql Class Destructor")
        self.CloseDB()

    def ConfigLog(self):
        mDateTime = str(datetime.datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
        mLogPath = f'Log/{mDateTime}.log'
        if not os.path.exists('Log'):
            os.makedirs('Log')

        if DEFINE_LOG == 0:
            log.basicConfig(level=log.INFO)
            log.disable(log.INFO)
        elif DEFINE_LOG == 1:
            log.basicConfig(level=log.INFO)
        else:
            log.basicConfig(filename=mLogPath,
                            filemode='w',
                            format="[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s",
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=log.INFO)

    def OpenDB(self):
        try:
            self.mConn = psycopg2.connect(database=self.mDataBase,
                                          user=self.mUser,
                                          password=self.mPassword,
                                          host=self.mHost,
                                          port=self.mPort)
        except psycopg2.OperationalError as e:
            log.info(f'OpenDB Error:\n{e}')
            return False

        self.mConn.autocommit = True
        log.info('OpenDB Success')
        return True

    def CloseDB(self):
        try:
            self.mConn.close()
        except psycopg2.OperationalError as e:
            log.info(f'CloseDB Error:\n{e}')
            return False

        log.info('CloseDB Success')
        return True

    def CreateTable(self):
        try:
            mCursor = self.mConn.cursor()
            mQuery = "CREATE TABLE IF NOT EXISTS pre_generated_data(id SERIAL PRIMARY KEY, geometry TEXT, magnetic_field TEXT)"
            mCursor.execute(mQuery)
            mQuery = "CREATE TABLE IF NOT EXISTS input(id SERIAL PRIMARY KEY, geometry TEXT, values TEXT)"
            mCursor.execute(mQuery)
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'CreateTable Error:\n{e}')
            return False

        return True

    def InsertPreGeneratedData(self, geometry, magnetic_field):
        try:
            mCursor = self.mConn.cursor()
            mQuery = f'INSERT INTO pre_generated_data(geometry, magnetic_field) VALUES (\'{np.array_str(geometry)}\',\'{np.array_str(magnetic_field)}\')'
            mCursor.execute(mQuery)
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'InsertPreGeneratedData Error:\n{e}')
            return False

        log.info(f'InsertPreGeneratedData Success')
        return True

    def InsertInputData(self, geometry: str, values):
        try:
            mCursor = self.mConn.cursor()
            mQuery = f'INSERT INTO input(geometry, values) VALUES (\'{geometry}\', \'{np.array_str(values)}\')'
            mCursor.execute(mQuery)
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'InsertInputData Error:\n{e}')
            return False

        log.info(f'InsertInputData Success')
        return True

    # ouput: [[np.array_geometry, np.array_magnetic_field],[np.array_geometry, np.array_magnetic_field], ...]
    def SelectAllPreGeneratedData(self):
        mResult = []
        try:
            mCursor = self.mConn.cursor()
            mQuery = "SELECT geometry, magnetic_field FROM pre_generated_data"
            mCursor.execute(mQuery)
            for row in mCursor.fetchall():
                tmp_result = []
                str_geometry = row[0]
                str_magnetic_field = row[1]
                tmp_result.append(np.array(str_geometry))
                tmp_result.append(np.array(str_magnetic_field))
                mResult.append(tmp_result)
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'SelectAllPreGeneratedData Error:\n{e}')
            return mResult

        log.info(f'SelectAllPreGeneratedData Success')
        return mResult

    # ouput: [[np.array_geometry, np.array_magnetic_field],[np.array_geometry, np.array_magnetic_field], ...]
    def SelectRandomPreGeneratedData(self, num_row: int):
        mResult = []
        try:
            mCursor = self.mConn.cursor()
            mQuery = f'SELECT geometry, magnetic_field FROM pre_generated_data ORDER BY random() LIMIT {num_row}'
            mCursor.execute(mQuery)
            for row in mCursor.fetchall():
                tmp_result = []
                str_geometry = row[0]
                str_magnetic_field = row[1]
                tmp_result.append(np.array(str_geometry))
                tmp_result.append(np.array(str_magnetic_field))
                mResult.append(tmp_result)
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'SelectRandomPreGeneratedData Error:\n{e}')
            return mResult

        log.info(f'SelectRandomPreGeneratedData Success')
        return mResult

    # ouput: [[string_geometry, np.array_values],[string_geometry, np.array_values], ...]
    def SelectAllInputData(self):
        mResult = []
        try:
            mCursor = self.mConn.cursor()
            mQuery = "SELECT geometry, values FROM input"
            mCursor.execute(mQuery)
            for row in mCursor.fetchall():
                tmp_result = []
                str_geometry = row[0]
                str_values = row[1]
                tmp_result.append(str_geometry)
                tmp_result.append(np.array(str_values))
                mResult.append(tmp_result)
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'SelectAllData Error:\n{e}')
            return mResult

        log.info(f'SelectAllData Success')
        return mResult

    # ouput: ['values', 'cone', 'cone', 'cone', 'values', ...]
    def SelectAllGeometryFromInputTable(self):
        mResult = []
        try:
            mCursor = self.mConn.cursor()
            mQuery = "SELECT geometry FROM input"
            mCursor.execute(mQuery)
            for row in mCursor.fetchall():
                str_geometry = row[0]
                mResult.append(str_geometry)
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'SelectAllGeometryFromInputTable Error:\n{e}')
            return mResult

        log.info(f'SelectAllGeometryFromInputTable Success')
        return mResult

    def DeleteAll(self):
        try:
            mCursor = self.mConn.cursor()
            mCursor.execute("DELETE FROM pre_generated_data")
            mCursor.execute("DELETE FROM input")
            mCursor.execute("VACUUM FULL")
            self.mConn.commit()
        except psycopg2.OperationalError as e:
            log.info(f'DeleteAll Error:\n{e}')
            return False

        log.info(f'DeleteAll Success')
        return True
