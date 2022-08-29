from PostgreSql import PostgreSql
import numpy as np
import random
import time


def TestPostgreSql():
    mPostgreSql = PostgreSql()
    list_geometry = ['cone', 'cylinder', 'values']

    # # Create new table
    mPostgreSql.CreateTable()

    # Delete all data
    print("Delete all data")
    mPostgreSql.DeleteAll()
    print("***************************************************************************")

    # Insert data
    for i in range(50):
        time1 = time.time()
        random_numpy_array = np.random.rand(64, 64, 64, 3)
        random_geometry = random.choice(tuple(list_geometry))
        time2 = time.time()

        mPostgreSql.InsertPreGeneratedData(random_numpy_array, random_numpy_array)
        time3 = time.time()

        mPostgreSql.InsertInputData(random_geometry, random_numpy_array)
        time4 = time.time()

        # Test time duration
        print(f'Time InsertPreGeneratedData = {time3 - time2} s')
        print(f'Time InsertInputData = {time4 - time3} s')
        print("***************************************************************************")

    # Select all data from PreGeneratedData
    time1 = time.time()
    result = mPostgreSql.SelectAllPreGeneratedData()
    time2 = time.time()
    # print(f'Select all data from PreGeneratedData:\n{result}')
    # Test time duration
    print(f'Time SelectAllPreGeneratedData = {time2 - time1} s')
    print("***************************************************************************")

    # Select all data from InputData
    time3 = time.time()
    result = mPostgreSql.SelectAllInputData()
    time4 = time.time()
    # print(f'Select all data:\n{result}')
    # Test time duration
    print(f'Time SelectAllInputData = {time4 - time3} s')
    print("***************************************************************************")

    # Select random 30 data from PreGeneratedData
    time1 = time.time()
    result = mPostgreSql.SelectRandomPreGeneratedData(30)
    time2 = time.time()
    # print(f'Select random data from PreGeneratedData:\n{result}')
    # Test time duration
    print(f'Time SelectRandomPreGeneratedData = {time2 - time1} s')
    print("***************************************************************************")

    # Select SelectAllGeometryFromInputTable
    time1 = time.time()
    result = mPostgreSql.SelectAllGeometryFromInputTable()
    time2 = time.time()
    # print(f'Select random data from PreGeneratedData:\n{result}')
    # Test time duration
    print(f'Time SelectAllGeometryFromInputTable = {time2 - time1} s')
    print("***************************************************************************")


def main():
    TestPostgreSql()


if __name__ == '__main__':
    main()
