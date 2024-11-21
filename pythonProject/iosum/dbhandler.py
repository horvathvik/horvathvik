"""
DB interactions specific to project

DL db:
thiram_sers.db
TABLES: spectra
COLUMNS: ID, x_data, y_data, analyte

Data is pulled from the KTNL SERS v2 database
"""

from types import NoneType
import numpy as np
import sqlite3
import datetime
import os

# con=sqlite3.connect("sers.db")
# cur=con.cursor()
# cur.execute("CREATE TABLE substrates(substrateID,material,gap,gas,hidrogenContent,flowRate,type,depositionTime,frequency,temperature,date)")
# con.commit()
# cur.execute("CREATE TABLE ramanSpectra(substrateID,xData,yData,analyte,concentration,integrationTime,avg,power,date)")
# con.commit()
# con.close()


def create_db(path, name, tables, columns):
    """
    :param path: str, relative path to the database
    :param name: str, name of the database
    :param tables: list of strings, names of the tables
    :param columns: nested list of strings, names of the columns for each table
    :return: sqlite3 database
    """

    if len(tables) != len(columns):
        raise Exception("Tables and columns should have the same lenght.")
    else:
        try:
            con = sqlite3.connect(path+"/"+name+".db")
        except ConnectionError as err:
            print("Connection couldn't be made. Saving unsuccessful.")
            err.with_traceback()
            print(err)
            return 0
        except sqlite3.OperationalError as err:
            print("Path invalid. Creating directory.")
            if not os.path.exists(path):
                os.makedirs(path)
            con = sqlite3.connect(path+"/"+name+".db")

        tables_present = get_tables(path,name)
        if len(tables_present) > 0:
            print("Database already exist. Proceeding will override its contents.")
            command = input("Proceed? y/n \n")
            while command != 'y' and command != 'n':
                print("Invalid command.")
                command = input("Proceed? y/n \n")
            if command == 'y':
                print("Proceeding.")
                try:
                    cur = con.cursor()
                    for table in tables_present:
                        table = table[0]
                        cur.execute("DROP TABLE "+table+";")
                    con.commit()
                    cur.close()
                except SyntaxError as err:
                    err.with_traceback()
                    print(err)
            elif command == 'n':
                print("Command terminated.")
                con.close()
                return 0

        try:
            cur = con.cursor()
            for i,table in enumerate(tables):
                cols_statement = ""
                for col in columns[i]:
                    cols_statement += (col+",")
                cols_statement = cols_statement.removesuffix(",")
                cur.execute("CREATE TABLE "+table+"("+cols_statement+")")
                con.commit()
        except SyntaxError as err:
            err.with_traceback()
            print("Saving unsuccessful.")
            print(err)
        except TypeError as err:
            print("Saving unsuccessful.")
            err.with_traceback()
            print(err)
        finally:
            con.close()

    return 0

def generate_id(path, db_name):
    reftable = get_tables(path, db_name)[0][0]

    con = sqlite3.connect(path + '/'  + db_name + ".db")
    cur = con.cursor()
    try:
        ids = cur.execute("SELECT ID FROM "+ reftable + ";")
        ids = ids.fetchall()

        if len(ids)>0:
            ids_all = np.zeros(len(ids), dtype=int)
            for count, identity in enumerate(ids):
                ids_all[count] = identity[0]

            max_id = np.max(ids_all)
            new_identity = max_id + 1
            new_identity = int(new_identity)
        else:
            new_identity = 0
    finally:
        con.close()
    return new_identity


def adapt_array(array):
    values = ""
    for i in range(len(array)):
        values += (str(array[i]) + ";")
    return values


def convert_array(val):
    """convert a str of values separated by ; to ndarray"""
    #val = val.decode()
    val = val.split(";")
    res = np.zeros(len(val) - 1)
    for i in range(len(res)):
        res[i] = float(val[i])
    return res


def register_adapters():
    sqlite3.register_adapter(np.ndarray, adapt_array)
    return 0


def register_converters():
    sqlite3.register_converter("array", convert_array)
    return 0

def get_tables(path, db_name):
    tables = ''
    try:
        con = sqlite3.connect(path+"/"+db_name+'.db')
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%';")
        tables = cur.fetchall()
    except ConnectionError as inst:
        print(inst)
        print("\nInvalid connection: "+path+db_name)
    finally:
        con.close()
    return tables

def get_cols(path, db_name, table_name):
    cols = []
    try:
        con = sqlite3.connect(path + "/" + db_name + '.db')
        cur = con.cursor()
        cur.execute("PRAGMA table_info(" + table_name + ");")
        cols = cur.fetchall()
    except ConnectionError as inst:
        print(inst)
        print("\nInvalid connection: " + path + "/" + db_name)
    finally:
        con.close()
    return cols

def check_type(data):
    allowed_types = {int, str, float, datetime.date, np.ndarray}
    for val in data:
        if type(val) not in allowed_types:
            print("Unable to update - " + str(type(val)))
            raise TypeError("Invalid data type encountered")

    return 0


def check_id_exists(path, db_name, identity):
    reftable = get_tables(path, db_name)[0][0]

    con = sqlite3.connect(path + '/'  + db_name + ".db")
    cur = con.cursor()
    try:
        ids = cur.execute("SELECT ID FROM " + reftable + ";")
        ids = ids.fetchall()
        ids_set = set()
        for identity_tuple in ids:
            ids_set.add(identity_tuple[0])
        if identity in ids_set:
            res = True
        else:
            res = False
    finally:
        con.close()

    return res


def add_values_batch(path, db_name, table_name, data):
    register_adapters()

    # Check if all the data is of an expected type
    for data_line in data:
        check_type(data_line)

    con = None
    try:
        con = sqlite3.connect(path + '/'  + db_name + ".db")
        register_adapters()
    except ConnectionError as inst:
        print(inst)
        print("\nInvalid connection: "+path+db_name+table_name)

    if con is not NoneType:
        try:
            cur = con.cursor()

            cur.execute("PRAGMA table_info(" + table_name + ");")
            columns = len(cur.fetchall())  # the number of columns in the selected table
            insert_statement = '?'
            for i in range(columns - 1):
                insert_statement += ', ?'

            data_batch = []
            identity_set = set()
            for data_line in data:
                identity_set.add(data_line[0])
                data_batch.append(data_line)

            for identity in identity_set:
                exists = check_id_exists(path, db_name, identity)
                if exists:
                    con.close()
                    raise ValueError("Error while adding to database. ID " + str(identity) + " is already in the database.")

            print("Adding " + str(len(data_batch)) + " items to " + str(table_name))
            command = input("Proceed? y/n \n")
            while command != 'y' and command != 'n':
                print("Invalid command.")
                command = input("Proceed? y/n \n")
            if command == 'y':
                cur.executemany("INSERT INTO " + table_name + " VALUES (" + insert_statement + ")", data_batch)
                con.commit()
                data_batch.clear()
            elif command == 'n':
                data_batch.clear()
        finally:
            con.close()

    return 0


def delete_values(path, db_name, table_name, identity=None):
    con = sqlite3.connect(path + '/'  + db_name + ".db")
    cur = con.cursor()

    try:
        if identity is None:
            selection_statement = "DELETE FROM " + table_name + ";"
        else:
            #TODO identity should be a number
            selection_statement = 'DELETE FROM ' + table_name + ' WHERE ID=' + identity + ';'

        if selection_statement == 'DELETE FROM ' + table_name + ';':
            print("This will clear the table " + table_name)
            command = input("Proceed? y/n\n")
            while command != 'y' and command != 'n':
                print("Invalid command.")
                print(command)
                command = input("Proceed? y/n \n")
            if command == 'y':
                cur.execute("DELETE FROM " + table_name + ";")
                con.commit()
                print("Table cleared")

        else:
            print("Deleting elements:\n" + selection_statement)
            command = input("\nProceed? y/n\n")
            while command != 'y' and command != 'n':
                print("Invalid command.")
                print(command)
                command = input("Proceed? y/n \n")
            if command == 'y':
                cur.execute(selection_statement)
                con.commit()

    finally:
        con.close()
    return 0


def select_all(path, db_name, table_name):
    register_converters()

    con = sqlite3.connect(path + '/'  + db_name + ".db")
    cur = con.cursor()
    try:
        cur.execute("SELECT * FROM " + table_name + ";")
        entries = cur.fetchall()
    finally:
        con.close()
    return entries

#TODO refactor this function for more flexibility
def select_values(path, db_name, identity=None, label=None):
    register_converters()
    if identity is not None:
        selection_statement = 'SELECT * FROM spectra WHERE ID=' + str(identity) + ";"
    if label is not None:
        selection_statement = 'SELECT * FROM spectra WHERE label=' + str(label) + ";"
    else:
        selection_statement = 'SELECT * FROM spectra;'
    result = 0

    try:
        con = sqlite3.connect(path + '/'  + db_name + ".db", detect_types=sqlite3.PARSE_COLNAMES)
        cur = con.cursor()
        cur.execute(selection_statement)
        result = cur.fetchall()
        con.close()

    except TypeError as inst:
        print('Selection statements: ', selection_statement)
        print(inst)
    except ConnectionError as inst:
        print('Selection statements: ', selection_statement)
        print(inst)
    except RuntimeError as inst:
        print('Selection statements: ', selection_statement)
        print(inst)
    except sqlite3.OperationalError as inst:
        print('Selection statements: ', selection_statement)
        print(inst)

    except IndexError as inst:
        print('Location: ', path + db_name + ".db")
        print('Selection statements: ', selection_statement)
        print(inst)

    return result


