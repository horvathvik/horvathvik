"""
Legacy dbhandler

Handles connection with the KTNL database containing all data
"""
import numpy as np
import sqlite3
import datetime

def generateID(path, db_name):
    con = sqlite3.connect(path + '/' + db_name + ".db")
    cur = con.cursor()
    try:
        ids = cur.execute("SELECT substrateID FROM substrates")
        ids = ids.fetchall()

        ids_all = np.zeros(len(ids), dtype=int)
        for count, identity in enumerate(ids):
            ids_all[count] = identity[0]

        max_id = np.max(ids_all)
        if len(ids) > 0:
            new_identity = max_id + 1
        else:
            new_identity = 0
        new_identity = int(new_identity)
    finally:
        con.close()
    return new_identity


def adapt_array(array):
    values = ""
    for i in range(len(array)):
        values += (str(array[i]) + ";")
    return values


def adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()


def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())


def convert_array(val):
    """convert a str of values separated by ; to ndarray"""
    val = val.decode()
    val = val.split(";")
    res = np.zeros(len(val) - 1)
    for i in range(len(res)):
        res[i] = float(val[i])
    return res


def register_adapters():
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_adapter(datetime.date, adapt_date_iso)
    return 0


def register_converters():
    sqlite3.register_converter("date", convert_date)
    sqlite3.register_converter("array", convert_array)
    return 0


def typecheck(data):
    allowed_types = {int, str, float, datetime.date, np.ndarray}
    for val in data:
        if type(val) not in allowed_types:
            print("Unable to update - " + str(val.type()))
            raise TypeError("Invalid data type encountered")
    return 0


def check_IDexists(path, db_name, identity):
    con = sqlite3.connect(path + '/'  + db_name + ".db")
    cur = con.cursor()
    try:
        # Get IDs - based on susbtrates table
        ids = cur.execute("SELECT substrateID FROM substrates")
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


def add_values(path, db_name, table_name, data):
    register_adapters()

    # Check if all the data is of an expected type
    for data_line in data:
        typecheck(data_line)

    try:
        con = sqlite3.connect(path + '/'  + db_name + ".db")
        cur = con.cursor()

        cur.execute("PRAGMA table_info(" + table_name + ");")
        columns = len(cur.fetchall())  # the number of columns in the selected table
        insert_statement = '?'
        for i in range(columns - 1):
            insert_statement += ', ?'

        if table_name == 'substrates':
            for data_line in data:
                identity = data_line[0]
                exists = check_IDexists(path, db_name, identity)
                if exists:
                    con.close()
                    raise ValueError("Error while adding to substrates. ID "
                                     + str(identity) + " is already in the database.")

                print("Adding new entry to table " + table_name + " with ID " + str(identity))
                command = input("Proceed? y/n \n")

                while command != 'y' and command != 'n':
                    print("Invalid command.")
                    command = input("Proceed? y/n \n")

                if command == 'y':
                    cur.execute("INSERT INTO " + table_name + " VALUES (" + insert_statement + ")", data_line)
                    con.commit()
                elif command == 'n':
                    print("Data not inserted.")


        elif table_name == "ramanSpectra":
            data_batch = []
            identity_set = set()
            for data_line in data:
                identity_set.add(data_line[0])

            for identity in identity_set:
                exists = check_IDexists(path, db_name, identity)
                if not exists:
                    con.close()
                    raise ValueError("Error while adding to ramanSpectra. ID "
                                     + str(identity) + " is not in the database.")
                for data_line in data:
                    if data_line[0] == identity:
                        data_batch.append(data_line)

                print("Adding " + str(len(data_batch)) + " items to ramanSpectra with ID " + str(identity))
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
                    break
        else:
            print("Invalid table name for the KTNL DB.")
    finally:
        con.close()

    return 0


def delete_values(path, db_name, table_name, substrateID=None,
                  material=None, gap=None, gas=None, hidrogenContent=None,
                  flowRate=None, stype=None, depositionTime=None,
                  frequency=None, temperature=None, date=None,
                  analyte=None, concentration=None, integrationTime=None,
                  avg=None, power=None, comment=None, clear_table=False):
    con = sqlite3.connect(path + '/'  + db_name + ".db")
    cur = con.cursor()

    try:
        if type(date) == datetime.date:
            date = date.strftime('%Y-%m-%d')

        filters = [substrateID, material, gap, gas, hidrogenContent, flowRate,
                   stype, depositionTime, frequency, temperature, date, analyte,
                   concentration, integrationTime, avg, power, comment]
        filter_names = ['substrateID', 'material', 'gap', 'gas', 'hidrogenContent',
                        'flowRate', 'stype', 'depositionTime', 'frequency',
                        'temperature', 'date', 'analyte', 'concentration',
                        'integrationTime', 'avg', 'power', 'comment']

        selection_statement = 'DELETE FROM ' + table_name + ' WHERE '

        for index, filter_ in enumerate(filters):
            if filter_ != None:
                if type(filter_) == str:
                    filter_ = "'" + filter_ + "'"
                else:
                    filter_ = str(filter_)
                selection_statement = selection_statement + filter_names[index] + '=' + filter_ + ' AND '
        selection_statement = selection_statement.removesuffix(' AND ')
        selection_statement = selection_statement.removesuffix(' WHERE ')  # if no filters are applied
        selection_statement += ';'

        if selection_statement == 'DELETE FROM ' + table_name + ';':
            print("This will clear the table " + table_name)
            command = input("Proceed? y/n\n")
            while (command != 'y' and command != 'n'):
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
            while (command != 'y' and command != 'n'):
                print("Invalid command.")
                print(command)
                command = input("Proceed? y/n \n")
            if command == 'y':
                cur.execute(selection_statement)
                con.commit()

    finally:
        con.close()
    # TODO - conditional delete
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


def select_from_KTNLdb(path, db_name, table, selected_items, substrateID=None,
                       material=None, gap=None, gas=None, hidrogenContent=None,
                       flowRate=None, stype=None, depositionTime=None,
                       frequency=None, temperature=None, date=None,
                       analyte=None, concentration=None, integrationTime=None,
                       avg=None, power=None, comment=None):
    register_converters()

    if type(date) == datetime.date:
        date = date.strftime('%Y-%m-%d')

    filters = [substrateID, material, gap, gas, hidrogenContent, flowRate,
               stype, depositionTime, frequency, temperature, date, analyte,
               concentration, integrationTime, avg, power, comment]
    filter_names = ['substrateID', 'material', 'gap', 'gas', 'hidrogenContent',
                    'flowRate', 'type', 'depositionTime', 'frequency',
                    'temperature', 'date', 'analyte', 'concentration',
                    'integrationTime', 'avg', 'power', 'comment']

    # Create selection statement and get items from the db
    selection_statements = ['SELECT ']  # Default selection statement [0]

    for item in selected_items:
        if type(item) != str:
            item = str(item)
            raise Warning('Selected item keys must be type str. Other types will be converted to str.')

        if item == 'xData':
            selection_statements.append('SELECT xData as "xData[array]" FROM ')
        elif item == 'yData':
            selection_statements.append('SELECT yData as "yData[array]" FROM ')
        elif item == 'date':
            selection_statements.append('SELECT date as "date[date]" FROM ')
        else:
            selection_statements[0] = selection_statements[0] + item + ', '
    selection_statements[0] = selection_statements[0].removesuffix(', ')

    if len(selected_items) == 0:
        full_selection = True
        selection_statements[0] += ('*')
    else:
        full_selection = False

    selection_statements[0] = selection_statements[0] + ' FROM '
    if len(selection_statements[0]) < 14: selection_statements.pop(0)

    for i, selection_statement in enumerate(selection_statements):
        selection_statement = selection_statement + table + ' WHERE '

        for index, filter_ in enumerate(filters):
            if filter_ != None:
                if type(filter_) == str:
                    filter_ = "'" + filter_ + "'"
                else:
                    filter_ = str(filter_)
                selection_statement = selection_statement + filter_names[index] + '=' + filter_ + ' AND '
        selection_statement = selection_statement.removesuffix(' AND ')
        selection_statement = selection_statement.removesuffix('WHERE ')  # if no filters are applied
        selection_statement += ';'
        selection_statements[i] = selection_statement

    result = []

    try:
        con = sqlite3.connect(path + '/'  + db_name + ".db", detect_types=sqlite3.PARSE_COLNAMES)
        cur = con.cursor()
        for selection_statement in selection_statements:
            cur.execute(str(selection_statement))
            result.append(cur.fetchall())
        con.close()
    except TypeError as inst:
        print('Selection statements: ', selection_statements)
        print(inst)
    except ConnectionError as inst:
        print('Selection statements: ', selection_statements)
        print(inst)
    except RuntimeError as inst:
        print('Selection statements: ', selection_statements)
        print(inst)
    except sqlite3.OperationalError as inst:
        print('Selection statements: ', selection_statements)
        print(inst)

    # Add selected items to a dictionary
    # TODO - format stuff in case of a SELECT * quiery
    result_formatted = []

    try:
        for list_index in range(len(result[0])):

            result_dict = {'substrateID': None, 'material': None, 'gap': None, 'gas': None,
                           'hidrogenContent': None, 'flowRate': None, 'type': None,
                           'depositionTime': None, 'frequency': None, 'temperature': None,
                           'date': None, 'xData': None, 'yData': None, 'analyte': None, 'concentration': None,
                           'integrationTime': None, 'avg': None, 'power': None, 'comment': None}
            keys = list(result_dict)

            if full_selection:
                #TODO all results are str (even dates and ndarrays) - fix this
                if len(selected_items) == 0:
                    try:
                        con = sqlite3.connect(path + '/' + db_name + ".db", detect_types=sqlite3.PARSE_COLNAMES)
                        cur = con.cursor()
                        cur.execute("PRAGMA table_info(" + table + ");")
                        cols = cur.fetchall()
                        con.close()
                    except sqlite3.OperationalError as inst:
                        print(inst)
                    for i in range(len(cols)):
                        selected_items.append(cols[i][1])

                item_index = 0
                for item in selected_items:
                    if item in keys:
                        result_dict[item] = result[0][list_index][item_index]
                        item_index += 1
                    else:
                        raise Warning("\nUnexpected item: " + item)

            else:
                item_index = 0
                for item in selected_items:
                    if type(item) != str:
                        item = str(item)
                        raise Warning('Selected item keys must be type str. Other types will be converted to str.')

                    if item in keys:
                        if item == 'xData':
                            for selection_index, statement in enumerate(selection_statements):
                                if item in statement[:17]:
                                    result_dict[item] = result[selection_index][list_index][0]
                        elif item == 'yData':
                            for selection_index, statement in enumerate(selection_statements):
                                if item in statement[:17]:
                                    result_dict[item] = result[selection_index][list_index][0]
                        elif item == 'date':
                            for selection_index, statement in enumerate(selection_statements):
                                if item in statement[:17]:
                                    result_dict[item] = result[selection_index][list_index][0]
                        else:
                            result_dict[item] = result[0][list_index][item_index]
                            item_index += 1
                    else:
                        raise Warning("\nUnexpected item: " + item)

            result_formatted.append(result_dict)
    except IndexError as inst:
        print('Location: ', path + '/' + db_name + ".db")
        print('Selection statements: ', selection_statements)
        print(inst)

    return result_formatted



