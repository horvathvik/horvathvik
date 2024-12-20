"""
Handles the preprocessing of data.
Returns a database ready for training a model
"""
#TODO refactor with better naming conventions
#TODO refactor the resampling method in the Spectrum class (include nonlinear interpolation)
#TODO retain more of the connection between the measured spectrum and the training data (at least the raman shifts)

import os
import iosum.dbhandler
import iosum.dbhandlerleg
import random
from iosum import dbhandler, dbhandlerleg
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline as CubicSpline

from global_vars import FULL_DB_PATH, FIGURE_DB_PATH, TRAIN_DB_PATH, TRAIN_DB_NAME
from spectra.Spectrum import RamanSpectrum as RamanSpectrum


#deprecacted - used as reference for refactor
def resample(original_x, original_y, desired_lenght):
    new_x = np.linspace(original_x[0], original_x[-1], desired_lenght)
    cs = CubicSpline(original_x, original_y)
    new_y = cs(new_x)

    return new_x, new_y

def create_figure(x_data, y_data, folder, name):
    plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel('Raman-shift (1/cm)')
    plt.ylabel('Intensity (au.)')
    if not os.path.exists("figures/"+folder):
        os.makedirs("figures/"+folder)
    plt.savefig("figures/"+folder+'/'+name+".png")
    plt.close()
    return 0

def create_RamanSpectrum(get_data = False, path=None, db_name=None, index = None, substrateID = None, date=None,
                         analyte=None, concentration=None, integrationTime=None,
                         avg=None, power=None, comment=None, xData = None, yData = None):
    if get_data:
        spectrum_data = dbhandlerleg.select_from_KTNLdb(path, db_name, 'ramanSpectra',
                                                     ['xData', 'yData', 'date', 'analyte',
                                                      'concentration', 'power', 'integrationTime',
                                                      'avg'], substrateID=substrateID, date=date,
                                                     analyte=analyte, concentration=concentration,
                                                     integrationTime=integrationTime, avg=avg,
                                                     power=power, comment=comment)[index]
    else:
        spectrum_data = {'xData': xData, 'yData': yData, 'date':date, 'analyte':analyte, 'concentration':concentration,
                         'power':power, 'integrationTime':integrationTime, 'avg':avg}

    raman_object = RamanSpectrum(spectrum_data['xData'],
                                 spectrum_data['yData'], spectrum_data['date'],
                                 spectrum_data['analyte'],
                                 spectrum_data['concentration'],
                                 spectrum_data['power'],
                                 spectrum_data['integrationTime'],
                                 spectrum_data['avg'])

    return raman_object

#Get the data used for training and testing - any restrictions are considered before building the db
thiram_1000uM = dbhandlerleg.select_from_KTNLdb(FULL_DB_PATH, 'sers-KTNL_v2', 'RamanSpectra',
                                         ['xData', 'yData'], analyte='thiram', concentration=1, substrateID=102)

thiram_100uM = dbhandlerleg.select_from_KTNLdb(FULL_DB_PATH, 'sers-KTNL_v2', 'RamanSpectra',
                                         ['xData', 'yData'], analyte='thiram', concentration=0.1, substrateID=102)

thiram_10uM = dbhandlerleg.select_from_KTNLdb(FULL_DB_PATH, 'sers-KTNL_v2', 'RamanSpectra',
                                         ['xData', 'yData'], analyte='thiram', concentration=0.01, substrateID=102)

none_all = dbhandlerleg.select_from_KTNLdb(FULL_DB_PATH, 'sers-KTNL_v2', 'RamanSpectra',
                                         ['xData', 'yData'], analyte='none', substrateID=102)

water_all = dbhandlerleg.select_from_KTNLdb(FULL_DB_PATH, 'sers-KTNL_v2', 'RamanSpectra',
                                         ['xData', 'yData'], analyte='water', substrateID=102)

#these are for checkking the time between the susbtrate fabrication and the raman measurements
date_substrate = dbhandlerleg.select_from_KTNLdb(FULL_DB_PATH,'sers-KTNL_v2','substrates',
                                                 ['date'], substrateID=102)
dates = dbhandlerleg.select_from_KTNLdb(FULL_DB_PATH,'sers-KTNL_v2','RamanSpectra',
                                        ['date'], analyte='water', substrateID=102)


#Randomize order of the categories
random.shuffle(thiram_1000uM)
random.shuffle(thiram_100uM)
random.shuffle(thiram_10uM)
random.shuffle(none_all)
random.shuffle(water_all)

print(len(thiram_1000uM))
print(type(thiram_1000uM))
print(len(thiram_100uM))
print(len(thiram_10uM))
print(len(none_all))
print(len(water_all))


#Apply pre-processing and assign labels
data_x = []
data_y = []
labels = []
spectrum_size = len(thiram_1000uM[-1]['xData']) #size for resampling



#Label 2
for i in range(50):
    thiram_spectrum = create_RamanSpectrum(xData=thiram_1000uM[i]['xData'], yData=thiram_1000uM[i]['yData'])
    thiram_spectrum.crop(300, 3000)
    #thiram_spectrum.normalize_area()
    spectrum_size = len(thiram_spectrum.xData) #ezt lehetne a cikluson kívül, hogy biztos egyenlő legyen mindig
    thiram_spectrum.resample(np.linspace(thiram_spectrum.xData[0], thiram_spectrum.xData[-1], spectrum_size))
    create_figure(thiram_spectrum.xData, thiram_spectrum.yData, FIGURE_DB_PATH+'/label2',str(i+1))
    data_x.append(thiram_spectrum.xData)
    data_y.append(thiram_spectrum.yData)
    labels.append(2)

#Label 1
for i in range(50):
    thiram_spectrum = create_RamanSpectrum(xData=thiram_100uM[i]['xData'], yData=thiram_100uM[i]['yData'])
    thiram_spectrum.crop(300, 3000)
    #thiram_spectrum.normalize_area()
    thiram_spectrum.resample(np.linspace(thiram_spectrum.xData[0], thiram_spectrum.xData[-1], spectrum_size))
    create_figure(thiram_spectrum.xData, thiram_spectrum.yData, FIGURE_DB_PATH+'/label1', '24ppm'+str(i + 1))
    data_x.append(thiram_spectrum.xData)
    data_y.append(thiram_spectrum.yData)
    labels.append(1)

#Label 0
for i in range(15):
    thiram_spectrum = create_RamanSpectrum(xData=thiram_10uM[i]['xData'], yData=thiram_10uM[i]['yData'])
    thiram_spectrum.crop(300, 3000)
    #thiram_spectrum.normalize_area()
    thiram_spectrum.resample(np.linspace(thiram_spectrum.xData[0], thiram_spectrum.xData[-1], spectrum_size))
    create_figure(thiram_spectrum.xData, thiram_spectrum.yData, FIGURE_DB_PATH+'/label0', '2.4ppm'+str(i + 1))
    data_x.append(thiram_spectrum.xData)
    data_y.append(thiram_spectrum.yData)
    labels.append(0)
for i in range(20):
    none_spectrum = create_RamanSpectrum(xData=none_all[i]['xData'], yData=none_all[i]['yData'])
    none_spectrum.crop(300, 3000)
    #none_spectrum.normalize_area()
    none_spectrum.resample(np.linspace(none_spectrum.xData[0], none_spectrum.xData[-1], spectrum_size))
    create_figure(none_spectrum.xData, none_spectrum.yData, FIGURE_DB_PATH+'/label0', 'none'+str(i + 1))
    data_x.append(none_spectrum.xData)
    data_y.append(none_spectrum.yData)
    labels.append(0)
for i in range(15):
    water_spectrum = create_RamanSpectrum(xData=water_all[i]['xData'], yData=water_all[i]['yData'])
    water_spectrum.crop(300, 3000)
    #water_spectrum.normalize_area()
    water_spectrum.resample(np.linspace(water_spectrum.xData[0], water_spectrum.xData[-1], spectrum_size))
    create_figure(water_spectrum.xData, water_spectrum.yData, FIGURE_DB_PATH+'/label0', 'water'+str(i + 1))
    data_x.append(water_spectrum.xData)
    data_y.append(water_spectrum.yData)
    labels.append(0)



#Create output - database
NAME_TABLES = ['spectra']
dbhandler.create_db(TRAIN_DB_PATH, TRAIN_DB_NAME, NAME_TABLES, [['ID','x_data', 'y_data', 'label']])

data_toadd = []
id_db = iosum.dbhandler.generate_id(TRAIN_DB_PATH, TRAIN_DB_NAME)
for i in range(len(data_x)):
    data_toadd.append((id_db, data_x[i], data_y[i], labels[i]))
    id_db += 1
dbhandler.add_values_batch(TRAIN_DB_PATH, TRAIN_DB_NAME, NAME_TABLES[0],data_toadd)
