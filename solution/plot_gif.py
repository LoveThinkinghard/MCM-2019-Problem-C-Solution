# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import imageio
import openpyxl as xl
import pandas as pd
#%%
# load NFLIS_Data 
# you can download them from: https://www.comap.com/undergraduate/contests/mcm/contests/2019/problems/
x = xl.load_workbook('./data/MCM_NFLIS_Data.xlsx')
data_sheet = x['Data']

rows = data_sheet.rows
columns = data_sheet.columns

lines = []
for column in columns:
    line = [col.value for col in column]
    lines.append(line)

year = np.array(lines[0][1:])
state_name = np.array(lines[1][1:])
county_name = np.array(lines[2][1:])
state = np.array(lines[3][1:]).astype(int)
county = np.array(lines[4][1:]).astype(int)
state_county = np.array(lines[5][1:]).astype(int)
drug_name = np.array(lines[6][1:])
drug_report = np.array(lines[7][1:]).astype(int)
drug_report_county = np.array(lines[8][1:]).astype(int)
drug_report_state = np.array(lines[9][1:]).astype(int)

t = np.unique(state_county)
drug_tags = np.unique(drug_name)
#%%
# load map_data, you can download them from: https://simplemaps.com/data/us-cities
data = pd.read_csv('./data/uscitiesv1.4.csv')

county_loc = []
for x in t:
    loc = np.array(data[(data['county_fips'] == x)].loc[:, ['lat', 'lng']]).T
    county_loc.append([loc[1].mean(), loc[0].mean()])

county_loc = np.array(county_loc).T

# county BEDFORD CITY, VA 51515 has no data in this file, we add it by hand
# we also find something interesting that 
# you also can't find any data about her in the socio-economic data set after 2013
# and her drug use report number is very low
# maybe that's just because she is a small county
county_loc[0, 369]=-79.52351
county_loc[1, 369]=37.318585

plt.scatter(county_loc[0], county_loc[1])
#%%
# it will take several minutes
m = [21, 39, 42, 51, 54]
for drag in drug_tags:
    s = []
    for n in range(8):
        x = np.zeros(t.size)
        select = np.where((drug_name==drag)&(year==2010+n))
        for i in range(select[0].size):
            x[np.where((t == state_county[select][i]))] = drug_report[select][i]
        s.append(x)
    s = np.array(s)

    scaler = s.max() / 92
    
    for j in range(8):
        plt.clf()
        plt.title(r'{} of {} | max: {}'.format(drag, 2010+j, s.max()))
        
        for n in range(5):
            cut = np.where(((t/1000).astype(int) == m[n]))
            plt.scatter(county_loc[0][cut], county_loc[1][cut], s=np.array(s[j][cut])/scaler)
        plt.legend(['KY', 'OH', 'PA', 'VA', 'WA'])
    
        cut = np.where((s[j]==0))
        plt.scatter(county_loc[0][cut], county_loc[1][cut], s=np.array(s[j][cut])/scaler+0.5, c='w')
        plt.savefig(r'./temp/{}.png'.format(j))
    
    plt.clf()
    plt.title(r'{} of {} | max: {}'.format(drag, 2010+j, s.max()))
    
    for n in range(5):
        cut = np.where(((t/1000).astype(int) == m[n]))
        plt.scatter(county_loc[0][cut], county_loc[1][cut], s=np.array(s[j][cut])/scaler)
    plt.legend(['KY', 'OH', 'PA', 'VA', 'WA'])
    
    cut = np.where((s[j]==0))
    plt.scatter(county_loc[0][cut], county_loc[1][cut], s=np.array(s[j][cut])/scaler+0.5, c='w')
    plt.savefig(r'./temp/{}.png'.format(j+1))
    
    frames = []
    for i in range(9):
        frames.append(imageio.imread(r'./temp/{}.png'.format(i)))
    imageio.mimsave(r'./gifs/{} map.gif'.format(drag.replace('/', '^')), frames, 'GIF', duration = 0.5)

#%%
# below is what you *need* to run 'model1.py'
all_s = []
for drag in drug_tags:
    s = []
    for n in range(8):
        x = np.zeros(t.size)
        select = np.where((drug_name==drag)&(year==2010+n))
        for i in range(select[0].size):
            x[np.where((t == state_county[select][i]))] = drug_report[select][i]
        s.append(x)
    s = np.array(s)
    all_s.append(s)
all_s = np.array(all_s)

cut = np.array([all_s[i].sum()>100 for i in range(69)])
drug_use = drug_tags[cut]
all_s_use = all_s[cut]

np.save(r'./data/drug_use', drug_use)
np.save(r'./data/all_s_use', all_s_use)
np.save(r'./data/county', t)