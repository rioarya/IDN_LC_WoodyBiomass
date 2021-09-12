# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:21:55 2019

@author: raryapratama
"""

#%%

#Step (1): Import Python libraries, set land conversion scenarios general parameters

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
import pandas as pd


#RIL Scenario

##Set parameters
#Parameters for primary forest
initAGB = 233            #source: van Beijma et al. (2018)
initAGB_min = 233-72
initAGB_max = 233 + 72




tf = 201

#Parameters for residue decomposition (Source: De Rosa et al., 2017)
a = 0.082
b = 2.53


#%%

#Step (2_1): C loss from the harvesting/clear cut


#df1 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S1')
df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')
dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')


t = range(0,tf,1)


#c_loss_S1 = df1['C_loss'].values
c_firewood_energy_S2 = df2['Firewood_other_energy_use'].values
c_firewood_energy_E = dfE['Firewood_other_energy_use'].values


#%%

#Step (2_2): C loss from the harvesting/clear cut as wood pellets


dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')

c_pellets_E = dfE['Wood_pellets'].values

#%%

#Step (3): Aboveground biomass (AGB) decomposition


#S2
df = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')

tf = 201

t = np.arange(tf)


def decomp_S2(t,remainAGB_S2):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_S2



#set zero matrix
output_decomp_S2 = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_S2 in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_S2[i:,i] = decomp_S2(t[:len(t)-i],remain_part_S2)

print(output_decomp_S2[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_S2 = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_S2[:,i] = np.diff(output_decomp_S2[:,i])
    i = i + 1 

print(subs_matrix_S2[:,:4])
print(len(subs_matrix_S2))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_S2 = subs_matrix_S2.clip(max=0)

print(subs_matrix_S2[:,:4])

#make the results as absolute values
subs_matrix_S2 = abs(subs_matrix_S2)
print(subs_matrix_S2[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_S2 = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_S2)

subs_matrix_S2 = np.vstack((zero_matrix_S2, subs_matrix_S2))

print(subs_matrix_S2[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_S2 = (tf,1)
decomp_tot_S2 = np.zeros(matrix_tot_S2) 

i = 0
while i < tf:
    decomp_tot_S2[:,0] = decomp_tot_S2[:,0] + subs_matrix_S2[:,i]
    i = i + 1

print(decomp_tot_S2[:,0])




#S2_C
df = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_C_S2')

tf = 201

t = np.arange(tf)


def decomp_S2_C(t,remainAGB_S2_C):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_S2_C



#set zero matrix
output_decomp_S2_C = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_S2_C in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_S2_C[i:,i] = decomp_S2_C(t[:len(t)-i],remain_part_S2_C)

print(output_decomp_S2_C[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_S2_C = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_S2_C[:,i] = np.diff(output_decomp_S2_C[:,i])
    i = i + 1 

print(subs_matrix_S2_C[:,:4])
print(len(subs_matrix_S2_C))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_S2_C = subs_matrix_S2_C.clip(max=0)

print(subs_matrix_S2_C[:,:4])

#make the results as absolute values
subs_matrix_S2_C = abs(subs_matrix_S2_C)
print(subs_matrix_S2_C[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_S2_C = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_S2_C)

subs_matrix_S2_C = np.vstack((zero_matrix_S2_C, subs_matrix_S2_C))

print(subs_matrix_S2_C[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_S2_C = (tf,1)
decomp_tot_S2_C = np.zeros(matrix_tot_S2_C) 

i = 0
while i < tf:
    decomp_tot_S2_C[:,0] = decomp_tot_S2_C[:,0] + subs_matrix_S2_C[:,i]
    i = i + 1

print(decomp_tot_S2_C[:,0])




#E
df = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')

tf = 201

t = np.arange(tf)


def decomp_E(t,remainAGB_E):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_E



#set zero matrix
output_decomp_E = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_E in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_E[i:,i] = decomp_E(t[:len(t)-i],remain_part_E)

print(output_decomp_E[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_E = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_E[:,i] = np.diff(output_decomp_E[:,i])
    i = i + 1 

print(subs_matrix_E[:,:4])
print(len(subs_matrix_E))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_E = subs_matrix_E.clip(max=0)

print(subs_matrix_E[:,:4])

#make the results as absolute values
subs_matrix_E = abs(subs_matrix_E)
print(subs_matrix_E[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_E = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_E)

subs_matrix_E = np.vstack((zero_matrix_E, subs_matrix_E))

print(subs_matrix_E[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_E = (tf,1)
decomp_tot_E = np.zeros(matrix_tot_E) 

i = 0
while i < tf:
    decomp_tot_E[:,0] = decomp_tot_E[:,0] + subs_matrix_E[:,i]
    i = i + 1

print(decomp_tot_E[:,0])



#E_C
df = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_C_E')

tf = 201

t = np.arange(tf)


def decomp_E_C(t,remainAGB_E_C):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_E_C



#set zero matrix
output_decomp_E_C = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_E_C in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_E_C[i:,i] = decomp_E_C(t[:len(t)-i],remain_part_E_C)

print(output_decomp_E_C[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_E_C = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_E_C[:,i] = np.diff(output_decomp_E_C[:,i])
    i = i + 1 

print(subs_matrix_E_C[:,:4])
print(len(subs_matrix_E_C))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_E_C = subs_matrix_E_C.clip(max=0)

print(subs_matrix_E_C[:,:4])

#make the results as absolute values
subs_matrix_E_C = abs(subs_matrix_E_C)
print(subs_matrix_E_C[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_E_C = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_E_C)

subs_matrix_E_C = np.vstack((zero_matrix_E_C, subs_matrix_E_C))

print(subs_matrix_E_C[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_E_C = (tf,1)
decomp_tot_E_C = np.zeros(matrix_tot_E_C) 

i = 0
while i < tf:
    decomp_tot_E_C[:,0] = decomp_tot_E_C[:,0] + subs_matrix_E_C[:,i]
    i = i + 1

print(decomp_tot_E_C[:,0])






#plotting
t = np.arange(0,tf)

#plt.plot(t,decomp_tot_S1,label='S1')
plt.plot(t,decomp_tot_S2,label='S2')
plt.plot(t,decomp_tot_E,label='E')
plt.plot(t,decomp_tot_S2_C,label='S2_C')
plt.plot(t,decomp_tot_E_C,label='E_C')
plt.xlim(0,200)

plt.legend(loc='best', frameon=False)

plt.show()



#%%

#Step (4): Dynamic stock model of in-use wood materials


from dynamic_stock_model import DynamicStockModel



df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')
dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')


#product lifetime
#building materials
B = 50


TestDSM2 = DynamicStockModel(t = df2['Year'].values, i = df2['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})
TestDSME = DynamicStockModel(t = dfE['Year'].values, i = dfE['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})



CheckStr2, ExitFlag2 = TestDSM2.dimension_check()
CheckStrE, ExitFlagE = TestDSME.dimension_check()



Stock_by_cohort2, ExitFlag2 = TestDSM2.compute_s_c_inflow_driven()
Stock_by_cohortE, ExitFlagE = TestDSME.compute_s_c_inflow_driven()



S2, ExitFlag2   = TestDSM2.compute_stock_total()
SE, ExitFlagE   = TestDSME.compute_stock_total()



O_C2, ExitFlag2 = TestDSM2.compute_o_c_from_s_c()
O_CE, ExitFlagE = TestDSME.compute_o_c_from_s_c()



O2, ExitFlag2   = TestDSM2.compute_outflow_total()
OE, ExitFlagE   = TestDSME.compute_outflow_total()


DS2, ExitFlag2  = TestDSM2.compute_stock_change()
DSE, ExitFlagE  = TestDSME.compute_stock_change()



Bal2, ExitFlag2 = TestDSM2.check_stock_balance()
BalE, ExitFlagE = TestDSME.check_stock_balance()




#print output flow
print(TestDSM2.o)
print(TestDSME.o)

plt.xlim(0,100)

plt.show()


#%%

#Step (5): Biomass growth

# RIL_Scenario biomass growth, following RIL disturbance

#recovery time, follow the one by Alice-guier

#H = [M, E, C_M, C_E]
#LD0 = [M, E, C_M, C_E]

H = [5.78, 7.71, 5.78, 7.71]
LD0 = [53.46-5.78, 53.46-7.71, 29.29-5.78, 29.29-7.71]
s = 1.106 


#RIL
RT = ((H[0] + LD0[0])*100/initAGB)**s 

print(RT)


#growth per year (Mg C/ha.yr)
gpy = (H[0] + LD0[0])/RT

print(gpy)


tf_RIL_S1 = 36


A1 = range(0,tf_RIL_S1,1)

#caculate the disturbed natural forest recovery carbon regrowth over time following RIL
def Y_RIL_S1(A1):
    return 44/12*1000*gpy*A1

seq_RIL = np.array([Y_RIL_S1(A1i) for A1i in A1])

print(len(seq_RIL))

print(seq_RIL)

##3 times 35-year cycle of new AGB following logging (RIL)
counter_35y = range(0,6,1)

y_RIL = []

for i in counter_35y:
    y_RIL.append(seq_RIL)


    
flat_list_RIL = []
for sublist in y_RIL:
    for item in sublist:
        flat_list_RIL.append(item)

#the length of the list is now 216, so we remove the last 15 elements of the list to make the len=tf
flat_list_RIL = flat_list_RIL[:len(flat_list_RIL)-15]

print(flat_list_RIL)



#plotting
t = np.arange(0,tf,1)
plt.xlim([0, 200])
plt.plot(t, flat_list_RIL, color='darkviolet')





#yearly sequestration

## RIL (35-year cycle)
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_RIL (https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_RIL = [p - q for q, p in zip(flat_list_RIL, flat_list_RIL[1:])]


#since there is no sequestration between the replanting year (e.g., year 35 to 36), we have to replace negative numbers in 'flat_list_RIL' with 0 values
flat_list_RIL = [0 if i < 0 else i for i in flat_list_RIL]


#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_RIL.insert(0,var)


#make 'flat_list_RIL' elements negative numbers to denote sequestration
flat_list_RIL = [ -x for x in flat_list_RIL]

print(flat_list_RIL)



#RIL_C

RT_C = ((H[2] + LD0[2])*100/initAGB)**s 

print(RT_C)


#growth per year (Mg C/ha.yr)
gpy_C = (H[2] + LD0[2])/RT_C

print(gpy_C)


tf_RIL_C = 36


A1 = range(0,tf_RIL_C,1)

#caculate the disturbed natural forest recovery carbon regrowth over time following RIL
def Y_RIL_C(A1):
    return 44/12*1000*gpy_C*A1

seq_RIL_C = np.array([Y_RIL_C(A1i) for A1i in A1])

print(len(seq_RIL_C))

print(seq_RIL_C)

##3 times 35-year cycle of new AGB following logging (RIL)
counter_35y = range(0,6,1)

y_RIL_C = []

for i in counter_35y:
    y_RIL_C.append(seq_RIL_C)


    
flat_list_RIL_C = []
for sublist_C in y_RIL_C:
    for item in sublist_C:
        flat_list_RIL_C.append(item)

#the length of the list is now 216, so we remove the last 15 elements of the list to make the len=tf
flat_list_RIL_C = flat_list_RIL_C[:len(flat_list_RIL_C)-15]





#plotting
t = np.arange(0,tf,1)
plt.xlim([0, 200])
plt.plot(t, flat_list_RIL_C, color='darkviolet')





#yearly sequestration

## RIL (35-year cycle)
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_RIL (https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_RIL_C = [p - q for q, p in zip(flat_list_RIL_C, flat_list_RIL_C[1:])]


#since there is no sequestration between the replanting year (e.g., year 35 to 36), we have to replace negative numbers in 'flat_list_RIL' with 0 values
flat_list_RIL_C = [0 if i < 0 else i for i in flat_list_RIL_C]


#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_RIL_C.insert(0,var)


#make 'flat_list_RIL' elements negative numbers to denote sequestration
flat_list_RIL_C = [ -x for x in flat_list_RIL_C]

print(flat_list_RIL_C)


#%%
#Step (5_1): Biomass C sequestration of the remaining unharvested block

df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')
df2_C = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_C_S2')
dfE= pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')
dfE_C = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_C_E')



t = range(0,tf,1)



RIL_seq_S2= df2['RIL_seq'].values
RIL_seq_C_S2= df2_C['RIL_seq'].values
RIL_seq_E = dfE['RIL_seq'].values
RIL_seq_C_E = dfE_C['RIL_seq'].values


#%%

#Step (6): post-harvest processing of wood 


#post-harvest wood processing
df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')
dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')


t = range(0,tf,1)

PH_Emissions_HWP2 = df2['PH_Emissions_HWP'].values
PH_Emissions_HWPE = dfE ['PH_Emissions_HWP'].values


#%%

#Step (7_1): landfill gas decomposition (CH4)

#CH4 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl



#S2
df2_CH4 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')

tf = 201

t = np.arange(tf)


def decomp_CH4_S2(t,Landfill_decomp_CH4_S2):
    return (1-(1-np.exp(-k*t)))*Landfill_decomp_CH4_S2



#set zero matrix
output_decomp_CH4_S2 = np.zeros((len(t),len(df2_CH4['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_S2 in enumerate(df2_CH4['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_S2[i:,i] = decomp_CH4_S2(t[:len(t)-i],remain_part_CH4_S2)

print(output_decomp_CH4_S2[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_S2 = np.zeros((len(t)-1,len(df2_CH4['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_S2[:,i] = np.diff(output_decomp_CH4_S2[:,i])
    i = i + 1 

print(subs_matrix_CH4_S2[:,:4])
print(len(subs_matrix_CH4_S2))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_S2 = subs_matrix_CH4_S2.clip(max=0)

print(subs_matrix_CH4_S2[:,:4])

#make the results as absolute values
subs_matrix_CH4_S2 = abs(subs_matrix_CH4_S2)
print(subs_matrix_CH4_S2[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_S2 = np.zeros((len(t)-200,len(df2_CH4['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_S2)

subs_matrix_CH4_S2 = np.vstack((zero_matrix_CH4_S2, subs_matrix_CH4_S2))

print(subs_matrix_CH4_S2[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_S2 = (tf,1)
decomp_tot_CH4_S2 = np.zeros(matrix_tot_CH4_S2) 

i = 0
while i < tf:
    decomp_tot_CH4_S2[:,0] = decomp_tot_CH4_S2[:,0] + subs_matrix_CH4_S2[:,i]
    i = i + 1

print(decomp_tot_CH4_S2[:,0])



#E
dfE_CH4 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')

tf = 201

t = np.arange(tf)


def decomp_CH4_E(t,Landfill_decomp_CH4_E):
    return (1-(1-np.exp(-k*t)))*Landfill_decomp_CH4_E



#set zero matrix
output_decomp_CH4_E = np.zeros((len(t),len(dfE_CH4['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_E in enumerate(dfE_CH4['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_E[i:,i] = decomp_CH4_E(t[:len(t)-i],remain_part_CH4_E)

print(output_decomp_CH4_E[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_E = np.zeros((len(t)-1,len(dfE_CH4['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_E[:,i] = np.diff(output_decomp_CH4_E[:,i])
    i = i + 1 

print(subs_matrix_CH4_E[:,:4])
print(len(subs_matrix_CH4_E))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_E = subs_matrix_CH4_E.clip(max=0)

print(subs_matrix_CH4_E[:,:4])

#make the results as absolute values
subs_matrix_CH4_E = abs(subs_matrix_CH4_E)
print(subs_matrix_CH4_E[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_E = np.zeros((len(t)-200,len(dfE_CH4['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_E)

subs_matrix_CH4_E = np.vstack((zero_matrix_CH4_E, subs_matrix_CH4_E))

print(subs_matrix_CH4_E[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_E = (tf,1)
decomp_tot_CH4_E = np.zeros(matrix_tot_CH4_E) 

i = 0
while i < tf:
    decomp_tot_CH4_E[:,0] = decomp_tot_CH4_E[:,0] + subs_matrix_CH4_E[:,i]
    i = i + 1

print(decomp_tot_CH4_E[:,0])

#plotting
t = np.arange(0,tf)

#plt.plot(t,decomp_tot_CH4_S1,label='S1')
plt.plot(t,decomp_tot_CH4_S2,label='S2')
plt.plot(t,decomp_tot_CH4_E,label='E')


plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()


#%%

#Step (7_2): landfill gas decomposition (CO2)

#CO2 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl


#S2
df2_CO2 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')

tf = 201

t = np.arange(tf)


def decomp_CO2_S2(t,Landfill_decomp_CO2_S2):
    return (1-(1-np.exp(-k*t)))*Landfill_decomp_CO2_S2



#set zero matrix
output_decomp_CO2_S2 = np.zeros((len(t),len(df2_CO2['Landfill_decomp_CO2'].values)))


for i,remain_part_CO2_S2 in enumerate(df2_CO2['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_CO2_S2[i:,i] = decomp_CO2_S2(t[:len(t)-i],remain_part_CO2_S2)

print(output_decomp_CO2_S2[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CO2_S2 = np.zeros((len(t)-1,len(df2_CO2['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_CO2_S2[:,i] = np.diff(output_decomp_CO2_S2[:,i])
    i = i + 1 

print(subs_matrix_CO2_S2[:,:4])
print(len(subs_matrix_CO2_S2))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CO2_S2 = subs_matrix_CO2_S2.clip(max=0)

print(subs_matrix_CO2_S2[:,:4])

#make the results as absolute values
subs_matrix_CO2_S2 = abs(subs_matrix_CO2_S2)
print(subs_matrix_CO2_S2[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CO2_S2 = np.zeros((len(t)-200,len(df2_CO2['Landfill_decomp_CO2'].values)))
print(zero_matrix_CO2_S2)

subs_matrix_CO2_S2 = np.vstack((zero_matrix_CO2_S2, subs_matrix_CO2_S2))

print(subs_matrix_CO2_S2[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CO2_S2 = (tf,1)
decomp_tot_CO2_S2 = np.zeros(matrix_tot_CO2_S2) 

i = 0
while i < tf:
    decomp_tot_CO2_S2[:,0] = decomp_tot_CO2_S2[:,0] + subs_matrix_CO2_S2[:,i]
    i = i + 1

print(decomp_tot_CO2_S2[:,0])



#E
dfE_CO2 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')

tf = 201

t = np.arange(tf)


def decomp_CO2_E(t,Landfill_decomp_CO2_E):
    return (1-(1-np.exp(-k*t)))*Landfill_decomp_CO2_E



#set zero matrix
output_decomp_CO2_E = np.zeros((len(t),len(dfE_CO2['Landfill_decomp_CO2'].values)))


for i,remain_part_CO2_E in enumerate(dfE_CO2['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_CO2_E[i:,i] = decomp_CO2_E(t[:len(t)-i],remain_part_CO2_E)

print(output_decomp_CO2_E[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CO2_E = np.zeros((len(t)-1,len(dfE_CO2['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_CO2_E[:,i] = np.diff(output_decomp_CO2_E[:,i])
    i = i + 1 

print(subs_matrix_CO2_E[:,:4])
print(len(subs_matrix_CO2_E))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CO2_E = subs_matrix_CO2_E.clip(max=0)

print(subs_matrix_CO2_E[:,:4])

#make the results as absolute values
subs_matrix_CO2_E = abs(subs_matrix_CO2_E)
print(subs_matrix_CO2_E[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CO2_E = np.zeros((len(t)-200,len(dfE_CO2['Landfill_decomp_CO2'].values)))
print(zero_matrix_CO2_E)

subs_matrix_CO2_E = np.vstack((zero_matrix_CO2_E, subs_matrix_CO2_E))

print(subs_matrix_CO2_E[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CO2_E = (tf,1)
decomp_tot_CO2_E = np.zeros(matrix_tot_CO2_E) 

i = 0
while i < tf:
    decomp_tot_CO2_E[:,0] = decomp_tot_CO2_E[:,0] + subs_matrix_CO2_E[:,i]
    i = i + 1

print(decomp_tot_CO2_E[:,0])

#plotting
t = np.arange(0,tf)

#plt.plot(t,decomp_tot_CO2_S1,label='S1')
plt.plot(t,decomp_tot_CO2_S2,label='S2')
plt.plot(t,decomp_tot_CO2_E,label='E')


plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()


#%%

#Step (8): Sum the emissions and sequestration (net carbon balance), CO2 and CH4 are separated

#https://stackoverflow.com/questions/52703442/python-sum-values-from-multiple-lists-more-than-two
#C_loss + C_remainAGB + C_remainHWP + PH_Emissions_PO


Emissions_S2 = [c_firewood_energy_S2, decomp_tot_S2[:,0], TestDSM2.o, PH_Emissions_HWP2, decomp_tot_CO2_S2[:,0]]
Emissions_E = [c_firewood_energy_E, c_pellets_E, decomp_tot_E[:,0], TestDSME.o, PH_Emissions_HWPE, decomp_tot_CO2_E[:,0]]
Emissions_S2_C = [c_firewood_energy_S2, decomp_tot_S2_C[:,0], TestDSM2.o, PH_Emissions_HWP2, decomp_tot_CO2_S2[:,0]]
Emissions_E_C = [c_firewood_energy_E, c_pellets_E, decomp_tot_E_C[:,0], TestDSME.o, PH_Emissions_HWPE, decomp_tot_CO2_E[:,0]]




Emissions_RIL_S2 = [sum(x) for x in zip(*Emissions_S2)]
Emissions_RIL_E = [sum(x) for x in zip(*Emissions_E)]
Emissions_RIL_S2_C = [sum(x) for x in zip(*Emissions_S2_C)]
Emissions_RIL_E_C = [sum(x) for x in zip(*Emissions_E_C)]


#CH4_S2
Emissions_CH4_RIL_S2 = decomp_tot_CH4_S2[:,0]


#CH4_E
Emissions_CH4_RIL_E = decomp_tot_CH4_E[:,0]


#%%

#Step (9): Generate the excel file (emissions_seq_scenarios.xlsx) from Step (8) calculation


#print year column
year = []
for x in range (0, tf):
    year.append(x)
print (year)



#print CH4 emission column
import itertools
lst = [0]
Emissions_CH4 = list(itertools.chain.from_iterable(itertools.repeat(x, tf) for x in lst))
print(Emissions_CH4)


#print emission ref 
lst1 = [0]
Emission_ref = list(itertools.chain.from_iterable(itertools.repeat(x, tf) for x in lst1))
print(Emission_ref)



#replace the first element with 1 to denote the emission reference as year 0 (for dynGWP calculation)
Emission_ref[0] = 1
print(Emission_ref)





Col1 = year
#Col2_S1 = Emissions_RIL_S1
Col2_S2 = Emissions_RIL_S2
Col2_E = Emissions_RIL_E
Col2_S2_C = Emissions_RIL_S2_C
Col2_E_C = Emissions_RIL_E_C
#Col3_1 = Emissions_CH4_RIL_S1
Col3_2 = Emissions_CH4_RIL_S2
Col3_E = Emissions_CH4_RIL_E
Col4 = Emission_ref
Col5_2 = [x + y for x, y in zip(flat_list_RIL, RIL_seq_S2)]
Col5_E = [x + y for x, y in zip(flat_list_RIL, RIL_seq_E)]
Col5_C_2 = [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_S2)]
Col5_C_E = [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_E)]


df2 = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_S2,'kg_CH4':Col3_2,'kg_CO2_seq':Col5_2,'emission_ref':Col4})
dfE = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_E,'kg_CH4':Col3_E,'kg_CO2_seq':Col5_E,'emission_ref':Col4})
df2_C = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_S2_C,'kg_CH4':Col3_2,'kg_CO2_seq':Col5_C_2,'emission_ref':Col4})
dfE_C = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_E_C,'kg_CH4':Col3_E,'kg_CO2_seq':Col5_C_E,'emission_ref':Col4})

writer = pd.ExcelWriter('emissions_seq_RIL_Ret.xlsx', engine = 'xlsxwriter')

df2.to_excel(writer, sheet_name = 'RIL_S2', header=True, index=False)
dfE.to_excel(writer, sheet_name = 'RIL_E', header=True, index=False)
df2_C.to_excel(writer, sheet_name = 'RIL_C_S2', header=True, index=False)
dfE_C.to_excel(writer, sheet_name = 'RIL_C_E', header=True, index=False)


writer.save()
writer.close()


#%%

## DYNAMIC LCA    (wood-based scenarios)

# Step (10): Set General Parameters for Dynamic LCA calculation

aCH4 = 0.129957e-12;    # methane - instantaneous radiative forcing per unit mass [W/m2 /kgCH4]
TauCH4 = 12;    # methane - lifetime (years)
aCO2 = 0.0018088e-12;    # CO2 - instantaneous radiative forcing per unit mass [W/m2 /kgCO2]
TauCO2 = [172.9,  18.51,  1.186];    # CO2 parameters according to Bern carbon cycle-climate model
aBern = [0.259, 0.338, 0.186];        # CO2 parameters according to Bern carbon cycle-climate model
a0Bern = 0.217;                     # CO2 parameters according to Bern carbon cycle-climate model
tf = 202                           #until 202 because we want to get the DCF(t-i) until DCF(201) to determine the impact from the emission from the year 200 (There is no DCF(0))


#%%

#Step (11): Bern 2.5 CC Model, determine atmospheric load (C(t)) for GHG (CO2 and CH4)


t = range(0,tf,1)


## CO2 calculation formula
# time dependant atmospheric load for CO2, Bern model
def C_CO2(t):
    return a0Bern + aBern[0]*np.exp(-t/TauCO2[0]) + aBern[1]*np.exp(-t/TauCO2[1]) + aBern[2]*np.exp(-t/TauCO2[2])

output_CO2 = np.array([C_CO2(ti) for ti in t])

print(output_CO2)


## CH4 calculation formula
# time dependant atmospheric load for non-CO2 GHGs (Methane)
def C_CH4(t):
    return np.exp(-t/TauCH4)

output_CH4 = np.array([C_CH4(ti) for ti in t])




plt.xlim([0, 200])
plt.ylim([0,1.1])

plt.plot(t, output_CO2, output_CH4)


plt.xlabel('Time (year)')
plt.ylabel('Fraction of CO$_2$')

plt.show()


output_CH4.size
#%%

#determine the C(t) for CO2
s = []

t = np.arange(0,tf,1)

for i in t:
    s.append(quad(C_CO2,i-1,i))
    
res_list_CO2 = [x[0] for x in s]

len(res_list_CO2)

#%%

#determine the C(t) for CH4
s = []

for i in t:
    s.append(quad(C_CH4,i-1,i))

res_list_CH4 = [p[0] for p in s]


#plot
plt.xlim([0, 200])
plt.ylim([0,1.5])

 
plt.plot(t, res_list_CO2, res_list_CH4)
plt.show()

#%%

#Step (12): Determine dynamic characterization factors (DCF) for CO2 and CH4

DCF_inst_CO2 = aCO2 * np.array(res_list_CO2)


print(DCF_inst_CO2)


DCF_inst_CH4 = aCH4 * np.array(res_list_CH4)


plt.xlim([0, 200])
plt.ylim([0,4e-15])


plt.plot(t, DCF_inst_CO2, DCF_inst_CH4)
plt.xlabel('Time (year)')
plt.ylabel('DCF_inst (10$^{-15}$ W/m$^2$.kg CO$_2$)')
plt.show()

len(DCF_inst_CO2)

#%%

#Step (13): import emission data from emissions_seq_scenarios.xlsx (Step (9))

 
##read wood-based data
#read S2
df = pd.read_excel('emissions_seq_RIL_Ret.xlsx', 'RIL_S2')
emission_CO2_S2 = df['kg_CO2'].tolist()
emission_CH4_S2 = df['kg_CH4'].tolist()
emission_CO2_seq_S2 = df['kg_CO2_seq'].tolist()


#read S2_C
df = pd.read_excel('emissions_seq_RIL_Ret.xlsx', 'RIL_C_S2')
emission_CO2_S2_C = df['kg_CO2'].tolist()
emission_CH4_S2_C = df['kg_CH4'].tolist()
emission_CO2_seq_S2_C = df['kg_CO2_seq'].tolist()



emission_CO2_ref = df['emission_ref'].tolist() 

#read E
df = pd.read_excel('emissions_seq_RIL_Ret.xlsx', 'RIL_E') # can also index sheet by name or fetch all sheets
emission_CO2_E = df['kg_CO2'].tolist()
emission_CH4_E = df['kg_CH4'].tolist()
emission_CO2_seq_E = df['kg_CO2_seq'].tolist()


#read E_C
df = pd.read_excel('emissions_seq_RIL_Ret.xlsx', 'RIL_C_E') # can also index sheet by name or fetch all sheets
emission_CO2_E_C = df['kg_CO2'].tolist()
emission_CH4_E_C = df['kg_CH4'].tolist()
emission_CO2_seq_E_C = df['kg_CO2_seq'].tolist()


#%%

#Step (14): import emission data from the counter-use of non-renewable materials/energy scenarios (NR)


#read S2
df = pd.read_excel('RIL_Ret.xlsx', 'NonRW_RIL_S2')
emission_NonRW_RIL_S2 = df['NonRW_emissions'].tolist()
emission_NonRW_RIL_S2_seq = df['kg_CO2_seq'].tolist()


#read E
df = pd.read_excel('RIL_Ret.xlsx', 'NonRW_RIL_E') # can also index sheet by name or fetch all sheets
emission_NonRW_RIL_E = df['NonRW_emissions'].tolist()
emission_NonRW_RIL_E_seq = df['kg_CO2_seq'].tolist()


#%%

#Step (15): Determine the time elapsed dynamic characterization factors, DCF(t-ti), for CO2 and CH4


#DCF(t-i) CO2
matrix = (tf-1,tf-1)
DCF_CO2_ti = np.zeros(matrix)

for t in range(0,tf-1):
    i = -1
    while i < t:
        DCF_CO2_ti[i+1,t] = DCF_inst_CO2[t-i]
        i = i + 1

print(DCF_CO2_ti)

#sns.heatmap(DCF_CO2_ti)

DCF_CO2_ti.shape



#DCF(t-i) CH4
matrix = (tf-1,tf-1)
DCF_CH4_ti = np.zeros(matrix)

for t in range(0,tf-1):
    i = -1
    while i < t:
        DCF_CH4_ti[i+1,t] = DCF_inst_CH4[t-i]
        i = i + 1

print(DCF_CH4_ti)
#sns.heatmap(DCF_CH4_ti)

DCF_CH4_ti.shape

#%%

# Step (16): Calculate instantaneous global warming impact (GWI) 


#S2
t = np.arange(0,tf-1,1)

matrix_GWI_S2 = (tf-1,3)
GWI_inst_S2 = np.zeros(matrix_GWI_S2)



for t in range(0,tf-1):
    GWI_inst_S2[t,0] = np.sum(np.multiply(emission_CO2_S2,DCF_CO2_ti[:,t]))
    GWI_inst_S2[t,1] = np.sum(np.multiply(emission_CH4_S2,DCF_CH4_ti[:,t]))
    GWI_inst_S2[t,2] = np.sum(np.multiply(emission_CO2_seq_S2,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2 = (tf-1,1)
GWI_inst_tot_S2 = np.zeros(matrix_GWI_tot_S2)

GWI_inst_tot_S2[:,0] = np.array(GWI_inst_S2[:,0] + GWI_inst_S2[:,1] + GWI_inst_S2[:,2])
  
print(GWI_inst_tot_S2[:,0])


#S2_C
t = np.arange(0,tf-1,1)

matrix_GWI_S2_C = (tf-1,3)
GWI_inst_S2_C = np.zeros(matrix_GWI_S2_C)



for t in range(0,tf-1):
    GWI_inst_S2_C[t,0] = np.sum(np.multiply(emission_CO2_S2_C,DCF_CO2_ti[:,t]))
    GWI_inst_S2_C[t,1] = np.sum(np.multiply(emission_CH4_S2_C,DCF_CH4_ti[:,t]))
    GWI_inst_S2_C[t,2] = np.sum(np.multiply(emission_CO2_seq_S2_C,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2_C = (tf-1,1)
GWI_inst_tot_S2_C = np.zeros(matrix_GWI_tot_S2_C)

GWI_inst_tot_S2_C[:,0] = np.array(GWI_inst_S2_C[:,0] + GWI_inst_S2_C[:,1] + GWI_inst_S2_C[:,2])
  
print(GWI_inst_tot_S2_C[:,0])



#E 
t = np.arange(0,tf-1,1)

matrix_GWI_E = (tf-1,3)
GWI_inst_E = np.zeros(matrix_GWI_E)



for t in range(0,tf-1):
    GWI_inst_E[t,0] = np.sum(np.multiply(emission_CO2_E,DCF_CO2_ti[:,t]))
    GWI_inst_E[t,1] = np.sum(np.multiply(emission_CH4_E,DCF_CH4_ti[:,t]))
    GWI_inst_E[t,2] = np.sum(np.multiply(emission_CO2_seq_E,DCF_CO2_ti[:,t]))

matrix_GWI_tot_E = (tf-1,1)
GWI_inst_tot_E = np.zeros(matrix_GWI_tot_E)

GWI_inst_tot_E[:,0] = np.array(GWI_inst_E[:,0] + GWI_inst_E[:,1] + GWI_inst_E[:,2])
  
print(GWI_inst_tot_E[:,0])


#E_C 
t = np.arange(0,tf-1,1)

matrix_GWI_E_C = (tf-1,3)
GWI_inst_E_C = np.zeros(matrix_GWI_E_C)



for t in range(0,tf-1):
    GWI_inst_E_C[t,0] = np.sum(np.multiply(emission_CO2_E_C,DCF_CO2_ti[:,t]))
    GWI_inst_E_C[t,1] = np.sum(np.multiply(emission_CH4_E_C,DCF_CH4_ti[:,t]))
    GWI_inst_E_C[t,2] = np.sum(np.multiply(emission_CO2_seq_E_C,DCF_CO2_ti[:,t]))

matrix_GWI_tot_E_C = (tf-1,1)
GWI_inst_tot_E_C = np.zeros(matrix_GWI_tot_E_C)

GWI_inst_tot_E_C[:,0] = np.array(GWI_inst_E_C[:,0] + GWI_inst_E_C[:,1] + GWI_inst_E_C[:,2])
  
print(GWI_inst_tot_E_C[:,0])



##nonRW 

#S2
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_RIL_S2 = (tf-1,2)
GWI_inst_NonRW_RIL_S2 = np.zeros(matrix_GWI_NonRW_RIL_S2)



for t in range(0,tf-1):
    GWI_inst_NonRW_RIL_S2[t,0] = np.sum(np.multiply(emission_NonRW_RIL_S2,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_RIL_S2[t,1] = np.sum(np.multiply(emission_NonRW_RIL_S2_seq,DCF_CO2_ti[:,t]))
    

matrix_GWI_tot_NonRW_RIL_S2 = (tf-1,1)
GWI_inst_tot_NonRW_RIL_S2 = np.zeros(matrix_GWI_tot_NonRW_RIL_S2)

GWI_inst_tot_NonRW_RIL_S2[:,0] = np.array(GWI_inst_NonRW_RIL_S2[:,0] + GWI_inst_NonRW_RIL_S2[:,1])
  
print(GWI_inst_tot_NonRW_RIL_S2[:,0])


#E 
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_RIL_E = (tf-1,2)  
GWI_inst_NonRW_RIL_E = np.zeros(matrix_GWI_NonRW_RIL_E)



for t in range(0,tf-1):
    GWI_inst_NonRW_RIL_E[t,0] = np.sum(np.multiply(emission_NonRW_RIL_E,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_RIL_E[t,1] = np.sum(np.multiply(emission_NonRW_RIL_E_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_NonRW_RIL_E = (tf-1,1)
GWI_inst_tot_NonRW_RIL_E = np.zeros(matrix_GWI_tot_NonRW_RIL_E)

GWI_inst_tot_NonRW_RIL_E[:,0] = np.array(GWI_inst_NonRW_RIL_E[:,0] + GWI_inst_NonRW_RIL_E[:,1])
  
print(GWI_inst_tot_NonRW_RIL_E[:,0])



#plotting

t = np.arange(0,tf-1,1)

#create zero list to highlight the horizontal line for 0
def zerolistmaker(n):
    listofzeros = [0] * (n)
    return listofzeros


#convert to flat list

#NonRW
GWI_inst_tot_NonRW_RIL_S2 = np.array([item for sublist in GWI_inst_tot_NonRW_RIL_S2 for item in sublist])
GWI_inst_tot_NonRW_RIL_E = np.array([item for sublist in GWI_inst_tot_NonRW_RIL_E for item in sublist])


##Wood-based scenario
GWI_inst_tot_S2 = np.array([item for sublist in GWI_inst_tot_S2 for item in sublist])
GWI_inst_tot_E = np.array([item for sublist in GWI_inst_tot_E for item in sublist])

##Wood-based scenario_C
GWI_inst_tot_S2_C = np.array([item for sublist in GWI_inst_tot_S2_C for item in sublist])
GWI_inst_tot_E_C = np.array([item for sublist in GWI_inst_tot_E_C for item in sublist])



plt.plot(t, GWI_inst_tot_NonRW_RIL_S2, color='forestgreen', label='NR_RIL_M', ls='--', alpha=0.55)
plt.plot(t, GWI_inst_tot_NonRW_RIL_E, color='lightcoral', label='NR_RIL_E', ls='--', alpha=0.55)



plt.plot(t, GWI_inst_tot_S2, color='forestgreen', label='RIL_M')
plt.plot(t, GWI_inst_tot_E, color='lightcoral', label='RIL_E')
plt.plot(t, GWI_inst_tot_S2_C, color='turquoise', label='RIL_C_M')
plt.plot(t, GWI_inst_tot_E_C, color='cornflowerblue', label='RIL_C_E')

plt.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWI_inst_tot_NonRW_RIL_S2, GWI_inst_tot_NonRW_RIL_S1, color='lightcoral', alpha=0.3)
#plt.fill_between(t, GWI_inst_tot_NonRW_RIL_S2, GWI_inst_tot_NonRW_RIL_E, color='lightcoral', alpha=0.3)

plt.grid(True)

#plt.legend(loc='best', frameon=False)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={'size': 12}, frameon=False)

plt.xlim(0,200)
#plt.ylim(-5e-10,1e-10)

plt.title('Instantaneous GWI, RIL',fontsize=12)

plt.xlabel('Time (year)',fontsize=12)
#plt.ylabel('GWI_inst (10$^{-13}$ W/m$^2$)')
plt.ylabel('GWI_inst (W/(m$^2$.ha.year))',fontsize=12)

plt.tick_params(axis='both', labelsize=12)

plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_inst_NonRW_RIL', dpi=300)

plt.show()

#%%

#Step (17): Calculate cumulative global warming impact (GWI)


GWI_cum_S2 = np.cumsum(GWI_inst_tot_S2)
GWI_cum_E = np.cumsum(GWI_inst_tot_E)
GWI_cum_S2_C = np.cumsum(GWI_inst_tot_S2_C)
GWI_cum_E_C = np.cumsum(GWI_inst_tot_E_C)


GWI_cum_NonRW_RIL_S2 = np.cumsum(GWI_inst_tot_NonRW_RIL_S2)
GWI_cum_NonRW_RIL_E = np.cumsum(GWI_inst_tot_NonRW_RIL_E)



t = np.arange(0,tf-1,1)

plt.xlabel('Time (year)',fontsize=12)
#plt.ylabel('GWI_cum (10$^{-11}$ W/m$^2$)')
plt.ylabel('GWI_cum (W/(m$^2$.ha))',fontsize=12)

plt.xlim(0,200)
#plt.ylim(-6e-8,0.5e-8)


plt.title('Cumulative GWI, RIL',fontsize=12)
#plt.grid(True)

plt.plot(t, GWI_cum_NonRW_RIL_S2, color='forestgreen', label='NR_RIL_M', ls='--', alpha=0.55)
plt.plot(t, GWI_cum_NonRW_RIL_E, color='lightcoral', label='NR_RIL_E', ls='--', alpha=0.55)


plt.plot(t, GWI_cum_S2, color='forestgreen', label='RIL_M')
plt.plot(t, GWI_cum_E, color='lightcoral', label='RIL_E')
plt.plot(t, GWI_cum_S2_C, color='turquoise', label='RIL_C_M')
plt.plot(t, GWI_cum_E_C, color='cornflowerblue', label='RIL_C_E')

plt.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWI_cum_NonRW_RIL_S2, GWI_cum_NonRW_RIL_S1, color='lightcoral', alpha=0.3) 
#plt.fill_between(t, GWI_cum_NonRW_RIL_S2, GWI_cum_NonRW_RIL_E, color='lightcoral', alpha=0.3) 

plt.grid(True)

#plt.legend(loc='best', frameon=False)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={'size': 12}, frameon=False)

plt.tick_params(axis='both', labelsize=12)

plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_cum_Non_RW_RIL', dpi=300)

plt.show()

#%%

#Step (18): Determine the Instantenous and Cumulative GWI for the  emission reference (1 kg CO2 emission at time zero) before performing dynamic GWP calculation

t = np.arange(0,tf-1,1)

matrix_GWI_ref = (tf-1,1)
GWI_inst_ref = np.zeros(matrix_GWI_S2)

for t in range(0,tf-1):
    GWI_inst_ref[t,0] = np.sum(np.multiply(emission_CO2_ref,DCF_CO2_ti[:,t]))

#print(GWI_inst_ref[:,0])

len(GWI_inst_ref)



#determine the GWI cumulative for the emission reference

t = np.arange(0,tf-1,1)

GWI_cum_ref = np.cumsum(GWI_inst_ref[:,0])
#print(GWI_cum_ref)

plt.xlabel('Time (year)')
plt.ylabel('GWI_cum_ref (10$^{-13}$ W/m$^2$.kgCO$_2$)')

plt.plot(t, GWI_cum_ref)



len(GWI_cum_ref)

#%%

#Step (19): Calculate dynamic global warming potential (GWPdyn)

##Wood based scenario
GWP_dyn_cum_S2 = [x/(y*1000) for x,y in zip(GWI_cum_S2, GWI_cum_ref)]
GWP_dyn_cum_E = [x/(y*1000) for x,y in zip(GWI_cum_E, GWI_cum_ref)]
GWP_dyn_cum_S2_C = [x/(y*1000) for x,y in zip(GWI_cum_S2_C, GWI_cum_ref)]
GWP_dyn_cum_E_C = [x/(y*1000) for x,y in zip(GWI_cum_E_C, GWI_cum_ref)]


##NonRW
GWP_dyn_cum_NonRW_RIL_S2 = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_RIL_S2, GWI_cum_ref)]
GWP_dyn_cum_NonRW_RIL_E = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_RIL_E, GWI_cum_ref)]




t = np.arange(0,tf-1,1)


fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)

ax.plot(t, GWP_dyn_cum_S2, color='forestgreen', label='RIL_M')
ax.plot(t, GWP_dyn_cum_E, color='lightcoral', label='RIL_E')
ax.plot(t, GWP_dyn_cum_S2_C, color='turquoise', label='RIL_C_M')
ax.plot(t, GWP_dyn_cum_E_C, color='cornflowerblue', label='RIL_C_E')


ax.plot(t, GWP_dyn_cum_NonRW_RIL_S2, color='forestgreen', label='NR_RIL_M', ls='--', alpha=0.55)
ax.plot(t, GWP_dyn_cum_NonRW_RIL_E, color='lightcoral', label='NR_RIL_E', ls='--', alpha=0.55)

ax.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWP_dyn_cum_NonRW_RIL_S2, GWP_dyn_cum_NonRW_RIL_S1, color='lightcoral', alpha=0.3) 
#plt.fill_between(t, GWP_dyn_cum_NonRW_RIL_S2, GWP_dyn_cum_NonRW_RIL_E, color='lightcoral', alpha=0.3) 

plt.grid(True)

#ax.legend(loc='best', frameon=False)
ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={'size': 12}, frameon=False)

ax.tick_params(axis='both', labelsize=12)

ax.set_xlim(0,200)
#ax.set_ylim(-400,50)


ax.set_xlabel('Time (year)', fontsize=12)
ax.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq/ha)', fontsize=12)

ax.set_title('Dynamic GWP, RIL', fontsize=12)


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_cum_NonRW_RIL', dpi=300)


plt.draw()


#%%

#Step (20): Exporting the data behind result graphs to Excel

year = []
for x in range (0, 201): 
    year.append(x) 

### Create Column
    
Col1 = year


##GWI_Inst
#GWI_inst from wood-based scenarios
Col_GI_2 = GWI_inst_tot_S2
Col_GI_3  = GWI_inst_tot_E
Col_GI_2_C = GWI_inst_tot_S2_C
Col_GI_3_C  = GWI_inst_tot_E_C



#GWI_inst from counter use scenarios
Col_GI_5  = GWI_inst_tot_NonRW_RIL_S2
Col_GI_6  = GWI_inst_tot_NonRW_RIL_E


#create column results
    
##GWI_cumulative
#GWI_cumulative from wood-based scenarios
Col_GC_2 = GWI_cum_S2
Col_GC_3 = GWI_cum_E
Col_GC_2_C = GWI_cum_S2_C
Col_GC_3_C = GWI_cum_E_C


#GWI_cumulative from counter use scenarios
Col_GC_5 = GWI_cum_NonRW_RIL_S2
Col_GC_6 = GWI_cum_NonRW_RIL_E



#create column results

##GWPdyn
#GWPdyn from wood-based scenarios
Col_GWP_2 = GWP_dyn_cum_S2
Col_GWP_3 = GWP_dyn_cum_E
Col_GWP_2_C = GWP_dyn_cum_S2_C
Col_GWP_3_C = GWP_dyn_cum_E_C


#GWPdyn from counter use scenarios
Col_GWP_5 = GWP_dyn_cum_NonRW_RIL_S2
Col_GWP_6 = GWP_dyn_cum_NonRW_RIL_E


#Create colum results
dfM_GI = pd.DataFrame.from_dict({'Year':Col1,'RIL_M (W/m2)':Col_GI_2, 'RIL_C_M (W/m2)':Col_GI_2_C,
                                 'RIL_E (W/m2)':Col_GI_3, 'RIL_C_E (W/m2)':Col_GI_3_C,
                                 'NR_RIL_M (W/m2)':Col_GI_5, 'NR_RIL_E (W/m2)':Col_GI_6})


dfM_GC = pd.DataFrame.from_dict({'Year':Col1,'RIL_M (W/m2)':Col_GC_2, 'RIL_C_M (W/m2)':Col_GC_2_C,
                                 'RIL_E (W/m2)':Col_GC_3, 'RIL_C_E (W/m2)':Col_GC_3_C,
                                 'NR_RIL_M (W/m2)':Col_GC_5, 'NR_RIL_E (W/m2)':Col_GC_6})

    
dfM_GWPdyn = pd.DataFrame.from_dict({'Year':Col1,'RIL_M (t-CO2-eq)':Col_GWP_2, 'RIL_C_M (t-CO2-eq)':Col_GWP_2_C, 
                                     'RIL_E (t-CO2-eq)':Col_GWP_3, 'RIL_C_E (t-CO2-eq)':Col_GWP_3_C,
                                  'NR_RIL_M (t-CO2-eq)':Col_GWP_5, 'NR_RIL_E (t-CO2-eq)':Col_GWP_6})

    
#Export to excel
writer = pd.ExcelWriter('GraphResults_RIL_Ret.xlsx', engine = 'xlsxwriter')


dfM_GI.to_excel(writer, sheet_name = 'Inst_GWI_RIL', header=True, index=False)

dfM_GC.to_excel(writer, sheet_name = 'Cumulative GWI_RIL', header=True, index=False)

dfM_GWPdyn.to_excel(writer, sheet_name = 'GWPdyn_RIL', header=True, index=False)



writer.save()
writer.close()

#%%

#Step (21): Generate the excel file for the individual carbon emission and sequestration flows

#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)



division = 1000*44/12
division_CH4 = 1000*16/12



flat_list_RIL = [x/division for x in flat_list_RIL]
flat_list_RIL_C = [x/division for x in flat_list_RIL_C]


#RIL_M
c_firewood_energy_S2 = [x/division for x in c_firewood_energy_S2]
decomp_tot_S2[:,0] = [x/division for x in decomp_tot_S2[:,0]]
RIL_seq_S2 = [x/division for x in RIL_seq_S2]
TestDSM2.o = [x/division for x in TestDSM2.o]
PH_Emissions_HWP2 = [x/division for x in PH_Emissions_HWP2]
#OC_storage_RIL_S2 = [x/division for x in OC_storage_RIL_S2]
decomp_tot_CO2_S2[:,0] = [x/division for x in decomp_tot_CO2_S2[:,0]]

decomp_tot_CH4_S2[:,0] = [x/division_CH4 for x in decomp_tot_CH4_S2[:,0]]


#RIL_C_M
decomp_tot_S2_C[:,0] = [x/division for x in decomp_tot_S2_C[:,0]]
RIL_seq_C_S2 = [x/division for x in RIL_seq_C_S2]





#RIL_E
c_firewood_energy_E = [x/division for x in c_firewood_energy_E]
RIL_seq_E = [x/division for x in RIL_seq_E]
c_pellets_E = [x/division for x in c_pellets_E]
decomp_tot_E[:,0] = [x/division for x in decomp_tot_E[:,0]]
TestDSME.o = [x/division for x in TestDSME.o]
PH_Emissions_HWPE = [x/division for x in PH_Emissions_HWPE]
#OC_storage_RIL_E = [x/division for x in OC_storage_RIL_E]
decomp_tot_CO2_E[:,0] = [x/division for x in decomp_tot_CO2_E]

decomp_tot_CH4_E[:,0] = [x/division_CH4 for x in decomp_tot_CH4_E]


#RIL_E_C
decomp_tot_E_C[:,0] = [x/division for x in decomp_tot_E_C[:,0]]
RIL_seq_C_E = [x/division for x in RIL_seq_C_E]



#landfill aggregate flows
Landfill_decomp_S2 = decomp_tot_CH4_S2, decomp_tot_CO2_S2
Landfill_decomp_E = decomp_tot_CH4_E, decomp_tot_CO2_E

Landfill_decomp_S2 = [sum(x) for x in zip(*Landfill_decomp_S2)]
Landfill_decomp_E = [sum(x) for x in zip(*Landfill_decomp_E)]

Landfill_decomp_S2 = [item for sublist in Landfill_decomp_S2 for item in sublist]
Landfill_decomp_E = [item for sublist in Landfill_decomp_E for item in sublist]



#RIL_M
Column1 = year
Column2 = c_firewood_energy_S2
Column3 = decomp_tot_S2[:,0]
Column3_C = decomp_tot_S2_C[:,0]
Column4 = TestDSM2.o
Column5 = PH_Emissions_HWP2
#Column6_1 = OC_storage_RIL_S2
Column6 = Landfill_decomp_S2
Column7 = flat_list_RIL 




#RIL_E
Column8 = c_firewood_energy_E
Column8_1 = c_pellets_E
Column9 = decomp_tot_E[:,0]
Column9_C = decomp_tot_E_C[:,0]
Column10 = TestDSME.o
Column11 = PH_Emissions_HWPE
#Column12_1 = OC_storage_RIL_E
Column12 = Landfill_decomp_E






#M
dfM = pd.DataFrame.from_dict({'Year':Column1,'F0-1: Biomass C sequestration (t-C)':[x + y for x, y in zip(flat_list_RIL, RIL_seq_S2)],
                              #'9: Landfill storage (t-C)': Column6_1,
                              'F1-0: Residue decomposition (t-C)':Column3,
                              'F6-0-1: Emissions from firewood/other energy use (t-C)':Column2,
                              'F8-0: Operational stage/processing emissions (t-C)':Column5,
                              'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column4,
                              'F7-0: Landfill gas decomposition (t-C)':Column6})


#M_C
dfM_C = pd.DataFrame.from_dict({'Year':Column1,'F0-1: Biomass C sequestration (t-C)':[x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_S2)],
                              #'9: Landfill storage (t-C)': Column6_1,
                              'F1-0: Residue decomposition (t-C)':Column3_C,
                              'F6-0-1: Emissions from firewood/other energy use (t-C)':Column2,
                              'F8-0: Operational stage/processing emissions (t-C)':Column5,
                              'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column4,
                              'F7-0: Landfill gas decomposition (t-C)':Column6})


dfE = pd.DataFrame.from_dict({'Year':Column1, 'F0-1: Biomass C sequestration (t-C)':[x + y for x, y in zip(flat_list_RIL, RIL_seq_E)],
                              #'9: Landfill storage (t-C)': Column12_1,
                              'F1-0: Residue decomposition (t-C)':Column9,
                              'F6-0-1: Emissions from firewood/other energy use (t-C)':Column8,
                              'F8-0: Operational stage/processing emissions (t-C)':Column11,
                              'F6-0-2: Energy use emissions from in-use stocks outflow  (t-C)':Column10,
                              'F7-0: Landfill gas decomposition (t-C)':Column12,
                              'F4-0: Emissions from from wood pellets use (t-C)': Column8_1})


dfE_C = pd.DataFrame.from_dict({'Year':Column1, 'F0-1: Biomass C sequestration (t-C)':[x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_E)],
                              #'9: Landfill storage (t-C)': Column12_1,
                              'F1-0: Residue decomposition (t-C)':Column9_C,
                              'F6-0-1: Emissions from firewood/other energy use (t-C)':Column8,
                              'F8-0: Operational stage/processing emissions (t-C)':Column11,
                              'F6-0-2: Energy use emissions from in-use stocks outflow  (t-C)':Column10,
                              'F7-0: Landfill gas decomposition (t-C)':Column12,
                              'F4-0: Emissions from from wood pellets use (t-C)': Column8_1})
    

    
writer = pd.ExcelWriter('C_flows_RIL_Ret.xlsx', engine = 'xlsxwriter')


dfM.to_excel(writer, sheet_name = 'RIL_M', header=True, index=False)
dfE.to_excel(writer, sheet_name = 'RIL_E', header=True, index=False)
dfM_C.to_excel(writer, sheet_name = 'RIL_C_M', header=True, index=False)
dfE_C.to_excel(writer, sheet_name = 'RIL_C_E', header=True, index=False)


writer.save()
writer.close()


#%%

#Step (22): Plot of the individual carbon emission and sequestration flows for normal and symlog-scale graphs

#RIL_M

fig=plt.figure()
fig.show()
ax1_s=fig.add_subplot(111)





ax1_s.plot(t, [x + y for x, y in zip(flat_list_RIL, RIL_seq_S2)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax1_s.plot(t, OC_storage_RIL_S2, color='darkturquoise', label='9: Landfill storage') 
ax1_s.plot(t, decomp_tot_S2[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax1_s.plot(t, c_firewood_energy_S2, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax1_s.plot(t, PH_Emissions_HWP2, color='orange', label='F8-0: Operational stage/processing emissions')
ax1_s.plot(t, TestDSM2.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow')
ax1_s.plot(t, Landfill_decomp_S2, color='yellow', label= 'F7-0: Landfill gas decomposition')


ax1_s.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax1_s.set_xlim(-1,200)

ax1_s.set_yscale('symlog')


ax1_s.set_xlabel('Time (year)')
ax1_s.set_ylabel('C flows(t-C) (symlog)')

ax1_s.set_title('Carbon flow, RIL_M (symlog-scale)')


plt.draw()

#%%

#plot for the individual carbon flows

#RIL_M

fig=plt.figure()
fig.show()
ax1=fig.add_subplot(111)


ax1.plot(t, [x + y for x, y in zip(flat_list_RIL, RIL_seq_S2)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax1.plot(t, OC_storage_RIL_S2, color='darkturquoise', label='9: Landfill storage') 
ax1.plot(t, decomp_tot_S2[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax1.plot(t, c_firewood_energy_S2, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax1.plot(t, PH_Emissions_HWP2, color='orange', label='F8-0: Operational stage/processing emissions')
ax1.plot(t, TestDSM2.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow')
ax1.plot(t, Landfill_decomp_S2, color='yellow', label= 'F7-0: Landfill gas decomposition')

 
ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax1.set_xlim(-1,200)
ax1.set_ylim(-3,10)

ax1.set_xlabel('Time (year)')
ax1.set_ylabel('C flows(t-C)')

ax1.set_title('Carbon flow, RIL_M')


#plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_1_RIL_M')


plt.draw()


#%%

#RIL_M_C




fig=plt.figure()
fig.show()
ax1_C_s=fig.add_subplot(111)


ax1_C_s.plot(t, [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_S2)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax1_s.plot(t, OC_storage_RIL_S2, color='darkturquoise', label='9: Landfill storage') 
ax1_C_s.plot(t, decomp_tot_S2_C[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax1_C_s.plot(t, c_firewood_energy_S2, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax1_C_s.plot(t, PH_Emissions_HWP2, color='orange', label='F8-0: Operational stage/processing emissions')
ax1_C_s.plot(t, TestDSM2.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow')
ax1_C_s.plot(t, Landfill_decomp_S2, color='yellow', label= 'F7-0: Landfill gas decomposition')


ax1_C_s.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax1_C_s.set_xlim(-1,200)

ax1_C_s.set_yscale('symlog')


ax1_C_s.set_xlabel('Time (year)')
ax1_C_s.set_ylabel('C flows(t-C) (symlog)')

ax1_C_s.set_title('Carbon flow, RIL_C_M (symlog-scale)')


plt.draw()

#%%

#plot for the individual carbon flows

#RIL_M_C

fig=plt.figure()
fig.show()
ax1_C=fig.add_subplot(111)


ax1_C.plot(t, [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_S2)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax1.plot(t, OC_storage_RIL_S2, color='darkturquoise', label='9: Landfill storage') 
ax1_C.plot(t, decomp_tot_S2_C[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax1_C.plot(t, c_firewood_energy_S2, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
ax1_C.plot(t, PH_Emissions_HWP2, color='orange', label='F8-0: Operational stage/processing emissions')
ax1_C.plot(t, TestDSM2.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow')
ax1_C.plot(t, Landfill_decomp_S2, color='yellow', label= 'F7-0: Landfill gas decomposition')

 
ax1_C.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax1_C.set_xlim(-1,200)
ax1_C.set_ylim(-3,10)

ax1_C.set_xlabel('Time (year)')
ax1_C.set_ylabel('C flows(t-C)')

ax1_C.set_title('Carbon flow, RIL_C_M')


#plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_1_RIL_M')


plt.draw()




#%%

#plot for the individual carbon flows - test for symlog-scale graphs

#RIL_E

fig=plt.figure()
fig.show()
ax2_s=fig.add_subplot(111)



ax2_s.plot(t, [x + y for x, y in zip(flat_list_RIL, RIL_seq_E)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax2_s.plot(t, OC_storage_RIL_E, color='darkturquoise', label='9: Landfill storage') 
ax2_s.plot(t, decomp_tot_E[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2_s.plot(t, c_firewood_energy_E, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
#ax2_s.plot(t, TestDSME.o, color='royalblue', label='14: Energy use from obsolete stocks')
ax2_s.plot(t, PH_Emissions_HWPE, color='orange', label='F8-0: Operational stage/processing emissions')
ax2_s.plot(t, Landfill_decomp_E, color='yellow', label= 'F7-0: Landfill gas decomposition')
ax2_s.plot(t, c_pellets_E, color='slategrey', label='F4-0: Emissions from wood pellets use') 


ax2_s.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2_s.set_xlim(-1,200)

ax2_s.set_yscale('symlog')


ax2_s.set_xlabel('Time (year)')
ax2_s.set_ylabel('C flows(t-C) (symlog)')

ax2_s.set_title('Carbon flow, RIL_E (symlog-scale)')

plt.draw()

#%%

#plot for the individual carbon flows

#RIL_E

fig=plt.figure()
fig.show()
ax2=fig.add_subplot(111)

ax2.plot(t, [x + y for x, y in zip(flat_list_RIL, RIL_seq_E)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax2.plot(t, OC_storage_RIL_E, color='darkturquoise', label='9: Landfill storage') 
ax2.plot(t, decomp_tot_E[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2.plot(t, c_firewood_energy_E, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
#ax2.plot(t, TestDSME.o, color='royalblue', label='14: Energy use from obsolete stocks')
ax2.plot(t, PH_Emissions_HWPE, color='orange', label='F8-0: Operational stage/processing emissions')
ax2.plot(t, Landfill_decomp_E, color='yellow', label= 'F7-0: Landfill gas decomposition')
ax2.plot(t, c_pellets_E, color='slategrey', label='F4-0: Emissions from wood pellets use') 
 
ax2.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2.set_xlim(-1,200)
ax2.set_ylim(-3,10)


ax2.set_xlabel('Time (year)')
ax2.set_ylabel('C flows(t-C)')

ax2.set_title('Carbon flow, RIL_E')


#plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_1_RIL_E')


plt.draw()


#%%


#RIL_E_C




fig=plt.figure()
fig.show()
ax2_C_s=fig.add_subplot(111)


ax2_C_s.plot(t, [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_E)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax2_s.plot(t, OC_storage_RIL_E, color='darkturquoise', label='9: Landfill storage') 
ax2_C_s.plot(t, decomp_tot_E_C[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2_C_s.plot(t, c_firewood_energy_E, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
#ax2_s.plot(t, TestDSME.o, color='royalblue', label='14: Energy use from obsolete stocks')
ax2_C_s.plot(t, PH_Emissions_HWPE, color='orange', label='F8-0: Operational stage/processing emissions')
ax2_C_s.plot(t, Landfill_decomp_E, color='yellow', label= 'F7-0: Landfill gas decomposition')
ax2_C_s.plot(t, c_pellets_E, color='slategrey', label='F4-0: Emissions from wood pellets use') 


ax2_C_s.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2_C_s.set_xlim(-1,200)

ax2_C_s.set_yscale('symlog')


ax2_C_s.set_xlabel('Time (year)')
ax2_C_s.set_ylabel('C flows(t-C) (symlog)')

ax2_C_s.set_title('Carbon flow, RIL_C_E (symlog-scale)')

plt.draw()

#%%

#plot for the individual carbon flows

#RIL_E_C

fig=plt.figure()
fig.show()
ax2_C=fig.add_subplot(111)

ax2_C.plot(t, [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_E)], color='darkkhaki', label='F0-1: Biomass C sequestration')
#ax2.plot(t, OC_storage_RIL_E, color='darkturquoise', label='9: Landfill storage') 
ax2_C.plot(t, decomp_tot_E_C[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2_C.plot(t, c_firewood_energy_E, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')
#ax2.plot(t, TestDSME.o, color='royalblue', label='14: Energy use from obsolete stocks')
ax2_C.plot(t, PH_Emissions_HWPE, color='orange', label='F8-0: Operational stage/processing emissions')
ax2_C.plot(t, Landfill_decomp_E, color='yellow', label= 'F7-0: Landfill gas decomposition')
ax2_C.plot(t, c_pellets_E, color='slategrey', label='F4-0: Emissions from wood pellets use') 
 
ax2_C.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2_C.set_xlim(-1,200)
ax2_C.set_ylim(-3,10)


ax2_C.set_xlabel('Time (year)')
ax2_C.set_ylabel('C flows(t-C)')

ax2_C.set_title('Carbon flow, RIL_C_E')


#plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_1_RIL_E')


plt.draw()

#%%

#Step (23): Generate the excel file for the net carbon balance




Agg_Cflow_S2 = [c_firewood_energy_S2, RIL_seq_S2, decomp_tot_S2[:,0], TestDSM2.o, PH_Emissions_HWP2, Landfill_decomp_S2, flat_list_RIL]
Agg_Cflow_E = [c_firewood_energy_E, RIL_seq_E, c_pellets_E, decomp_tot_E[:,0], TestDSME.o, PH_Emissions_HWPE, decomp_tot_CO2_E[:,0], Landfill_decomp_E, flat_list_RIL]
Agg_Cflow_S2_C = [c_firewood_energy_S2, RIL_seq_C_S2, decomp_tot_S2_C[:,0], TestDSM2.o, PH_Emissions_HWP2, Landfill_decomp_S2, flat_list_RIL_C]
Agg_Cflow_E_C = [c_firewood_energy_E, RIL_seq_C_E, c_pellets_E, decomp_tot_E_C[:,0], TestDSME.o, PH_Emissions_HWPE, decomp_tot_CO2_E[:,0], Landfill_decomp_E, flat_list_RIL_C]



Agg_Cflow_RIL_S2 = [sum(x) for x in zip(*Agg_Cflow_S2)]
Agg_Cflow_RIL_E = [sum(x) for x in zip(*Agg_Cflow_E)]
Agg_Cflow_RIL_S2_C = [sum(x) for x in zip(*Agg_Cflow_S2_C)]
Agg_Cflow_RIL_E_C = [sum(x) for x in zip(*Agg_Cflow_E_C)]


#create column year
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)

#Create colum results
dfM_RIL = pd.DataFrame.from_dict({'Year':year,'RIL_M (t-C)':Agg_Cflow_RIL_S2, 'RIL_C_M (t-C)':Agg_Cflow_RIL_S2_C, 
                                  'RIL_E (t-C)':Agg_Cflow_RIL_E, 'RIL_C_E (t-C)':Agg_Cflow_RIL_E_C})

    
#Export to excel
writer = pd.ExcelWriter('AggCFlow_RIL_Ret.xlsx', engine = 'xlsxwriter')


dfM_RIL.to_excel(writer, sheet_name = 'RIL', header=True, index=False)

writer.save()
writer.close()

#%%

#Step (24): Plot the net carbon balance 


fig=plt.figure()
fig.show()
ax3=fig.add_subplot(111)

# plot
ax3.plot(t, Agg_Cflow_RIL_S2, color='forestgreen', label='RIL_M') 
ax3.plot(t, Agg_Cflow_RIL_E, color='lightcoral', label='RIL_E') 
ax3.plot(t, Agg_Cflow_RIL_S2_C, color='turquoise', label='RIL_C_M') 
ax3.plot(t, Agg_Cflow_RIL_E_C, color='cornflowerblue', label='RIL_C_E') 

ax3.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

ax3.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={'size': 12}, frameon=False)

ax3.tick_params(axis='both', labelsize=12)

ax3.set_xlim(-0.35,85)
#ax3.set_xlim(-1,200)


#ax3.set_yscale('symlog')
 
ax3.set_xlabel('Time (year)', fontsize=12)
ax3.set_ylabel('C flows (t-C/ha)', fontsize=12)

ax3.set_title('Net carbon balance, RIL', fontsize=12)

plt.savefig('NCB RIL_Ret.png', dpi=300)

plt.draw()

#%%

#Step (25): Generate the excel file for documentation of individual carbon flows in the system definition (Fig. 1)


#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)


df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_S2')
dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'RIL_E')



Column1 = year
division = 1000*44/12
division_CH4 = 1000*16/12


##RIL_S2
## define the input flow for the landfill (F5-7)
OC_storage_S2 = df2['Other_C_storage'].values


OC_storage_S2 = [x/division for x in OC_storage_S2]
OC_storage_S2 = [abs(number) for number in OC_storage_S2]

C_LF_S2 = [x*1/0.82 for x in OC_storage_S2]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_S2 = [x/division for x in df2['Input_PF'].values]
HWP_S2_energy =  [x*1/3 for x in c_firewood_energy_S2]
HWP_S2_landfill = [x*1/0.82 for x in OC_storage_S2]

HWP_S2_sum = [HWP_S2, HWP_S2_energy, HWP_S2_landfill]
HWP_S2_sum = [sum(x) for x in zip(*HWP_S2_sum )]

#in-use stocks (S-4)
TestDSM2.s = [x/division for x in TestDSM2.s]
#TestDSM2.i = [x/division for x in TestDSM2.i]



# calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_S2 = (tf,1)
stocks_S2 = np.zeros(zero_matrix_stocks_S2)


i = 0
stocks_S2[0] = C_LF_S2[0] - Landfill_decomp_S2[0]

while i < tf-1:
    stocks_S2[i+1] = np.array(C_LF_S2[i+1] - Landfill_decomp_S2[i+1] + stocks_S2[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_S2 = [x1+x2 for (x1,x2) in zip(HWP_S2_sum, [x*2/3 for x in c_firewood_energy_S2])] 




## RIL_M: calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2 = (tf,1)
ForCstocks_S2 = np.zeros(zero_matrix_ForCstocks_S2)

i = 0
ForCstocks_S2[0] = initAGB - flat_list_RIL[0] - decomp_tot_S2[0] - HWP_logged_S2[0]

while i < tf-1:
    ForCstocks_S2[i+1] = np.array(ForCstocks_S2[i] - flat_list_RIL[i+1] - decomp_tot_S2[i+1] - HWP_logged_S2[i+1])
    i = i + 1
    
    
    
    
## RIL_C_M: calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2_C = (tf,1)
ForCstocks_S2_C = np.zeros(zero_matrix_ForCstocks_S2_C)

i = 0
ForCstocks_S2_C[0] = initAGB - flat_list_RIL[0] - decomp_tot_S2_C[0] - HWP_logged_S2[0]

while i < tf-1:
    ForCstocks_S2_C[i+1] = np.array(ForCstocks_S2_C[i] - flat_list_RIL[i+1] - decomp_tot_S2_C[i+1] - HWP_logged_S2[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
df2_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'NonRW_RIL_S2')
NonRW_amount_S2 = df2_amount['NonRW_amount'].values

NonRW_amount_S2 = [x/1000 for x in NonRW_amount_S2]



##NonRW emissions (F9-0-2)
emission_NonRW_RIL_S2 = [x/division for x in emission_NonRW_RIL_S2]
    





#create columns
dfM = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': [x + y for x, y in zip(flat_list_RIL, RIL_seq_S2)],
                                    'F1-0 (t-C)': decomp_tot_S2[:,0],
                                    #'F1a-2 (t-C)': PF_S2_Ac_7y,
                                    #'F1c-2 (t-C)': FP_S2_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_S2, 
                                    'St-1 (t-C)':ForCstocks_S2[:,0], 
                                    'F2-3 (t-C)': HWP_S2_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_S2], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_S2_sum, [x*1/0.82 for x in OC_storage_S2], [x*1/3 for x in c_firewood_energy_S2])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_S2],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_S2], 
                                   # 'F4-0 (t-C)':,
                                    'St-4 (t-C)': TestDSM2.s, 
                                    #'S-4-i (t-C)': TestDSM2.i,
                                    'F4-5 (t-C)': TestDSM2.o,
                                    'F5-6 (t-C)': TestDSM2.o, 
                                    'F5-7 (t-C)': C_LF_S2,
                                    'F6-0-1 (t-C)': c_firewood_energy_S2,
                                    'F6-0-2 (t-C)': TestDSM2.o,
                                    'St-7 (t-C)': stocks_S2[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_S2,
                                    'F8-0 (t-C)': PH_Emissions_HWP2,
                                    'S9-0 (t)': NonRW_amount_S2, 
                                    'F9-0 (t-C)': emission_NonRW_RIL_S2,
                                    })


dfM_C = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_S2)],
                                    'F1-0 (t-C)': decomp_tot_S2_C[:,0],
                                    #'F1a-2 (t-C)': PF_S2_Ac_7y,
                                    #'F1c-2 (t-C)': FP_S2_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_S2, 
                                    'St-1 (t-C)':ForCstocks_S2_C[:,0], 
                                    'F2-3 (t-C)': HWP_S2_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_S2], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_S2_sum, [x*1/0.82 for x in OC_storage_S2], [x*1/3 for x in c_firewood_energy_S2])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_S2],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_S2], 
                                   # 'F4-0 (t-C)':,
                                    'St-4 (t-C)': TestDSM2.s, 
                                    #'S-4-i (t-C)': TestDSM2.i,
                                    'F4-5 (t-C)': TestDSM2.o,
                                    'F5-6 (t-C)': TestDSM2.o, 
                                    'F5-7 (t-C)': C_LF_S2,
                                    'F6-0-1 (t-C)': c_firewood_energy_S2,
                                    'F6-0-2 (t-C)': TestDSM2.o,
                                    'St-7 (t-C)': stocks_S2[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_S2,
                                    'F8-0 (t-C)': PH_Emissions_HWP2,
                                    'S9-0 (t)': NonRW_amount_S2, 
                                    'F9-0 (t-C)': emission_NonRW_RIL_S2,
                                    })

    
    



##RIL_E
## define the input flow for the landfill (F5-7)
OC_storage_E = dfE['Other_C_storage'].values


OC_storage_E = [x/division for x in OC_storage_E]
OC_storage_E = [abs(number) for number in OC_storage_E]

C_LF_E = [x*1/0.82 for x in OC_storage_E]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_E = [x/division for x in dfE['Wood_pellets'].values]
HWP_E_energy =  [x*1/3 for x in c_firewood_energy_E]
HWP_E_landfill = [x*1/0.82 for x in OC_storage_E]

HWP_E_sum = [HWP_E, HWP_E_energy, HWP_E_landfill]
HWP_E_sum = [sum(x) for x in zip(*HWP_E_sum )]

#in-use stocks (S-4)
TestDSME.s = [x/division for x in TestDSME.s]
#TestDSME.i = [x/division for x in TestDSME.i]



# calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_E = (tf,1)
stocks_E = np.zeros(zero_matrix_stocks_E)


i = 0
stocks_E[0] = C_LF_E[0] - Landfill_decomp_E[0]

while i < tf-1:
    stocks_E[i+1] = np.array(C_LF_E[i+1] - Landfill_decomp_E[i+1] + stocks_E[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_E = [x1+x2 for (x1,x2) in zip(HWP_E_sum, [x*2/3 for x in c_firewood_energy_E])] 




## RIL_E: calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_E = (tf,1)
ForCstocks_E = np.zeros(zero_matrix_ForCstocks_E)

i = 0
ForCstocks_E[0] = initAGB - flat_list_RIL[0] - decomp_tot_E[0] - HWP_logged_E[0]

while i < tf-1:
    ForCstocks_E[i+1] = np.array(ForCstocks_E[i] - flat_list_RIL[i+1] - decomp_tot_E[i+1] - HWP_logged_E[i+1])
    i = i + 1
    
    
    
## RIL_C_E: calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_E_C = (tf,1)
ForCstocks_E_C = np.zeros(zero_matrix_ForCstocks_E_C)

i = 0
ForCstocks_E_C[0] = initAGB - flat_list_RIL[0] - decomp_tot_E_C[0] - HWP_logged_E[0]

while i < tf-1:
    ForCstocks_E_C[i+1] = np.array(ForCstocks_E[i] - flat_list_RIL[i+1] - decomp_tot_E_C[i+1] - HWP_logged_E[i+1])
    i = i + 1



##NonRW materials/energy amount (F9-0-1)
dfE_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\RIL_Ret.xlsx', 'NonRW_RIL_E')
NonRW_amount_E = dfE_amount['NonRW_amount'].values

NonRW_amount_E = [x/1000 for x in NonRW_amount_E]



##NonRW emissions (F9-0-2)
emission_NonRW_RIL_E = [x/division for x in emission_NonRW_RIL_E]
    



#create columns
dfE = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': [x + y for x, y in zip(flat_list_RIL, RIL_seq_E)],
                                    'F1-0 (t-C)': decomp_tot_E[:,0],
                                    #'F1a-2 (t-C)': PF_E_Ac_7y,
                                    #'F1c-2 (t-C)': FP_E_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_E, 
                                    'St-1 (t-C)':ForCstocks_E[:,0], 
                                    'F2-3 (t-C)': HWP_E_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_E], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_E_sum, [x*1/0.82 for x in OC_storage_E], [x*1/3 for x in c_firewood_energy_E])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_E],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_E], 
                                    'F4-0 (t-C)': c_pellets_E,
                                    'St-4 (t-C)': TestDSME.s, 
                                    #'S-4-i (t-C)': TestDSME.i,
                                    'F4-5 (t-C)': TestDSME.o,
                                    'F5-6 (t-C)': TestDSME.o, 
                                    'F5-7 (t-C)': C_LF_E,
                                    'F6-0-1 (t-C)': c_firewood_energy_E,
                                    'F6-0-2 (t-C)': TestDSME.o,
                                    'St-7 (t-C)': stocks_E[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_E,
                                    'F8-0 (t-C)': PH_Emissions_HWPE,
                                    'S9-0 (t)': NonRW_amount_E, 
                                    'F9-0 (t-C)': emission_NonRW_RIL_E,
                                    })



dfE_C = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': [x + y for x, y in zip(flat_list_RIL_C, RIL_seq_C_E)],
                                    'F1-0 (t-C)': decomp_tot_E_C[:,0],
                                    #'F1a-2 (t-C)': PF_E_Ac_7y,
                                    #'F1c-2 (t-C)': FP_E_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_E, 
                                    'St-1 (t-C)':ForCstocks_E_C[:,0], 
                                    'F2-3 (t-C)': HWP_E_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_E], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_E_sum, [x*1/0.82 for x in OC_storage_E], [x*1/3 for x in c_firewood_energy_E])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_E],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_E], 
                                    'F4-0 (t-C)': c_pellets_E,
                                    'St-4 (t-C)': TestDSME.s, 
                                    #'S-4-i (t-C)': TestDSME.i,
                                    'F4-5 (t-C)': TestDSME.o,
                                    'F5-6 (t-C)': TestDSME.o, 
                                    'F5-7 (t-C)': C_LF_E,
                                    'F6-0-1 (t-C)': c_firewood_energy_E,
                                    'F6-0-2 (t-C)': TestDSME.o,
                                    'St-7 (t-C)': stocks_E[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_E,
                                    'F8-0 (t-C)': PH_Emissions_HWPE,
                                    'S9-0 (t)': NonRW_amount_E, 
                                    'F9-0 (t-C)': emission_NonRW_RIL_E,
                                    })




writer = pd.ExcelWriter('C_flows_SysDef_RIL_Ret.xlsx', engine = 'xlsxwriter')



dfM.to_excel(writer, sheet_name = 'RIL_M', header=True, index=False)
dfE.to_excel(writer, sheet_name = 'RIL_E2', header=True, index=False)
dfM_C.to_excel(writer, sheet_name = 'RIL_C_M', header=True, index=False)
dfE_C.to_excel(writer, sheet_name = 'RIL_C_E2', header=True, index=False)



writer.save()
writer.close()
#%%
