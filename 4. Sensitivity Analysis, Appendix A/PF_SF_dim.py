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


#PF_SF Scenario

##Set parameters
#Parameters for primary forest
initAGB = 233 #t-C           #source: van Beijma et al. (2018)
initAGB_min = 233-72 #t-C
initAGB_max = 233 + 72 #t-C

#parameters for secondary forest. Sourc: Busch et al. (2019)
coeff_MF_nonpl = 11.47
coeff_DF_nonpl = 11.24
coeff_GL_nonpl = 9.42
coeff_MF_pl =17.2

tf = 201  #years

a = 0.082
b = 2.53



#%%

#Step (2_1): C loss from the harvesting/clear cut



df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S2')
df3 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')


t = range(0,tf,1)



c_firewood_energy_S2 = df2['Firewood_other_energy_use'].values

c_firewood_energy_E = df3['Firewood_other_energy_use'].values



#%%

#Step (2_2): C loss from the harvesting/clear cut as wood pellets

dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')

c_pellets_E = df3['Wood_pellets'].values


#%%

#Step (3): Aboveground biomass (AGB) decomposition


#S2
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S2')

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


#E
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')

tf = 201

t = np.arange(tf)


def decomp_E_trial(t,remainAGB_E):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_E



#set zero matrix
output_decomp_E = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_E in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_E[i:,i] = decomp_E_trial(t[:len(t)-i],remain_part_E)

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
zero_matrix_E_trial = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_E_trial)

subs_matrix_E = np.vstack((zero_matrix_E_trial, subs_matrix_E))

print(subs_matrix_E[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_E = (tf,1)
decomp_tot_E = np.zeros(matrix_tot_E) 

i = 0
while i < tf:
    decomp_tot_E[:,0] = decomp_tot_E[:,0] + subs_matrix_E[:,i]
    i = i + 1

print(decomp_tot_E[:,0])




#plotting
t = np.arange(0,tf)

#plt.plot(t,decomp_tot_S1,label='S1')
plt.plot(t,decomp_tot_S2,label='S2')
plt.plot(t,decomp_tot_E,label='E')


plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()

type(decomp_tot_E[:,0])


#%%

#Step (4): Dynamic stock model of in-use wood materials


from dynamic_stock_model import DynamicStockModel



df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S2')
dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')


#product lifetime
#building materials
B = 35


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
#print(TestDSM1.o)
print(TestDSM2.o)
print(TestDSME.o)



#%%

#Step (5): Biomass growth
    
#Secondary forest biomass growth (Busch et al. 2019)

t = range(0,tf,1)


#calculate the biomass and carbon content of moist forest
def Cgrowth_1(t):
    return (44/12*1000*coeff_MF_nonpl*(np.sqrt(t)))

flat_list_moist = Cgrowth_1(t)

#calculate the biomass and carbon content of moist forest
def Cgrowth_2(t):
    return (44/12*1000*coeff_DF_nonpl*(np.sqrt(t)))

flat_list_dry = Cgrowth_2(t)


#plotting
plt.plot (t,flat_list_moist, label = 'Moist Forest, non-plantation')
plt.plot (t,flat_list_dry, label = 'Dry forest, non-plantation')



plt.xlim([0, 200])


plt.xlabel('Year')
plt.ylabel('Carbon stock (tC/ha)')
plt.title('')

plt.legend(loc='upper left')

plt.savefig('C:\Work\Programming\C_removal_fig.png', dpi=300)

plt.show()




###Yearly Sequestration 

###Moist Forest
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_moist'(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_moist = [p - q for q, p in zip(flat_list_moist, flat_list_moist[1:])]



#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_moist.insert(0,var)


#make 'flat_list_moist' elements negative numbers to denote sequestration
flat_list_moist = [ -x for x in flat_list_moist]

print(flat_list_moist)


#Dry forest
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_dry'(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_dry = [t - u for u, t in zip(flat_list_dry, flat_list_dry[1:])]



#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_dry.insert(0,var)

#make 'flat_list_dry' elements negative numbers to denote sequestration
flat_list_dry = [ -x for x in flat_list_dry]


print(flat_list_dry)

#%%

#Step(6): post-harvest processing of wood 


#post-harvest wood processing
#df1 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S1')
df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S2')
df3 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')


t = range(0,tf,1)


#PH_Emissions_HWP1_S1 = df1['PH_Emissions_HWP'].values
PH_Emissions_HWP1_S2 = df2['PH_Emissions_HWP'].values
PH_Emissions_HWP1_E = df3['PH_Emissions_HWP'].values



#%%

#Step (7_1): landfill gas decomposition (CH4)

#CH4 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl



#S2
df2_CH4 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S2')

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
dfE_CH4 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')

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

#type(decomp_tot_CH4_S1[:,0])


#%%

#Step (7_2): landfill gas decomposition (CO2)

#CO2 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl


#S2
df2_CO2 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S2')

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
dfE_CO2 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')

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

Emissions_PF_SF_S2 = [c_firewood_energy_S2, decomp_tot_S2[:,0], TestDSM2.o, PH_Emissions_HWP1_S2, decomp_tot_CO2_S2[:,0]]
Emissions_PF_SF_E = [c_firewood_energy_E, c_pellets_E, decomp_tot_E[:,0], TestDSME.o, PH_Emissions_HWP1_E, decomp_tot_CO2_E[:,0]]


Emissions_PF_SF_S2 = [sum(x) for x in zip(*Emissions_PF_SF_S2)]
Emissions_PF_SF_E = [sum(x) for x in zip(*Emissions_PF_SF_E)]

print(Emissions_PF_SF_S2)


#CH4_S2
Emissions_CH4_PF_SF_S2 = decomp_tot_CH4_S2[:,0]


#CH4_E
Emissions_CH4_PF_SF_E = decomp_tot_CH4_E[:,0]



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
#Col2_1 = Emissions_PF_SF_S1
Col2_2 = Emissions_PF_SF_S2
Col2_E = Emissions_PF_SF_E
#Col3_1 = Emissions_CH4_PF_SF_S1
Col3_2 = Emissions_CH4_PF_SF_S2
Col3_E = Emissions_CH4_PF_SF_E
Col4 = flat_list_moist
Col5 = Emission_ref
Col6 = flat_list_dry



#S2
df2_moi = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_2,'kg_CH4':Col3_2,'kg_CO2_seq':Col4,'emission_ref':Col5})
df2_dry = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_2,'kg_CH4':Col3_2,'kg_CO2_seq':Col6,'emission_ref':Col5})

#E
dfE_moi = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_E,'kg_CH4':Col3_E,'kg_CO2_seq':Col4,'emission_ref':Col5})
dfE_dry = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_E,'kg_CH4':Col3_E,'kg_CO2_seq':Col6,'emission_ref':Col5})

writer = pd.ExcelWriter('emissions_seq_PF_SF_dim.xlsx', engine = 'xlsxwriter')


df2_moi.to_excel(writer, sheet_name = 'S2_moist', header=True, index=False)
df2_dry.to_excel(writer, sheet_name = 'S2_dry', header=True, index=False)
dfE_moi.to_excel(writer, sheet_name = 'E_moist', header=True, index=False)
dfE_dry.to_excel(writer, sheet_name = 'E_dry', header=True, index=False)


writer.save()
writer.close()


#%%

## DYNAMIC LCA, for wood-based scenarios

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

#wood-based 
#read S2_moist
df = pd.read_excel('emissions_seq_PF_SF_dim.xlsx', 'S2_moist') # can also index sheet by name or fetch all sheets
emission_CO2_S2moi = df['kg_CO2'].tolist()
emission_CH4_S2moi = df['kg_CH4'].tolist()
emission_CO2_seq_S2moi = df['kg_CO2_seq'].tolist()

emission_CO2_ref = df['emission_ref'].tolist() 

#read S2_dry
df = pd.read_excel('emissions_seq_PF_SF_dim.xlsx', 'S2_dry')
emission_CO2_S2dry = df['kg_CO2'].tolist()
emission_CH4_S2dry = df['kg_CH4'].tolist()
emission_CO2_seq_S2dry = df['kg_CO2_seq'].tolist()


#read E_moist
df = pd.read_excel('emissions_seq_PF_SF_dim.xlsx', 'E_moist') # can also index sheet by name or fetch all sheets
emission_CO2_Emoi = df['kg_CO2'].tolist()
emission_CH4_Emoi = df['kg_CH4'].tolist()
emission_CO2_seq_Emoi = df['kg_CO2_seq'].tolist()


#read E_dry
df = pd.read_excel('emissions_seq_PF_SF_dim.xlsx', 'E_dry')
emission_CO2_Edry = df['kg_CO2'].tolist()
emission_CH4_Edry = df['kg_CH4'].tolist()
emission_CO2_seq_Edry = df['kg_CO2_seq'].tolist()

#%%

#Step (14): import emission data from the counter-use of non-renewable materials/energy scenarios (NR)
 

#read S2
df = pd.read_excel('PF_SF_dim.xlsx', 'NonRW_PF_SF_S2') # can also index sheet by name or fetch all sheets
emission_NonRW_PF_SF_S2 = df['NonRW_emissions'].tolist()
emission_NonRW_PF_SF_S2_seq = df['kg_CO2_seq'].tolist()

emission_CO2_ref = df['emission_ref'].tolist() 

#read E
df = pd.read_excel('PF_SF_dim.xlsx', 'NonRW_PF_SF_E') # can also index sheet by name or fetch all sheets
emission_NonRW_PF_SF_E = df['NonRW_emissions'].tolist()
emission_NonRW_PF_SF_E_seq = df['kg_CO2_seq'].tolist()



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

#Step (16): Calculate instantaneous global warming impact (GWI) 

##wood-based 
#S2_moist
t = np.arange(0,tf-1,1)

matrix_GWI_S2moi = (tf-1,3)
GWI_inst_S2moi = np.zeros(matrix_GWI_S2moi)



for t in range(0,tf-1):
    GWI_inst_S2moi[t,0] = np.sum(np.multiply(emission_CO2_S2moi,DCF_CO2_ti[:,t]))
    GWI_inst_S2moi[t,1] = np.sum(np.multiply(emission_CH4_S2moi,DCF_CH4_ti[:,t]))
    GWI_inst_S2moi[t,2] = np.sum(np.multiply(emission_CO2_seq_S2moi,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2moi = (tf-1,1)
GWI_inst_tot_S2moi = np.zeros(matrix_GWI_tot_S2moi)

GWI_inst_tot_S2moi[:,0] = np.array(GWI_inst_S2moi[:,0] + GWI_inst_S2moi[:,1] + GWI_inst_S2moi[:,2])
  
print(GWI_inst_tot_S2moi[:,0])

#S2_dry
t = np.arange(0,tf-1,1)

matrix_GWI_S2dry = (tf-1,3)
GWI_inst_S2dry = np.zeros(matrix_GWI_S2dry)



for t in range(0,tf-1):
    GWI_inst_S2dry[t,0] = np.sum(np.multiply(emission_CO2_S2dry,DCF_CO2_ti[:,t]))
    GWI_inst_S2dry[t,1] = np.sum(np.multiply(emission_CH4_S2dry,DCF_CH4_ti[:,t]))
    GWI_inst_S2dry[t,2] = np.sum(np.multiply(emission_CO2_seq_S2dry,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2dry = (tf-1,1)
GWI_inst_tot_S2dry = np.zeros(matrix_GWI_tot_S2dry)

GWI_inst_tot_S2dry[:,0] = np.array(GWI_inst_S2dry[:,0] + GWI_inst_S2dry[:,1] + GWI_inst_S2dry[:,2])
  
print(GWI_inst_tot_S2dry[:,0])


#E_moist
t = np.arange(0,tf-1,1)

matrix_GWI_Emoi = (tf-1,3)
GWI_inst_Emoi = np.zeros(matrix_GWI_Emoi)



for t in range(0,tf-1):
    GWI_inst_Emoi[t,0] = np.sum(np.multiply(emission_CO2_Emoi,DCF_CO2_ti[:,t]))
    GWI_inst_Emoi[t,1] = np.sum(np.multiply(emission_CH4_Emoi,DCF_CH4_ti[:,t]))
    GWI_inst_Emoi[t,2] = np.sum(np.multiply(emission_CO2_seq_Emoi,DCF_CO2_ti[:,t]))

matrix_GWI_tot_Emoi = (tf-1,1)
GWI_inst_tot_Emoi = np.zeros(matrix_GWI_tot_Emoi)

GWI_inst_tot_Emoi[:,0] = np.array(GWI_inst_Emoi[:,0] + GWI_inst_Emoi[:,1] + GWI_inst_Emoi[:,2])
  
print(GWI_inst_tot_Emoi[:,0])


#E_dry
t = np.arange(0,tf-1,1)

matrix_GWI_Edry = (tf-1,3)
GWI_inst_Edry = np.zeros(matrix_GWI_Edry)



for t in range(0,tf-1):
    GWI_inst_Edry[t,0] = np.sum(np.multiply(emission_CO2_Edry,DCF_CO2_ti[:,t]))
    GWI_inst_Edry[t,1] = np.sum(np.multiply(emission_CH4_Edry,DCF_CH4_ti[:,t]))
    GWI_inst_Edry[t,2] = np.sum(np.multiply(emission_CO2_seq_Edry,DCF_CO2_ti[:,t]))

matrix_GWI_tot_Edry = (tf-1,1)
GWI_inst_tot_Edry = np.zeros(matrix_GWI_tot_Edry)

GWI_inst_tot_Edry[:,0] = np.array(GWI_inst_Edry[:,0] + GWI_inst_Edry[:,1] + GWI_inst_Edry[:,2])
  
print(GWI_inst_tot_Edry[:,0])


##NonRW
#S2
t = np.arange(0,tf-1,1)

matrix_GWI_S2 = (tf-1,2)
GWI_inst_S2 = np.zeros(matrix_GWI_S2)


for t in range(0,tf-1):
    GWI_inst_S2[t,0] = np.sum(np.multiply(emission_NonRW_PF_SF_S2,DCF_CO2_ti[:,t]))
    GWI_inst_S2[t,1] = np.sum(np.multiply(emission_NonRW_PF_SF_S2_seq,DCF_CO2_ti[:,t]))


matrix_GWI_tot_S2 = (tf-1,1)
GWI_inst_tot_S2 = np.zeros(matrix_GWI_tot_S2)

GWI_inst_tot_S2[:,0] = np.array(GWI_inst_S2[:,0] + GWI_inst_S2[:,1])
  
print(GWI_inst_tot_S2[:,0])


#E
t = np.arange(0,tf-1,1)

matrix_GWI_E = (tf-1,2)
GWI_inst_E = np.zeros(matrix_GWI_E)


for t in range(0,tf-1):
    GWI_inst_E[t,0] = np.sum(np.multiply(emission_NonRW_PF_SF_E,DCF_CO2_ti[:,t]))
    GWI_inst_E[t,1] = np.sum(np.multiply(emission_NonRW_PF_SF_E_seq,DCF_CO2_ti[:,t]))

matrix_GWI_tot_E = (tf-1,1)
GWI_inst_tot_E = np.zeros(matrix_GWI_tot_E)

GWI_inst_tot_E[:,0] = np.array(GWI_inst_E[:,0] + GWI_inst_E[:,1])
  
print(GWI_inst_tot_E[:,0])





t = np.arange(0,tf-1,1)

#create zero list to highlight the horizontal line for 0
def zerolistmaker(n):
    listofzeros = [0] * (n)
    return listofzeros


#convert to flat list
GWI_inst_tot_S2 = np.array([item for sublist in GWI_inst_tot_S2 for item in sublist])
GWI_inst_tot_E = np.array([item for sublist in GWI_inst_tot_E for item in sublist])

GWI_inst_tot_S2moi = np.array([item for sublist in GWI_inst_tot_S2moi for item in sublist])
GWI_inst_tot_S2dry = np.array([item for sublist in GWI_inst_tot_S2dry for item in sublist])
GWI_inst_tot_Emoi = np.array([item for sublist in GWI_inst_tot_Emoi for item in sublist])
GWI_inst_tot_Edry = np.array([item for sublist in GWI_inst_tot_Edry for item in sublist])



plt.plot(t, GWI_inst_tot_S2, color='deeppink', label='NR_M', ls='--')
plt.plot(t, GWI_inst_tot_E, color='royalblue', label='NR_E', ls='--')


plt.plot(t, GWI_inst_tot_S2moi, color='lightcoral', label='M_moist')
plt.plot(t, GWI_inst_tot_S2dry, color='deeppink', label='M_dry')
plt.plot(t, GWI_inst_tot_Emoi, color='royalblue', label='E_moist')
plt.plot(t, GWI_inst_tot_Edry, color='deepskyblue', label='E_dry')

plt.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWI_inst_tot_S1, GWI_inst_tot_S2, color='lightcoral', alpha=0.3)
#plt.fill_between(t, GWI_inst_tot_E, GWI_inst_tot_S2, color='lightcoral', alpha=0.3)

plt.grid(True)

plt.legend(bbox_to_anchor=(1.04,1), loc='upper left', frameon=False)


plt.xlim(0,200)
plt.ylim(-0.5e-9,1.4e-9)


plt.title('Instantaneous GWI, PF_SF')

plt.xlabel('Time (year)')
#plt.ylabel('GWI_inst (10$^{-13}$ W/m$^2$)')
plt.ylabel('GWI_inst (W/m$^2$)')


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_inst_Non_RW_PF_SF_dim', dpi=300)

plt.show()


#%%

#Step (17): Calculate cumulative global warming impact (GWI)

##wood-based 
GWI_cum_S2moi = np.cumsum(GWI_inst_tot_S2moi)
GWI_cum_S2dry = np.cumsum(GWI_inst_tot_S2dry)
GWI_cum_Emoi = np.cumsum(GWI_inst_tot_Emoi)
GWI_cum_Edry = np.cumsum(GWI_inst_tot_Edry)


##NonRW
GWI_cum_S2 = np.cumsum(GWI_inst_tot_S2)

GWI_cum_E = np.cumsum(GWI_inst_tot_E)



plt.xlabel('Time (year)')
plt.ylabel('GWI_cum (10$^{-11}$ W/m$^2$)')
plt.ylabel('GWI_cum (W/m$^2$)')


plt.xlim(0,200)
plt.ylim(-0.3e-7,2e-7)

plt.title('Cumulative GWI, PF_SF')

#plt.plot(t, GWI_cum_S1, color='mediumseagreen', label='NR_M_EC', ls='--', alpha=0.55)
plt.plot(t, GWI_cum_S2, color='deeppink', label='NR_M', ls='--')
plt.plot(t, GWI_cum_E, color='royalblue', label='NR_E', ls='--')


#plt.plot(t, GWI_cum_S1moi, color='mediumseagreen', label='M_EC_moist')
#plt.plot(t, GWI_cum_S1dry, color='forestgreen', label='M_EC_dry')
plt.plot(t, GWI_cum_S2moi, color='lightcoral', label='M_moist')
plt.plot(t, GWI_cum_S2dry, color='deeppink', label='M_dry')
plt.plot(t, GWI_cum_Emoi, color='royalblue', label='E_moist')
plt.plot(t, GWI_cum_Edry, color='deepskyblue', label='E_dry')

plt.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWI_cum_S1, GWI_cum_S2, color='lightcoral', alpha=0.3)
#plt.fill_between(t, GWI_cum_E, GWI_cum_S2, color='lightcoral', alpha=0.3)

plt.grid(True)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_cum_NonRW_PF_SF_dim', dpi=300)

plt.show()

#%%

#Step (18): Determine the Instantenous and Cumulative GWI for the  emission reference (1 kg CO2 emission at time zero) before performing dynamic GWP calculation

#determine the GWI inst for the emission reference (1 kg CO2 emission at time zero)

t = np.arange(0,tf-1,1)

matrix_GWI_ref = (tf-1,1)
GWI_inst_ref = np.zeros(matrix_GWI_S2moi)

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

##Wood-based 
GWP_dyn_cum_S2moi = [x/(y*1000) for x,y in zip(GWI_cum_S2moi, GWI_cum_ref)]
GWP_dyn_cum_S2dry = [x/(y*1000) for x,y in zip(GWI_cum_S2dry, GWI_cum_ref)]
GWP_dyn_cum_Emoi = [x/(y*1000) for x,y in zip(GWI_cum_Emoi, GWI_cum_ref)]
GWP_dyn_cum_Edry = [x/(y*1000) for x,y in zip(GWI_cum_Edry, GWI_cum_ref)]



##NonRW

GWP_dyn_cum_NonRW_S2 = [x/(y*1000) for x,y in zip(GWI_cum_S2, GWI_cum_ref)]

GWP_dyn_cum_NonRW_E = [x/(y*1000) for x,y in zip(GWI_cum_E, GWI_cum_ref)]


fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)



ax.plot(t, GWP_dyn_cum_NonRW_S2, color='deeppink', ls='--', label='NR_M')
ax.plot(t, GWP_dyn_cum_NonRW_E, color='royalblue', ls='--', label='NR_E')


ax.plot(t, GWP_dyn_cum_S2moi, color='lightcoral', label='M_moist')
ax.plot(t, GWP_dyn_cum_S2dry, color='deeppink', label='M_dry')
ax.plot(t, GWP_dyn_cum_Emoi, color='royalblue', label='E_moist')
ax.plot(t, GWP_dyn_cum_Edry, color='deepskyblue', label='E_dry')

ax.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWP_dyn_cum_NonRW_S1, GWP_dyn_cum_NonRW_S2, color='lightcoral', alpha=0.3)
#plt.fill_between(t, GWP_dyn_cum_NonRW_E, GWP_dyn_cum_NonRW_S2, color='lightcoral', alpha=0.3)

plt.grid(True)

ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={'size': 13}, frameon=False)

ax.set_xlim(0,200)
ax.set_ylim(-250,1400)


ax.set_xlabel('Time (year)')
ax.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)')

ax.set_title('Dynamic GWP, PF_SF')


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_cum_NonRW_PF_SF_dim', dpi=300)


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
Col_GI_3  = GWI_inst_tot_S2moi
Col_GI_4  = GWI_inst_tot_S2dry
Col_GI_5  = GWI_inst_tot_Emoi
Col_GI_6  = GWI_inst_tot_Edry


#print(Col_GI_1)
#print(np.shape(Col_GI_1))

#GWI_inst from counter use scenarios
Col_GI_8  = GWI_inst_tot_S2
Col_GI_9  = GWI_inst_tot_E

#print(Col_GI_7)
#print(np.shape(Col_GI_7))


#create column results
    
##GWI_cumulative
#GWI_cumulative from wood-based scenarios
Col_GC_3 = GWI_cum_S2moi
Col_GC_4 = GWI_cum_S2dry
Col_GC_5 = GWI_cum_Emoi
Col_GC_6 = GWI_cum_Edry

#GWI_cumulative from counter use scenarios
Col_GC_8 = GWI_cum_S2
Col_GC_9 = GWI_cum_E


#create column results

##GWPdyn
#GWPdyn from wood-based scenarios
Col_GWP_3 = GWP_dyn_cum_S2moi
Col_GWP_4 = GWP_dyn_cum_S2dry
Col_GWP_5 = GWP_dyn_cum_Emoi
Col_GWP_6 = GWP_dyn_cum_Edry

#GWPdyn from counter use scenarios
Col_GWP_8 = GWP_dyn_cum_NonRW_S2
Col_GWP_9 = GWP_dyn_cum_NonRW_E


#Create colum results
df_GI = pd.DataFrame.from_dict({'Year':Col1,'M_moist (W/m2)':Col_GI_3,'M_dry (W/m2)':Col_GI_4,
                                 'E_moist (W/m2)':Col_GI_5, 'E_dry (W/m2)':Col_GI_6,
                                   'NR_M (W/m2)':Col_GI_8, 'NR_E (W/m2)':Col_GI_9})


df_GC = pd.DataFrame.from_dict({'Year':Col1,'M_moist (W/m2)':Col_GC_3,'M_dry (W/m2)':Col_GC_4,
                                 'E_moist (W/m2)':Col_GC_5, 'E_dry (W/m2)':Col_GC_6, 
                                   'NR_M (W/m2)':Col_GC_8, 'NR_E (W/m2)':Col_GC_9})

df_GWP = pd.DataFrame.from_dict({'Year':Col1,'M_moist (t-CO2-eq)':Col_GWP_3,'M_dry (t-CO2-eq)':Col_GWP_4, 
                                  'E_moist (t-CO2-eq)':Col_GWP_5, 'E_dry (t-CO2-eq)':Col_GWP_6,
                                   'NR_M (t-CO2-eq)':Col_GWP_8, 'NR_E (t-CO2-eq)':Col_GWP_9})

    
#Export to excel
writer = pd.ExcelWriter('GraphResults_PF_SF_dim.xlsx', engine = 'xlsxwriter')



df_GI.to_excel(writer, sheet_name = 'GWI_Inst_PF_SF', header=True, index=False)

df_GC.to_excel(writer, sheet_name = 'Cumulative GWI_PF_SF', header=True, index=False)

df_GWP.to_excel(writer, sheet_name = 'GWPdyn_PF_SF', header=True, index=False)



writer.save()
writer.close()

#%%

#Step (21): Generate the excel file for the individual carbon emission and sequestration flows

#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
#print (year)




division = 1000*44/12
division_CH4 = 1000*16/12



flat_list_moist = [x/division for x in flat_list_moist]
flat_list_dry = [x/division for x in flat_list_dry]


#M
c_firewood_energy_S2 = [x/division for x in c_firewood_energy_S2]
decomp_tot_S2[:,0] = [x/division for x in decomp_tot_S2[:,0]]
TestDSM2.o = [x/division for x in TestDSM2.o]
PH_Emissions_HWP1_S2 = [x/division for x in PH_Emissions_HWP1_S2]
#OC_storage_S2 = [x/division for x in OC_storage_S2]
decomp_tot_CO2_S2[:,0] = [x/division for x in decomp_tot_CO2_S2[:,0]]

decomp_tot_CH4_S2[:,0] = [x/division_CH4 for x in decomp_tot_CH4_S2[:,0]]


#E
c_firewood_energy_E = [x/division for x in c_firewood_energy_E]
c_pellets_E = [x/division for x in c_pellets_E]
decomp_tot_E[:,0] = [x/division for x in decomp_tot_E[:,0]]
TestDSME.o = [x/division for x in TestDSME.o]
PH_Emissions_HWP1_E = [x/division for x in PH_Emissions_HWP1_E]
#OC_storage_E = [x/division for x in OC_storage_E]
decomp_tot_CO2_E[:,0] = [x/division for x in decomp_tot_CO2_E]

decomp_tot_CH4_E[:,0] = [x/division_CH4 for x in decomp_tot_CH4_E]



#landfill aggregate flows


Landfill_decomp_S2 = decomp_tot_CH4_S2, decomp_tot_CO2_S2
Landfill_decomp_E = decomp_tot_CH4_E, decomp_tot_CO2_E



Landfill_decomp_S2 = [sum(x) for x in zip(*Landfill_decomp_S2)]
Landfill_decomp_E = [sum(x) for x in zip(*Landfill_decomp_E)]

Landfill_decomp_S2 = [item for sublist in Landfill_decomp_S2 for item in sublist]
Landfill_decomp_E = [item for sublist in Landfill_decomp_E for item in sublist]



#M
Column1 = year
Column2 = c_firewood_energy_S2
Column3 = decomp_tot_S2[:,0]
Column4 = TestDSM2.o
Column5 = PH_Emissions_HWP1_S2
Column6 = Landfill_decomp_S2
#Column6_1 = OC_storage_S2
Column7 = flat_list_moist
Column8 = flat_list_dry


#E
Column9 = c_firewood_energy_E
Column10 = c_pellets_E
Column11 = decomp_tot_E[:,0]
Column12 = TestDSME.o
Column13 = PH_Emissions_HWP1_E
Column14 = Landfill_decomp_E
#Column14_1 = OC_storage_E




#M
dfM_moi = pd.DataFrame.from_dict({'Year':Column1,'F0-1: Biomass C sequestration (moist) (t-C)':Column7,
                                  #'9: Landfill storage (t-C)':Column6_1,
                                  'F1-0: Residue decomposition (t-C)':Column3,
                                  'F6-0-1: Emissions from firewood/other energy use (t-C)':Column2, 
                                  'F8-0: Operational stage/processing emissions (t-C)':Column5, 
                                  'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column4,
                                  'F7-0: Landfill gas decomposition (t-C)':Column6,})
dfM_dry = pd.DataFrame.from_dict({'Year':Column1, 'F0-1: Biomass C sequestration (dry) (t-C)':Column8,
                                  #'9: Landfill storage (t-C)':Column6_1,
                                  'F1-0: Residue decomposition  (t-C)':Column3,
                                  'F6-0-1: Emissions from firewood/other energy use (t-C)':Column2,
                                  'F8-0: Operational stage/processing emissions (t-C)':Column5,
                                  'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column4,
                                  'F7-0: Landfill gas decomposition (t-C)':Column6})
  
#E
dfE_moi = pd.DataFrame.from_dict({'Year':Column1, 'F0-1: Biomass C sequestration (moist) (t-C)':Column7,
                                  #'9: Landfill storage (t-C)':Column14_1,
                                  'F1-0: Residue decomposition (t-C)':Column11,
                                  'F6-0-1: Emissions from firewood/other energy use (t-C)':Column9, 
                                  'F8-0: Operational stage/processing emissions (t-C)':Column13,
                                  'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column12,
                                  'F7-0: Landfill gas decomposition (t-C)':Column14,
                                  'F4-0: Emissions from wood pellets use (t-C)':Column10})
    
dfE_dry = pd.DataFrame.from_dict({'Year':Column1, 'F0-1: Biomass C sequestration (dry) (t-C)':Column8,
                                  #'9: Landfill storage (t-C)':Column14_1,
                                  'F1-0: Residue decomposition (t-C)':Column11, 
                                  'F6-0-1: Emissions from firewood/other energy use (t-C)':Column9,
                                  'F8-0: Operational stage/processing emissions (t-C)':Column13,
                                  'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column12,
                                  'F7-0: Landfill gas decomposition (t-C)':Column14,
                                  'F4-0: Emissions from wood pellets use (t-C)':Column10})
    

    
writer = pd.ExcelWriter('C_flows_PF_SF_dim.xlsx', engine = 'xlsxwriter')


dfM_moi.to_excel(writer, sheet_name = 'M_moist', header=True, index=False)
dfM_dry.to_excel(writer, sheet_name = 'M_dry', header=True, index=False)
dfE_moi.to_excel(writer, sheet_name = 'E_moist', header=True, index=False)
dfE_dry.to_excel(writer, sheet_name = 'E_dry', header=True, index=False)



writer.save()
writer.close()


#%%

#Step (22): Plot of the individual carbon emission and sequestration flows for normal and symlog-scale graphs

#PF_SF_M

fig=plt.figure()
fig.show()
ax1=fig.add_subplot(111)

#plot
ax1.plot(t, flat_list_moist, color='yellowgreen', label='F0-1: Biomass C sequestration (moist)') 
ax1.plot(t, flat_list_dry, color='darkkhaki', label='F0-1: Biomass C sequestration (dry)') 
#ax1.plot(t, OC_storage_S2, color='darkturquoise', label='9: Landfill storage') 
ax1.plot(t, decomp_tot_S2[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax1.plot(t, c_firewood_energy_S2, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax1.plot(t, PH_Emissions_HWP1_S2, color='orange', label='F8-0: Operational stage/processing emissions') 
ax1.plot(t, TestDSM2.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax1.plot(t, Landfill_decomp_S2, color='yellow', label='F7-0: Landfill gas decomposition') 


ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax1.set_xlim(-1,200)

ax1.set_yscale('symlog')

ax1.set_xlabel('Time (year)')
ax1.set_ylabel('C flows(t-C) (symlog)')

ax1.set_title('Carbon flow, PF_SF_M (symlog-scale)')

plt.show()


#%%

#plotting the individual C flows

#PF_SF_M

f, (ax_a, ax_b) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax_a.plot(t, flat_list_moist, color='yellowgreen', label='F0-1: Biomass C sequestration (moist)') 
ax_a.plot(t, flat_list_dry, color='darkkhaki', label='F0-1: Biomass C sequestration (dry)') 
#ax_a.plot(t, OC_storage_S2, color='darkturquoise', label='9: Landfill storage') 
ax_a.plot(t, decomp_tot_S2[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax_a.plot(t, c_firewood_energy_S2, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax_a.plot(t, PH_Emissions_HWP1_S2, color='orange', label='F8-0: Operational stage/processing emissions') 
ax_a.plot(t, TestDSM2.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax_a.plot(t, Landfill_decomp_S2, color='yellow', label='F7-0: Landfill gas decomposition') 


ax_b.plot(t, c_firewood_energy_S2, color='mediumseagreen') 
ax_b.plot(t, decomp_tot_S2[:,0], color='lightcoral')
ax_b.plot(t, TestDSM2.o, color='royalblue') 
ax_b.plot(t, PH_Emissions_HWP1_S2, color='orange') 
#ax_b.plot(t, OC_storage_S2, color='darkturquoise')
ax_b.plot(t, Landfill_decomp_S2, color='yellow')
ax_b.plot(t, flat_list_moist, color='yellowgreen') 
ax_b.plot(t, flat_list_dry, color='darkkhaki') 

# zoom-in / limit the view to different portions of the data
ax_a.set_xlim(-1,200)

ax_a.set_ylim(60, 75)  
ax_b.set_ylim(-25, 15)  

# hide the spines between ax and ax2
ax_a.spines['bottom'].set_visible(False)
ax_b.spines['top'].set_visible(False)
ax_a.xaxis.tick_top()
ax_a.tick_params(labeltop=False)  # don't put tick labels at the top
ax_b.xaxis.tick_bottom()

ax_a.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)


d = .012  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_a.transAxes, color='k', clip_on=False)
ax_a.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_b.transAxes)  # switch to the bottom axes
ax_b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



ax_b.set_xlabel('Time (year)')
ax_b.set_ylabel('C flows (t-C)')
ax_a.set_ylabel('C flows (t-C)')

ax_a.set_title('Carbon flow, PF_SF_M')
#plt.plot(t, Cflow_PF_SF_S1)
#plt.plot(t, Cflow_PF_SF_S2)
#plt.plot(t, Cflow_PF_SF_E)
#plt.xlim([0, 200])

plt.show()


#%%

#plot for the individual carbon flows - symlog-scale graphs

#PF_SF_E

fig=plt.figure()
fig.show()
ax2=fig.add_subplot(111)


#plot
ax2.plot(t, flat_list_moist, color='yellowgreen', label='F0-1: Biomass C sequestration (moist)') 
ax2.plot(t, flat_list_dry, color='darkkhaki', label='F0-1: Biomass C sequestration (dry)') 
#ax2.plot(t, OC_storage_E, color='darkturquoise', label= '9: Landfill storage') 
ax2.plot(t, decomp_tot_E[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2.plot(t, c_firewood_energy_E, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
#ax2.plot(t, TestDSME.o, color='royalblue', label='in-use stock output') 
ax2.plot(t, PH_Emissions_HWP1_E, color='orange', label='F8-0: Operational stage/processing emissions') 
ax2.plot(t, Landfill_decomp_E, color='yellow', label= 'F7-0: Landfill gas decomposition') 
ax2.plot(t, c_pellets_E, color='slategrey', label='F4-0: Emissions from wood pellets use') 



ax2.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2.set_xlim(-1,200)

ax2.set_yscale('symlog')

ax2.set_xlabel('Time (year)')
ax2.set_ylabel('C flows(t-C) (symlog)')

ax2.set_title('Carbon flow, PF_SF_E (symlog-scale)')

plt.show()



#%%

#plotting the individual C flows

#PF_SF_E

f, (ax_c, ax_d) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax_c.plot(t, flat_list_moist, color='yellowgreen', label='F0-1: Biomass C sequestration (moist)') 
ax_c.plot(t, flat_list_dry, color='darkkhaki', label='F0-1: Biomass C sequestration (dry)') 
#ax_c.plot(t, OC_storage_E, color='darkturquoise', label= '9: Landfill storage') 
ax_c.plot(t, decomp_tot_E[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax_c.plot(t, c_firewood_energy_E, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
#ax_c.plot(t, TestDSME.o, color='royalblue', label='in-use stock output') 
ax_c.plot(t, PH_Emissions_HWP1_E, color='orange', label='F8-0: Operational stage/processing emissions') 
ax_c.plot(t, Landfill_decomp_E, color='yellow', label= 'F7-0: Landfill gas decomposition') 
ax_c.plot(t, c_pellets_E, color='slategrey', label='F4-0: Emissions from wood pellets use') 


ax_d.plot(t, c_firewood_energy_E, color='mediumseagreen') 
ax_d.plot(t, c_pellets_E, color='slategrey')
ax_d.plot(t, decomp_tot_E[:,0], color='lightcoral')
ax_d.plot(t, TestDSME.o, color='royalblue') 
ax_d.plot(t, PH_Emissions_HWP1_E, color='orange') 
#ax_d.plot(t, OC_storage_E, color='darkturquoise')
ax_d.plot(t, Landfill_decomp_E, color='yellow')
ax_d.plot(t, flat_list_moist, color='yellowgreen') 
ax_d.plot(t, flat_list_dry, color='darkkhaki') 

# zoom-in / limit the view to different portions of the data
ax_c.set_xlim(-1,200)

ax_c.set_ylim(170, 190)  
ax_d.set_ylim(-25, 25)  

# hide the spines between ax and ax2
ax_c.spines['bottom'].set_visible(False)
ax_d.spines['top'].set_visible(False)
ax_c.xaxis.tick_top()
ax_c.tick_params(labeltop=False)  # don't put tick labels at the top
ax_d.xaxis.tick_bottom()

ax_c.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

d = .012  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_c.transAxes, color='k', clip_on=False)
ax_c.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_c.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_d.transAxes)  # switch to the bottom axes
ax_d.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_d.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



ax_d.set_xlabel('Time (year)')
ax_d.set_ylabel('C flows (t-C)')
ax_c.set_ylabel('C flows (t-C)')

ax_c.set_title('Carbon flow, PF_SF_E')
#plt.plot(t, Cflow_PF_SF_S1)
#plt.plot(t, Cflow_PF_SF_S2)
#plt.plot(t, Cflow_PF_SF_E)
#plt.xlim([0, 200])

plt.show()

#%%

#Step (23): Generate the excel file for the net carbon balance

Agg_Cflow_PF_SF_S2moi = [c_firewood_energy_S2, decomp_tot_S2[:,0], TestDSM2.o, PH_Emissions_HWP1_S2, Landfill_decomp_S2, flat_list_moist]
Agg_Cflow_PF_SF_S2dry = [c_firewood_energy_S2, decomp_tot_S2[:,0], TestDSM2.o, PH_Emissions_HWP1_S2, Landfill_decomp_S2, flat_list_dry]
Agg_Cflow_PF_SF_Emoi = [c_firewood_energy_E, c_pellets_E, decomp_tot_E[:,0], TestDSME.o, PH_Emissions_HWP1_E, Landfill_decomp_E, flat_list_moist]
Agg_Cflow_PF_SF_Edry = [c_firewood_energy_E, c_pellets_E, decomp_tot_E[:,0], TestDSME.o, PH_Emissions_HWP1_E, Landfill_decomp_E, flat_list_dry]



Agg_Cflow_PF_SF_S2moi = [sum(x) for x in zip(*Agg_Cflow_PF_SF_S2moi)]
Agg_Cflow_PF_SF_S2dry = [sum(x) for x in zip(*Agg_Cflow_PF_SF_S2dry)]
Agg_Cflow_PF_SF_Emoi = [sum(x) for x in zip(*Agg_Cflow_PF_SF_Emoi)]
Agg_Cflow_PF_SF_Edry = [sum(x) for x in zip(*Agg_Cflow_PF_SF_Edry)]


fig=plt.figure()
fig.show()
ax3=fig.add_subplot(111)

# plot
ax3.plot(t, Agg_Cflow_PF_SF_S2moi, color='lightcoral', label='M_moist') 
ax3.plot(t, Agg_Cflow_PF_SF_S2dry, color='deeppink', label='M_dry') 
ax3.plot(t, Agg_Cflow_PF_SF_Emoi, color='royalblue', label='E_moist') 
ax3.plot(t, Agg_Cflow_PF_SF_Edry, color='deepskyblue', label='E_dry') 


ax3.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax3.set_xlim(-0.3,85)


#ax3.set_yscale('symlog')
 
ax3.set_xlabel('Time (year)')
ax3.set_ylabel('C flows (t-C)')

ax3.set_title('Aggr. C-emissions/sequestration flow, PF_SF')

plt.draw()

#create column year
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)

#Create colum results
dfM_PF_SF = pd.DataFrame.from_dict({'Year':year,'M_moist (t-C)':Agg_Cflow_PF_SF_S2moi, 'M_dry (t-C)':Agg_Cflow_PF_SF_S2dry,
                                         'E_moist (t-C)':Agg_Cflow_PF_SF_Emoi, 'E_dry (t-C)':Agg_Cflow_PF_SF_Edry})

    
#Export to excel
writer = pd.ExcelWriter('AggCFlow_PF_SF_dim.xlsx', engine = 'xlsxwriter')


dfM_PF_SF.to_excel(writer, sheet_name = 'PF_SF', header=True, index=False)

writer.save()
writer.close()



#%%

#Step (24): Plot the net carbon balance 


f, (ax3a, ax3b) = plt.subplots(2, 1, sharex=True)

# plot
ax3a.plot(t, Agg_Cflow_PF_SF_S2moi, color='lightcoral', label='M_moist') 
ax3a.plot(t, Agg_Cflow_PF_SF_S2dry, color='deeppink', label='M_dry') 
ax3a.plot(t, Agg_Cflow_PF_SF_Emoi, color='royalblue', label='E_moist') 
ax3a.plot(t, Agg_Cflow_PF_SF_Edry, color='deepskyblue', label='E_dry') 

ax3a.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)


ax3b.plot(t, Agg_Cflow_PF_SF_S2moi, color='lightcoral') 
ax3b.plot(t, Agg_Cflow_PF_SF_S2dry, color='deeppink') 
ax3b.plot(t, Agg_Cflow_PF_SF_Emoi, color='royalblue') 
ax3b.plot(t, Agg_Cflow_PF_SF_Edry, color='deepskyblue') 

ax3b.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

# zoom-in / limit the view to different portions of the data
ax3a.set_xlim(-0.35,85)
#ax3a.set_xlim(-1,200)
ax3a.set_ylim(210, 230)  

ax3b.set_xlim(-0.35,85)
#ax3b.set_xlim(-1,200)
ax3b.set_ylim(-15, 10)  



# hide the spines between ax and ax2
ax3a.spines['bottom'].set_visible(False)
ax3b.spines['top'].set_visible(False)
ax3a.xaxis.tick_top()
ax3a.tick_params(labeltop=False)  # don't put tick labels at the top
ax3b.xaxis.tick_bottom()

ax3a.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

d = .012  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax3a.transAxes, color='k', clip_on=False)
ax3a.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax3a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax3b.transAxes)  # switch to the bottom axes
ax3b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax3b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



ax3b.set_xlabel('Time (year)')
ax3b.set_ylabel('C flows (t-C)')
ax3a.set_ylabel('C flows (t-C)')

ax3a.set_title('Net carbon balance, PF_SF')

plt.show()

#%%

#Step (25): Generate the excel file for documentation of individual carbon flows in the system definition (Fig. 1)


#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)


df2 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S2')
dfE = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_E')



Column1 = year
division = 1000*44/12
division_CH4 = 1000*16/12


## S2moi
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


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2moi = (tf,1)
ForCstocks_S2moi = np.zeros(zero_matrix_ForCstocks_S2moi)

i = 0
ForCstocks_S2moi[0] = initAGB - flat_list_moist[0] - decomp_tot_S2[0] - HWP_logged_S2[0]

while i < tf-1:
    ForCstocks_S2moi[i+1] = np.array(ForCstocks_S2moi[i] - flat_list_moist[i+1] - decomp_tot_S2[i+1] - HWP_logged_S2[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
df2_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF_dim.xlsx', 'NonRW_PF_SF_S2')
NonRW_amount_S2 = df2_amount['NonRW_amount'].values

NonRW_amount_S2 = [x/1000 for x in NonRW_amount_S2]



##NonRW emissions (F9-0-2)
emission_NonRW_PF_SF_S2 = [x/division for x in emission_NonRW_PF_SF_S2]
    

#create columns
dfM_moi = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_moist,
                                    'F1-0 (t-C)': decomp_tot_S2[:,0],
                                    #'F1a-2 (t-C)': PF_S2_Ac_7y,
                                    #'F1c-2 (t-C)': FP_S2_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_S2, 
                                    'St-1 (t-C)':ForCstocks_S2moi[:,0], 
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
                                    'F8-0 (t-C)': PH_Emissions_HWP1_S2,
                                    'S9-0 (t)': NonRW_amount_S2, 
                                    'F9-0 (t-C)': emission_NonRW_PF_SF_S2,
                                    })

    
    
    
##S2dry
## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2dry = (tf,1)
ForCstocks_S2dry = np.zeros(zero_matrix_ForCstocks_S2dry)

i = 0
ForCstocks_S2dry[0] = initAGB - flat_list_dry[0] - decomp_tot_S2[0] - HWP_logged_S2[0]

while i < tf-1:
    ForCstocks_S2dry[i+1] = np.array(ForCstocks_S2dry[i] - flat_list_dry[i+1] - decomp_tot_S2[i+1] - HWP_logged_S2[i+1])
    i = i + 1

    
#create columns
dfM_dry = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_dry,
                                    'F1-0 (t-C)': decomp_tot_S2[:,0],
                                    #'F1a-2 (t-C)': PF_S2_Ac_7y,
                                    #'F1c-2 (t-C)': FP_S2_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_S2, 
                                    'St-1 (t-C)':ForCstocks_S2dry[:,0], 
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
                                    'F8-0 (t-C)': PH_Emissions_HWP1_S2,
                                    'S9-0 (t)': NonRW_amount_S2, 
                                    'F9-0 (t-C)': emission_NonRW_PF_SF_S2,
                                    })


##E_moi
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


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_Emoi = (tf,1)
ForCstocks_Emoi = np.zeros(zero_matrix_ForCstocks_Emoi)

i = 0
ForCstocks_Emoi[0] = initAGB - flat_list_moist[0] - decomp_tot_E[0] - HWP_logged_E[0]

while i < tf-1:
    ForCstocks_Emoi[i+1] = np.array(ForCstocks_Emoi[i] - flat_list_moist[i+1] - decomp_tot_E[i+1] - HWP_logged_E[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
dfE_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF_dim.xlsx', 'NonRW_PF_SF_E')
NonRW_amount_E = dfE_amount['NonRW_amount'].values

NonRW_amount_E = [x/1000 for x in NonRW_amount_E]



##NonRW emissions (F9-0-2)
emission_NonRW_PF_SF_E = [x/division for x in emission_NonRW_PF_SF_E]
    

#create columns
dfE_moi = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_moist,
                                    'F1-0 (t-C)': decomp_tot_E[:,0],
                                    #'F1a-2 (t-C)': PF_E_Ac_7y,
                                    #'F1c-2 (t-C)': FP_E_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_E, 
                                    'St-1 (t-C)':ForCstocks_Emoi[:,0], 
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
                                    'F8-0 (t-C)': PH_Emissions_HWP1_E,
                                    'S9-0 (t)': NonRW_amount_E, 
                                    'F9-0 (t-C)': emission_NonRW_PF_SF_E,
                                    })

    
    
    

##E_dry
## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_Edry = (tf,1)
ForCstocks_Edry = np.zeros(zero_matrix_ForCstocks_Edry)

i = 0
ForCstocks_Edry[0] = initAGB - flat_list_dry[0] - decomp_tot_E[0] - HWP_logged_E[0]

while i < tf-1:
    ForCstocks_Edry[i+1] = np.array(ForCstocks_Edry[i] - flat_list_dry[i+1] - decomp_tot_E[i+1] - HWP_logged_E[i+1])
    i = i + 1

#create columns
dfE_dry = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_dry,
                                    'F1-0 (t-C)': decomp_tot_E[:,0],
                                    #'F1a-2 (t-C)': PF_E_Ac_7y,
                                    #'F1c-2 (t-C)': FP_E_Ac_7y,
                                    'F1-2 (t-C)': HWP_logged_E, 
                                    'St-1 (t-C)':ForCstocks_Edry[:,0], 
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
                                    'F8-0 (t-C)': PH_Emissions_HWP1_E,
                                    'S9-0 (t)': NonRW_amount_E, 
                                    'F9-0 (t-C)': emission_NonRW_PF_SF_E,
                                    })



writer = pd.ExcelWriter('C_flows_SysDef_PF_SF_dim.xlsx', engine = 'xlsxwriter')


dfM_moi.to_excel(writer, sheet_name = 'PF_SF_M_moi', header=True, index=False)
dfM_dry.to_excel(writer, sheet_name = 'PF_SF_M_dry', header=True, index=False)
dfE_moi.to_excel(writer, sheet_name = 'PF_SF_E2_moi', header=True, index=False)
dfE_dry.to_excel(writer, sheet_name = 'PF_SF_E2_dry', header=True, index=False)


writer.save()
writer.close()


#%%