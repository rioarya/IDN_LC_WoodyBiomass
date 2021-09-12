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


#PF_PO Scenario

##Set parameters
#Parameters for primary forest
initAGB = 233   #t-C         #source: van Beijma et al. (2018)
initAGB_min = 233-72 #t-C
initAGB_max = 233 + 72 #t-C

#parameters for oil palm plantation. Source: Khasanah et al. (2015) 
tf_palmoil = 26 #years
a_nucleus = 2.8167
b_nucleus = 6.8648
a_plasma = 2.5449
b_plasma = 5.0007
c_cont_po_nucleus = 0.5448  #fraction of carbon content in biomass
c_cont_po_plasma = 0.5454 #fraction of carbon content in biomass
tf = 201 #years

a = 0.082
b = 2.53

#%%

#Step (2_1): C loss from the harvesting/clear cut

df2nu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2nu')
df2pl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2pl')
df3nu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Enu')
df3pl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Epl')

t = range(0,tf,1)

c_firewood_energy_S2nu = df2nu['Firewood_other_energy_use'].values
c_firewood_energy_S2pl = df2pl['Firewood_other_energy_use'].values
c_firewood_energy_Enu = df3nu['Firewood_other_energy_use'].values
c_firewood_energy_Epl = df3pl['Firewood_other_energy_use'].values


#%%

#Step (2_2): C loss from the harvesting/clear cut as wood pellets

dfEnu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Enu')
dfEpl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Epl')

c_pellets_Enu = dfEnu['Wood_pellets'].values
c_pellets_Epl = dfEpl['Wood_pellets'].values


#%%

#Step (3): Aboveground biomass (AGB) decomposition

df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2nu')

tf = 201

t = np.arange(tf)


def decomp(t,remainAGB):
    return (1-(1-np.exp(-a*t))**b)*remainAGB



#set zero matrix
output_decomp = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp[i:,i] = decomp(t[:len(t)-i],remain_part)

print(output_decomp[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix[:,i] = np.diff(output_decomp[:,i])
    i = i + 1 

print(subs_matrix[:,:4])
print(len(subs_matrix))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix = subs_matrix.clip(max=0)

print(subs_matrix[:,:4])

#make the results as absolute values
subs_matrix = abs(subs_matrix)
print(subs_matrix[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix)

subs_matrix = np.vstack((zero_matrix, subs_matrix))

print(subs_matrix[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot = (tf,1)
decomp_emissions = np.zeros(matrix_tot) 

i = 0
while i < tf:
    decomp_emissions[:,0] = decomp_emissions[:,0] + subs_matrix[:,i]
    i = i + 1

print(decomp_emissions[:,0])





#%%

#Step (4): Dynamic stock model of in-use wood materials


from dynamic_stock_model import DynamicStockModel


df2nu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2nu')
df2pl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2pl')
df3nu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Enu')
df3pl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Epl')

#product lifetime
#building materials
B = 50


TestDSM2nu = DynamicStockModel(t = df2nu['Year'].values, i = df2nu['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})
TestDSM2pl = DynamicStockModel(t = df2pl['Year'].values, i = df2pl['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})
TestDSM3nu = DynamicStockModel(t = df3nu['Year'].values, i = df3nu['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})
TestDSM3pl = DynamicStockModel(t = df3pl['Year'].values, i = df3pl['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([B]), 'StdDev': np.array([0.3*B])})


CheckStr2nu, ExitFlag2nu = TestDSM2nu.dimension_check()
CheckStr2pl, ExitFlag2pl = TestDSM2pl.dimension_check()
CheckStr3nu, ExitFlag3nu = TestDSM3nu.dimension_check()
CheckStr3pl, ExitFlag3pl = TestDSM3pl.dimension_check()


Stock_by_cohort2nu, ExitFlag2nu = TestDSM2nu.compute_s_c_inflow_driven()
Stock_by_cohort2pl, ExitFlag2pl = TestDSM2pl.compute_s_c_inflow_driven()
Stock_by_cohort3nu, ExitFlag3nu = TestDSM3nu.compute_s_c_inflow_driven()
Stock_by_cohort3pl, ExitFlag3pl = TestDSM3pl.compute_s_c_inflow_driven()



S2nu, ExitFlag2nu   = TestDSM2nu.compute_stock_total()
S2pl, ExitFlag2pl   = TestDSM2pl.compute_stock_total()
S3nu, ExitFlag3nu   = TestDSM3nu.compute_stock_total()
S3pl, ExitFlag3pl   = TestDSM3pl.compute_stock_total()


O_C2nu, ExitFlag2nu = TestDSM2nu.compute_o_c_from_s_c()
O_C2pl, ExitFlag2pl = TestDSM2pl.compute_o_c_from_s_c()
O_C3nu, ExitFlag3nu = TestDSM3nu.compute_o_c_from_s_c()
O_C3pl, ExitFlag3pl = TestDSM3pl.compute_o_c_from_s_c()


O2nu, ExitFlag2nu   = TestDSM2nu.compute_outflow_total()
O2pl, ExitFlag2pl   = TestDSM2pl.compute_outflow_total()
O3nu, ExitFlag3nu   = TestDSM3nu.compute_outflow_total()
O3pl, ExitFlag3pl   = TestDSM3pl.compute_outflow_total()

DS2nu, ExitFlag2nu  = TestDSM2nu.compute_stock_change()
DS2pl, ExitFlag2pl  = TestDSM2pl.compute_stock_change()
DS3nu, ExitFlag3nu  = TestDSM3nu.compute_stock_change()
DS3pl, ExitFlag3pl  = TestDSM3pl.compute_stock_change()


Bal2nu, ExitFlag2nu = TestDSM2nu.check_stock_balance()
Bal2pl, ExitFlag2pl = TestDSM2pl.check_stock_balance()
Bal3nu, ExitFlag3nu = TestDSM3nu.check_stock_balance()
Bal3pl, ExitFlag3pl = TestDSM3pl.check_stock_balance()


#print output flow
print(TestDSM2nu.o)
print(TestDSM2pl.o)
print(TestDSM3nu.o)
print(TestDSM3pl.o)


#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)

Column1 = year

#create columns
df = pd.DataFrame.from_dict({'Year':Column1, 
                                    'PF_PO_Mnu': TestDSM2nu.o,
                                    'PF_PO_Mpl': TestDSM2pl.o,
                                    'PF_PO_Enu': TestDSM3nu.o,
                                    'PF_PO_Epl': TestDSM3pl.o,
                                      })

writer = pd.ExcelWriter('OutDSM_PF_PO_Ret.xlsx', engine = 'xlsxwriter')


df.to_excel(writer, sheet_name = 'PF_PO_Ret', header=True, index=False)

writer.save()
writer.close()


#%%

#Step (5): Biomass growth
    
    
#Model I Oil Palm Biomass Growth (Khasanah et al. (2015))


A = range(0,tf_palmoil,1)


#calculate the biomass and carbon content of palm oil trees over time
def Y_nucleus(A):
    return (44/12*1000*c_cont_po_nucleus*(a_nucleus*A + b_nucleus))


output_Y_nucleus = np.array([Y_nucleus(Ai) for Ai in A])

print(output_Y_nucleus)



def Y_plasma(A):
    return (44/12*1000*c_cont_po_plasma*(a_plasma*A + b_plasma))

output_Y_plasma = np.array([Y_plasma(Ai) for Ai in A])

print(output_Y_plasma)



##8 times 25-year cycle of new AGB of oil palm, one year gap between the cycle
#nucleus
counter = range(0,8,1)

y_nucleus = []

for i in counter:
    y_nucleus.append(output_Y_nucleus)


    
flat_list_nucleus = []
for sublist in y_nucleus:
    for item in sublist:
        flat_list_nucleus.append(item)

#the length of the list is now 208, so we remove the last 7 elements of the list to make the len=tf
flat_list_nucleus = flat_list_nucleus[:len(flat_list_nucleus)-7]


#plasma
y_plasma = []

for i in counter:
    y_plasma.append(output_Y_plasma)
    
flat_list_plasma = []
for sublist in y_plasma:
    for item in sublist:
        flat_list_plasma.append(item)

#the length of the list is now 208, so we remove the last 7 elements of the list to make the len=tf
flat_list_plasma = flat_list_plasma[:len(flat_list_plasma)-7]


#plotting
t = range (0,tf,1)


plt.xlim([0, 200])



plt.plot(t, flat_list_nucleus)
plt.plot(t, flat_list_plasma, color='seagreen')

plt.fill_between(t, flat_list_nucleus, flat_list_plasma, color='darkseagreen', alpha=0.4)


plt.xlabel('Time (year)')
plt.ylabel('AGB (tCO2-eq/ha)')

plt.show()




###Yearly Sequestration 

###Nucleus
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_nucleus(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_nucleus = [p - q for q, p in zip(flat_list_nucleus, flat_list_nucleus[1:])]


#since there is no sequestration between the replanting year (e.g., year 25 to 26), we have to replace negative numbers in 'flat_list_nuclues' with 0 values
flat_list_nucleus = [0 if i < 0 else i for i in flat_list_nucleus]


#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_nucleus.insert(0,var)


#make 'flat_list_nucleus' elements negative numbers to denote sequestration
flat_list_nucleus = [ -x for x in flat_list_nucleus]

print(flat_list_nucleus)


#Plasma
#find the yearly sequestration by calculating the differences between elements in list 'flat_list_plasma(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
flat_list_plasma = [t - u for u, t in zip(flat_list_plasma, flat_list_plasma[1:])]

#since there is no sequestration between the replanting year (e.g., year 25 to 26), we have to replace negative numbers in 'flat_list_plasma' with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
flat_list_plasma = [0 if i < 0 else i for i in flat_list_plasma]

#insert 0 value to the list as the first element, because there is no sequestration in year 0 
var = 0 
flat_list_plasma.insert(0,var)

#make 'flat_list_plasma' elements negative numbers to denote sequestration
flat_list_plasma = [ -x for x in flat_list_plasma]


print(flat_list_plasma)


#%%

#Step(6): post-harvest processing of wood/palm oil


#post-harvest wood processing
df2nu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2nu')
df2pl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2pl')
dfEnu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Enu')
dfEpl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Epl')

t = range(0,tf,1)

PH_Emissions_HWP_S2nu = df2nu['PH_Emissions_HWP'].values
PH_Emissions_HWP_S2pl = df2pl['PH_Emissions_HWP'].values
PH_Emissions_HWP_Enu = df3pl['PH_Emissions_HWP'].values
PH_Emissions_HWP_Epl = df3pl['PH_Emissions_HWP'].values


#post-harvest palm oil processing
df2nu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2nu')
df2pl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2pl')
dfEnu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Enu')
dfEpl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Epl')

t = range(0,tf,1)

PH_Emissions_PO_S2nu = df2nu['PH_Emissions_PO'].values
PH_Emissions_PO_S2pl = df2pl['PH_Emissions_PO'].values
PH_Emissions_PO_Enu = df3pl['PH_Emissions_PO'].values
PH_Emissions_PO_Epl = df3pl['PH_Emissions_PO'].values


#%%

#Step (7_1): landfill gas decomposition (CH4)


#CH4 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl


#S2nu
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_S2nu')

tf = 201

t = np.arange(tf)


def decomp_CH4_S2nu(t,remainAGB_CH4_S2nu):
    return (1-(1-np.exp(-k*t)))*remainAGB_CH4_S2nu



#set zero matrix
output_decomp_CH4_S2nu = np.zeros((len(t),len(df['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_S2nu in enumerate(df['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_S2nu[i:,i] = decomp_CH4_S2nu(t[:len(t)-i],remain_part_CH4_S2nu)

print(output_decomp_CH4_S2nu[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_S2nu = np.zeros((len(t)-1,len(df['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_S2nu[:,i] = np.diff(output_decomp_CH4_S2nu[:,i])
    i = i + 1 

print(subs_matrix_CH4_S2nu[:,:4])
print(len(subs_matrix_CH4_S2nu))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_S2nu = subs_matrix_CH4_S2nu.clip(max=0)

print(subs_matrix_CH4_S2nu[:,:4])

#make the results as absolute values
subs_matrix_CH4_S2nu = abs(subs_matrix_CH4_S2nu)
print(subs_matrix_CH4_S2nu[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_S2nu = np.zeros((len(t)-200,len(df['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_S2nu)

subs_matrix_CH4_S2nu = np.vstack((zero_matrix_CH4_S2nu, subs_matrix_CH4_S2nu))

print(subs_matrix_CH4_S2nu[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_S2nu = (tf,1)
decomp_tot_CH4_S2nu = np.zeros(matrix_tot_CH4_S2nu) 

i = 0
while i < tf:
    decomp_tot_CH4_S2nu[:,0] = decomp_tot_CH4_S2nu[:,0] + subs_matrix_CH4_S2nu[:,i]
    i = i + 1

print(decomp_tot_CH4_S2nu[:,0])



#S2pl
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_S2pl')

tf = 201

t = np.arange(tf)


def decomp_CH4_S2pl(t,remainAGB_CH4_S2pl):
    return (1-(1-np.exp(-k*t)))*remainAGB_CH4_S2pl



#set zero matrix
output_decomp_CH4_S2pl = np.zeros((len(t),len(df['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_S2pl in enumerate(df['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_S2pl[i:,i] = decomp_CH4_S2pl(t[:len(t)-i],remain_part_CH4_S2pl)

print(output_decomp_CH4_S2pl[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_S2pl = np.zeros((len(t)-1,len(df['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_S2pl[:,i] = np.diff(output_decomp_CH4_S2pl[:,i])
    i = i + 1 

print(subs_matrix_CH4_S2pl[:,:4])
print(len(subs_matrix_CH4_S2pl))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_S2pl = subs_matrix_CH4_S2pl.clip(max=0)

print(subs_matrix_CH4_S2pl[:,:4])

#make the results as absolute values
subs_matrix_CH4_S2pl = abs(subs_matrix_CH4_S2pl)
print(subs_matrix_CH4_S2pl[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_S2pl = np.zeros((len(t)-200,len(df['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_S2pl)

subs_matrix_CH4_S2pl = np.vstack((zero_matrix_CH4_S2pl, subs_matrix_CH4_S2pl))

print(subs_matrix_CH4_S2pl[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_S2pl = (tf,1)
decomp_tot_CH4_S2pl = np.zeros(matrix_tot_CH4_S2pl) 

i = 0
while i < tf:
    decomp_tot_CH4_S2pl[:,0] = decomp_tot_CH4_S2pl[:,0] + subs_matrix_CH4_S2pl[:,i]
    i = i + 1

print(decomp_tot_CH4_S2pl[:,0])



#Enu
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_Enu')

tf = 201

t = np.arange(tf)


def decomp_CH4_Enu(t,remainAGB_CH4_Enu):
    return (1-(1-np.exp(-k*t)))*remainAGB_CH4_Enu


#set zero matrix
output_decomp_CH4_Enu = np.zeros((len(t),len(df['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_Enu in enumerate(df['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_Enu[i:,i] = decomp_CH4_Enu(t[:len(t)-i],remain_part_CH4_Enu)

print(output_decomp_CH4_Enu[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_Enu = np.zeros((len(t)-1,len(df['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_Enu[:,i] = np.diff(output_decomp_CH4_Enu[:,i])
    i = i + 1 

print(subs_matrix_CH4_Enu[:,:4])
print(len(subs_matrix_CH4_Enu))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_Enu = subs_matrix_CH4_Enu.clip(max=0)

print(subs_matrix_CH4_Enu[:,:4])

#make the results as absolute values
subs_matrix_CH4_Enu = abs(subs_matrix_CH4_Enu)
print(subs_matrix_CH4_Enu[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_Enu = np.zeros((len(t)-200,len(df['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_Enu)

subs_matrix_CH4_Enu = np.vstack((zero_matrix_CH4_Enu, subs_matrix_CH4_Enu))

print(subs_matrix_CH4_Enu[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_Enu = (tf,1)
decomp_tot_CH4_Enu= np.zeros(matrix_tot_CH4_Enu) 

i = 0
while i < tf:
    decomp_tot_CH4_Enu[:,0] = decomp_tot_CH4_Enu[:,0] + subs_matrix_CH4_Enu[:,i]
    i = i + 1

print(decomp_tot_CH4_Enu[:,0])



#Epl
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_Epl')

tf = 201

t = np.arange(tf)


def decomp_CH4_Epl(t,remainAGB_CH4_Epl):
    return (1-(1-np.exp(-k*t)))*remainAGB_CH4_Epl


#set zero matrix
output_decomp_CH4_Epl = np.zeros((len(t),len(df['Landfill_decomp_CH4'].values)))


for i,remain_part_CH4_Epl in enumerate(df['Landfill_decomp_CH4'].values):
    #print(i,remain_part)
    output_decomp_CH4_Epl[i:,i] = decomp_CH4_Epl(t[:len(t)-i],remain_part_CH4_Epl)

print(output_decomp_CH4_Epl[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CH4_Epl = np.zeros((len(t)-1,len(df['Landfill_decomp_CH4'].values-1)))

i = 0
while i < tf:
    subs_matrix_CH4_Epl[:,i] = np.diff(output_decomp_CH4_Epl[:,i])
    i = i + 1 

print(subs_matrix_CH4_Epl[:,:4])
print(len(subs_matrix_CH4_Epl))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CH4_Epl = subs_matrix_CH4_Epl.clip(max=0)

print(subs_matrix_CH4_Epl[:,:4])

#make the results as absolute values
subs_matrix_CH4_Epl = abs(subs_matrix_CH4_Epl)
print(subs_matrix_CH4_Epl[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CH4_Epl = np.zeros((len(t)-200,len(df['Landfill_decomp_CH4'].values)))
print(zero_matrix_CH4_Epl)

subs_matrix_CH4_Epl = np.vstack((zero_matrix_CH4_Epl, subs_matrix_CH4_Epl))

print(subs_matrix_CH4_Epl[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CH4_Epl = (tf,1)
decomp_tot_CH4_Epl = np.zeros(matrix_tot_CH4_Epl) 

i = 0
while i < tf:
    decomp_tot_CH4_Epl[:,0] = decomp_tot_CH4_Epl[:,0] + subs_matrix_CH4_Epl[:,i]
    i = i + 1

print(decomp_tot_CH4_Epl[:,0])


#plotting
t = np.arange(0,tf)

plt.plot(t,decomp_tot_CH4_S2nu,label='CH4_S2nu')
plt.plot(t,decomp_tot_CH4_S2pl,label='CH4_S2pl')
plt.plot(t,decomp_tot_CH4_Enu,label='CH4_Enu')
plt.plot(t,decomp_tot_CH4_Epl,label='CH4_Epl')

plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()


#%%

#Step (7_2): landfill gas decomposition (CO2)

#CO2 decomposition 

hl = 20   #half-live

k = (np.log(2))/hl


#S2nu
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_S2nu')

tf = 201

t = np.arange(tf)


def decomp_CO2_S2nu(t,remainAGB_CO2_S2nu):
    return (1-(1-np.exp(-k*t)))*remainAGB_CO2_S2nu



#set zero matrix
output_decomp_CO2_S2nu = np.zeros((len(t),len(df['Landfill_decomp_CO2'].values)))


for i,remain_part_CO2_S2nu in enumerate(df['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_CO2_S2nu[i:,i] = decomp_CO2_S2nu(t[:len(t)-i],remain_part_CO2_S2nu)

print(output_decomp_CO2_S2nu[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CO2_S2nu = np.zeros((len(t)-1,len(df['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_CO2_S2nu[:,i] = np.diff(output_decomp_CO2_S2nu[:,i])
    i = i + 1 

print(subs_matrix_CO2_S2nu[:,:4])
print(len(subs_matrix_CO2_S2nu))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CO2_S2nu = subs_matrix_CO2_S2nu.clip(max=0)

print(subs_matrix_CO2_S2nu[:,:4])

#make the results as absolute values
subs_matrix_CO2_S2nu = abs(subs_matrix_CO2_S2nu)
print(subs_matrix_CO2_S2nu[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CO2_S2nu = np.zeros((len(t)-200,len(df['Landfill_decomp_CO2'].values)))
print(zero_matrix_CO2_S2nu)

subs_matrix_CO2_S2nu = np.vstack((zero_matrix_CO2_S2nu, subs_matrix_CO2_S2nu))

print(subs_matrix_CO2_S2nu[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CO2_S2nu = (tf,1)
decomp_tot_CO2_S2nu = np.zeros(matrix_tot_CO2_S2nu) 

i = 0
while i < tf:
    decomp_tot_CO2_S2nu[:,0] = decomp_tot_CO2_S2nu[:,0] + subs_matrix_CO2_S2nu[:,i]
    i = i + 1

print(decomp_tot_CO2_S2nu[:,0])



#S2pl
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_S2pl')

tf = 201

t = np.arange(tf)


def decomp_CO2_S2pl(t,remainAGB_CO2_S2pl):
    return (1-(1-np.exp(-k*t)))*remainAGB_CO2_S2pl



#set zero matrix
output_decomp_CO2_S2pl = np.zeros((len(t),len(df['Landfill_decomp_CO2'].values)))


for i,remain_part_CO2_S2pl in enumerate(df['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_CO2_S2pl[i:,i] = decomp_CO2_S2pl(t[:len(t)-i],remain_part_CO2_S2pl)

print(output_decomp_CO2_S2pl[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CO2_S2pl = np.zeros((len(t)-1,len(df['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_CO2_S2pl[:,i] = np.diff(output_decomp_CO2_S2pl[:,i])
    i = i + 1 

print(subs_matrix_CO2_S2pl[:,:4])
print(len(subs_matrix_CO2_S2pl))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CO2_S2pl = subs_matrix_CO2_S2pl.clip(max=0)

print(subs_matrix_CO2_S2pl[:,:4])

#make the results as absolute values
subs_matrix_CO2_S2pl = abs(subs_matrix_CO2_S2pl)
print(subs_matrix_CO2_S2pl[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CO2_S2pl = np.zeros((len(t)-200,len(df['Landfill_decomp_CO2'].values)))
print(zero_matrix_CO2_S2pl)

subs_matrix_CO2_S2pl = np.vstack((zero_matrix_CO2_S2pl, subs_matrix_CO2_S2pl))

print(subs_matrix_CO2_S2pl[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CO2_S2pl = (tf,1)
decomp_tot_CO2_S2pl = np.zeros(matrix_tot_CO2_S2pl) 

i = 0
while i < tf:
    decomp_tot_CO2_S2pl[:,0] = decomp_tot_CO2_S2pl[:,0] + subs_matrix_CO2_S2pl[:,i]
    i = i + 1

print(decomp_tot_CO2_S2pl[:,0])



#Enu
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_Enu')

tf = 201

t = np.arange(tf)


def decomp_CO2_Enu(t,remainAGB_CO2_Enu):
    return (1-(1-np.exp(-k*t)))*remainAGB_CO2_Enu


#set zero matrix
output_decomp_CO2_Enu = np.zeros((len(t),len(df['Landfill_decomp_CO2'].values)))


for i,remain_part_CO2_Enu in enumerate(df['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_CO2_Enu[i:,i] = decomp_CO2_Enu(t[:len(t)-i],remain_part_CO2_Enu)

print(output_decomp_CO2_Enu[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CO2_Enu = np.zeros((len(t)-1,len(df['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_CO2_Enu[:,i] = np.diff(output_decomp_CO2_Enu[:,i])
    i = i + 1 

print(subs_matrix_CO2_Enu[:,:4])
print(len(subs_matrix_CO2_Enu))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CO2_Enu = subs_matrix_CO2_Enu.clip(max=0)

print(subs_matrix_CO2_Enu[:,:4])

#make the results as absolute values
subs_matrix_CO2_Enu = abs(subs_matrix_CO2_Enu)
print(subs_matrix_CO2_Enu[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CO2_Enu = np.zeros((len(t)-200,len(df['Landfill_decomp_CO2'].values)))
print(zero_matrix_CO2_Enu)

subs_matrix_CO2_Enu = np.vstack((zero_matrix_CO2_Enu, subs_matrix_CO2_Enu))

print(subs_matrix_CO2_Enu[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CO2_Enu = (tf,1)
decomp_tot_CO2_Enu= np.zeros(matrix_tot_CO2_Enu) 

i = 0
while i < tf:
    decomp_tot_CO2_Enu[:,0] = decomp_tot_CO2_Enu[:,0] + subs_matrix_CO2_Enu[:,i]
    i = i + 1

print(decomp_tot_CO2_Enu[:,0])



#Epl
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO_Ret.xlsx', 'PF_PO_Epl')

tf = 201

t = np.arange(tf)


def decomp_CO2_Epl(t,remainAGB_CO2_Epl):
    return (1-(1-np.exp(-k*t)))*remainAGB_CO2_Epl


#set zero matrix
output_decomp_CO2_Epl = np.zeros((len(t),len(df['Landfill_decomp_CO2'].values)))


for i,remain_part_CO2_Epl in enumerate(df['Landfill_decomp_CO2'].values):
    #print(i,remain_part)
    output_decomp_CO2_Epl[i:,i] = decomp_CO2_Epl(t[:len(t)-i],remain_part_CO2_Epl)

print(output_decomp_CO2_Epl[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_CO2_Epl = np.zeros((len(t)-1,len(df['Landfill_decomp_CO2'].values-1)))

i = 0
while i < tf:
    subs_matrix_CO2_Epl[:,i] = np.diff(output_decomp_CO2_Epl[:,i])
    i = i + 1 

print(subs_matrix_CO2_Epl[:,:4])
print(len(subs_matrix_CO2_Epl))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_CO2_Epl = subs_matrix_CO2_Epl.clip(max=0)

print(subs_matrix_CO2_Epl[:,:4])

#make the results as absolute values
subs_matrix_CO2_Epl = abs(subs_matrix_CO2_Epl)
print(subs_matrix_CO2_Epl[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_CO2_Epl = np.zeros((len(t)-200,len(df['Landfill_decomp_CO2'].values)))
print(zero_matrix_CO2_Epl)

subs_matrix_CO2_Epl = np.vstack((zero_matrix_CO2_Epl, subs_matrix_CO2_Epl))

print(subs_matrix_CO2_Epl[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_CO2_Epl = (tf,1)
decomp_tot_CO2_Epl = np.zeros(matrix_tot_CO2_Epl) 

i = 0
while i < tf:
    decomp_tot_CO2_Epl[:,0] = decomp_tot_CO2_Epl[:,0] + subs_matrix_CO2_Epl[:,i]
    i = i + 1

print(decomp_tot_CO2_Epl[:,0])



#plotting
t = np.arange(0,tf)

plt.plot(t,decomp_tot_CO2_S2nu,label='CO2_S2nu')
plt.plot(t,decomp_tot_CO2_S2pl,label='CO2_S2pl')
plt.plot(t,decomp_tot_CO2_Enu,label='CO2_Enu')
plt.plot(t,decomp_tot_CO2_Epl,label='CO2_Epl')

plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()




#%%

#Step (8): Sum the emissions and sequestration (net carbon balance), CO2 and CH4 are separated


#https://stackoverflow.com/questions/52703442/python-sum-values-from-multiple-lists-more-than-two
#C_loss + C_remainAGB + C_remainHWP + PH_Emissions_PO


TestDSM2nu.o = [x * 0.5 for x in TestDSM2nu.o]
TestDSM2pl.o = [x * 0.5 for x in TestDSM2pl.o]
TestDSM3nu.o = [x * 0.5 for x in TestDSM3nu.o]
TestDSM3pl.o = [x * 0.5 for x in TestDSM3pl.o]



Emissions_PF_PO_S2nu = [c_firewood_energy_S2nu, decomp_emissions[:,0], TestDSM2nu.o, PH_Emissions_PO_S2nu, PH_Emissions_HWP_S2nu, decomp_tot_CO2_S2nu[:,0]]
Emissions_PF_PO_S2pl = [c_firewood_energy_S2pl, decomp_emissions[:,0], TestDSM2pl.o, PH_Emissions_PO_S2pl, PH_Emissions_HWP_S2pl, decomp_tot_CO2_S2pl[:,0]]
Emissions_PF_PO_Enu = [c_firewood_energy_Enu, c_pellets_Enu, decomp_emissions[:,0], TestDSM3nu.o, PH_Emissions_PO_Enu, PH_Emissions_HWP_Enu, decomp_tot_CO2_Enu[:,0]]
Emissions_PF_PO_Epl = [c_firewood_energy_Epl, c_pellets_Epl, decomp_emissions[:,0], TestDSM3pl.o, PH_Emissions_PO_Epl, PH_Emissions_HWP_Epl, decomp_tot_CO2_Epl[:,0]]


Emissions_PF_PO_S2nu = [sum(x) for x in zip(*Emissions_PF_PO_S2nu)]
Emissions_PF_PO_S2pl = [sum(x) for x in zip(*Emissions_PF_PO_S2pl)]
Emissions_PF_PO_Enu = [sum(x) for x in zip(*Emissions_PF_PO_Enu)]
Emissions_PF_PO_Epl = [sum(x) for x in zip(*Emissions_PF_PO_Epl)]


#CH4_S2nu
Emissions_CH4_PF_PO_S2nu = decomp_tot_CH4_S2nu[:,0]

#CH4_S2pl
Emissions_CH4_PF_PO_S2pl = decomp_tot_CH4_S2pl[:,0]

#CH4_Enu
Emissions_CH4_PF_PO_Enu = decomp_tot_CH4_Enu[:,0]      

#CH4_Epl
Emissions_CH4_PF_PO_Epl = decomp_tot_CH4_Epl[:,0] 




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

Col2_S2nu = Emissions_PF_PO_S2nu
Col2_S2pl = Emissions_PF_PO_S2pl
Col2_Enu = Emissions_PF_PO_Enu
Col2_Epl = Emissions_PF_PO_Epl


Col3_S2nu = Emissions_CH4_PF_PO_S2nu
Col3_S2pl = Emissions_CH4_PF_PO_S2pl
Col3_Enu = Emissions_CH4_PF_PO_Enu
Col3_Epl = Emissions_CH4_PF_PO_Epl


Col4 = flat_list_nucleus
Col5 = Emission_ref
Col6 = flat_list_plasma


#S2
df2_nu = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_S2nu,'kg_CH4':Col3_S2nu,'kg_CO2_seq':Col4,'emission_ref':Col5})
df2_pl = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_S2pl,'kg_CH4':Col3_S2pl,'kg_CO2_seq':Col6,'emission_ref':Col5})

#E
df3_nu = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_Enu,'kg_CH4':Col3_Enu,'kg_CO2_seq':Col4,'emission_ref':Col5})
df3_pl = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_Epl,'kg_CH4':Col3_Epl,'kg_CO2_seq':Col6,'emission_ref':Col5})

writer = pd.ExcelWriter('emissions_seq_PF_PO_Ret.xlsx', engine = 'xlsxwriter')


df2_nu.to_excel(writer, sheet_name = 'S2_nucleus', header=True, index=False)
df2_pl.to_excel(writer, sheet_name = 'S2_plasma', header=True, index=False)
df3_nu.to_excel(writer, sheet_name = 'E_nucleus', header=True, index=False)
df3_pl.to_excel(writer, sheet_name = 'E_plasma', header=True, index=False)


writer.save()
writer.close()


#%%

## DYNAMIC LCA -  wood-based scenarios

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


##wood-based
#read S2_nucleus
df = pd.read_excel('emissions_seq_PF_PO_Ret.xlsx', 'S2_nucleus') # can also index sheet by name or fetch all sheets
emission_CO2_S2nu = df['kg_CO2'].tolist()
emission_CH4_S2nu = df['kg_CH4'].tolist()
emission_CO2_seq_S2nu = df['kg_CO2_seq'].tolist()

emission_CO2_ref = df['emission_ref'].tolist() 

#read S2_plasma
df = pd.read_excel('emissions_seq_PF_PO_Ret.xlsx', 'S2_plasma')
emission_CO2_S2pl = df['kg_CO2'].tolist()
emission_CH4_S2pl = df['kg_CH4'].tolist()
emission_CO2_seq_S2pl = df['kg_CO2_seq'].tolist()


#read E_nucleus
df = pd.read_excel('emissions_seq_PF_PO_Ret.xlsx', 'E_nucleus') # can also index sheet by name or fetch all sheets
emission_CO2_Enu = df['kg_CO2'].tolist()
emission_CH4_Enu = df['kg_CH4'].tolist()
emission_CO2_seq_Enu = df['kg_CO2_seq'].tolist()


#read E_plasma
df = pd.read_excel('emissions_seq_PF_PO_Ret.xlsx', 'E_plasma')
emission_CO2_Epl = df['kg_CO2'].tolist()
emission_CH4_Epl = df['kg_CH4'].tolist()
emission_CO2_seq_Epl = df['kg_CO2_seq'].tolist()


#%%

#Step (14): import emission data from the counter-use of non-renewable materials/energy scenarios (NR)


#read S2_nucleus
df = pd.read_excel('NonRW_PF_PO.xlsx', 'PF_PO_S2nu') # can also index sheet by name or fetch all sheets
emission_NonRW_S2nu = df['NonRW_emissions'].tolist()
emission_Diesel_S2nu = df['Diesel_emissions'].tolist()
emission_NonRW_seq_S2nu = df['kg_CO2_seq'].tolist()

emission_CO2_ref = df['emission_ref'].tolist() 

#read S2_plasma
df = pd.read_excel('NonRW_PF_PO.xlsx', 'PF_PO_S2pl')
emission_NonRW_S2pl = df['NonRW_emissions'].tolist()
emission_Diesel_S2pl = df['Diesel_emissions'].tolist()
emission_NonRW_seq_S2pl = df['kg_CO2_seq'].tolist()


#read E_nucleus
df = pd.read_excel('NonRW_PF_PO.xlsx', 'PF_PO_Enu') # can also index sheet by name or fetch all sheets
emission_NonRW_Enu = df['NonRW_emissions'].tolist()
emission_Diesel_Enu = df['Diesel_emissions'].tolist()
emission_NonRW_seq_Enu = df['kg_CO2_seq'].tolist()


#read E_plasma
df = pd.read_excel('NonRW_PF_PO.xlsx', 'PF_PO_Epl')
emission_NonRW_Epl = df['NonRW_emissions'].tolist()
emission_Diesel_Epl = df['Diesel_emissions'].tolist()
emission_NonRW_seq_Epl = df['kg_CO2_seq'].tolist()


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
#S2_nucleus
t = np.arange(0,tf-1,1)

matrix_GWI_S2nu = (tf-1,3)
GWI_inst_S2nu = np.zeros(matrix_GWI_S2nu)



for t in range(0,tf-1):
    GWI_inst_S2nu[t,0] = np.sum(np.multiply(emission_CO2_S2nu,DCF_CO2_ti[:,t]))
    GWI_inst_S2nu[t,1] = np.sum(np.multiply(emission_CH4_S2nu,DCF_CH4_ti[:,t]))
    GWI_inst_S2nu[t,2] = np.sum(np.multiply(emission_CO2_seq_S2nu,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2nu = (tf-1,1)
GWI_inst_tot_S2nu = np.zeros(matrix_GWI_tot_S2nu)

GWI_inst_tot_S2nu[:,0] = np.array(GWI_inst_S2nu[:,0] + GWI_inst_S2nu[:,1] + GWI_inst_S2nu[:,2])
  
print(GWI_inst_tot_S2nu[:,0])

#S2_plasma
t = np.arange(0,tf-1,1)

matrix_GWI_S2pl = (tf-1,3)
GWI_inst_S2pl = np.zeros(matrix_GWI_S2pl)



for t in range(0,tf-1):
    GWI_inst_S2pl[t,0] = np.sum(np.multiply(emission_CO2_S2pl,DCF_CO2_ti[:,t]))
    GWI_inst_S2pl[t,1] = np.sum(np.multiply(emission_CH4_S2pl,DCF_CH4_ti[:,t]))
    GWI_inst_S2pl[t,2] = np.sum(np.multiply(emission_CO2_seq_S2pl,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S2pl = (tf-1,1)
GWI_inst_tot_S2pl = np.zeros(matrix_GWI_tot_S2pl)

GWI_inst_tot_S2pl[:,0] = np.array(GWI_inst_S2pl[:,0] + GWI_inst_S2pl[:,1] + GWI_inst_S2pl[:,2])
  
print(GWI_inst_tot_S2pl[:,0])


#E_nucleus
t = np.arange(0,tf-1,1)

matrix_GWI_Enu = (tf-1,3)
GWI_inst_Enu = np.zeros(matrix_GWI_Enu)



for t in range(0,tf-1):
    GWI_inst_Enu[t,0] = np.sum(np.multiply(emission_CO2_Enu,DCF_CO2_ti[:,t]))
    GWI_inst_Enu[t,1] = np.sum(np.multiply(emission_CH4_Enu,DCF_CH4_ti[:,t]))
    GWI_inst_Enu[t,2] = np.sum(np.multiply(emission_CO2_seq_Enu,DCF_CO2_ti[:,t]))

matrix_GWI_tot_Enu = (tf-1,1)
GWI_inst_tot_Enu = np.zeros(matrix_GWI_tot_Enu)

GWI_inst_tot_Enu[:,0] = np.array(GWI_inst_Enu[:,0] + GWI_inst_Enu[:,1] + GWI_inst_Enu[:,2])
  
print(GWI_inst_tot_Enu[:,0])


#E_plasma
t = np.arange(0,tf-1,1)

matrix_GWI_Epl = (tf-1,3)
GWI_inst_Epl = np.zeros(matrix_GWI_Epl)



for t in range(0,tf-1):
    GWI_inst_Epl[t,0] = np.sum(np.multiply(emission_CO2_Epl,DCF_CO2_ti[:,t]))
    GWI_inst_Epl[t,1] = np.sum(np.multiply(emission_CH4_Epl,DCF_CH4_ti[:,t]))
    GWI_inst_Epl[t,2] = np.sum(np.multiply(emission_CO2_seq_Epl,DCF_CO2_ti[:,t]))

matrix_GWI_tot_Epl = (tf-1,1)
GWI_inst_tot_Epl = np.zeros(matrix_GWI_tot_Epl)

GWI_inst_tot_Epl[:,0] = np.array(GWI_inst_Epl[:,0] + GWI_inst_Epl[:,1] + GWI_inst_Epl[:,2])
  
print(GWI_inst_tot_Epl[:,0])


## NonRW
#GWI_inst for all gases

#S2_nucleus
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_S2nu = (tf-1,3)
GWI_inst_NonRW_S2nu = np.zeros(matrix_GWI_NonRW_S2nu)



for t in range(0,tf-1):
    GWI_inst_NonRW_S2nu[t,0] = np.sum(np.multiply(emission_NonRW_S2nu,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_S2nu[t,1] = np.sum(np.multiply(emission_Diesel_S2nu,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_S2nu[t,2] = np.sum(np.multiply(emission_NonRW_seq_S2nu,DCF_CO2_ti[:,t]))

matrix_GWI_tot_NonRW_S2nu = (tf-1,1)
GWI_inst_tot_NonRW_S2nu = np.zeros(matrix_GWI_tot_NonRW_S2nu)

GWI_inst_tot_NonRW_S2nu[:,0] = np.array(GWI_inst_NonRW_S2nu[:,0] + GWI_inst_NonRW_S2nu[:,1] + GWI_inst_NonRW_S2nu[:,2])
  
print(GWI_inst_tot_NonRW_S2nu[:,0])



#S2_plasma
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_S2pl = (tf-1,3)
GWI_inst_NonRW_S2pl = np.zeros(matrix_GWI_NonRW_S2pl)



for t in range(0,tf-1):
    GWI_inst_NonRW_S2pl[t,0] = np.sum(np.multiply(emission_NonRW_S2pl,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_S2pl[t,1] = np.sum(np.multiply(emission_Diesel_S2pl,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_S2pl[t,2] = np.sum(np.multiply(emission_NonRW_seq_S2pl,DCF_CO2_ti[:,t]))


matrix_GWI_tot_NonRW_S2pl = (tf-1,1)
GWI_inst_tot_NonRW_S2pl = np.zeros(matrix_GWI_tot_NonRW_S2pl)

GWI_inst_tot_NonRW_S2pl[:,0] = np.array(GWI_inst_NonRW_S2pl[:,0] + GWI_inst_NonRW_S2pl[:,1] + GWI_inst_NonRW_S2pl[:,2])
  
print(GWI_inst_tot_NonRW_S2pl[:,0])


#E_nucleus
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_Enu = (tf-1,3)
GWI_inst_NonRW_Enu = np.zeros(matrix_GWI_NonRW_Enu)



for t in range(0,tf-1):
    GWI_inst_NonRW_Enu[t,0] = np.sum(np.multiply(emission_NonRW_Enu,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_Enu[t,1] = np.sum(np.multiply(emission_Diesel_Enu,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_Enu[t,2] = np.sum(np.multiply(emission_NonRW_seq_Enu,DCF_CO2_ti[:,t]))

matrix_GWI_tot_NonRW_Enu = (tf-1,1)
GWI_inst_tot_NonRW_Enu = np.zeros(matrix_GWI_tot_NonRW_Enu)

GWI_inst_tot_NonRW_Enu[:,0] = np.array(GWI_inst_NonRW_Enu[:,0] + GWI_inst_NonRW_Enu[:,1] + GWI_inst_NonRW_Enu[:,2])
  
print(GWI_inst_tot_NonRW_Enu[:,0])


#E_plasma
t = np.arange(0,tf-1,1)

matrix_GWI_NonRW_Epl = (tf-1,3)
GWI_inst_NonRW_Epl = np.zeros(matrix_GWI_NonRW_Epl)



for t in range(0,tf-1):
    GWI_inst_NonRW_Epl[t,0] = np.sum(np.multiply(emission_NonRW_Epl,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_Epl[t,1] = np.sum(np.multiply(emission_Diesel_Epl,DCF_CO2_ti[:,t]))
    GWI_inst_NonRW_Epl[t,2] = np.sum(np.multiply(emission_NonRW_seq_Epl,DCF_CO2_ti[:,t]))

matrix_GWI_tot_NonRW_Epl = (tf-1,1)
GWI_inst_tot_NonRW_Epl = np.zeros(matrix_GWI_tot_NonRW_Epl)

GWI_inst_tot_NonRW_Epl[:,0] = np.array(GWI_inst_NonRW_Epl[:,0] + GWI_inst_NonRW_Epl[:,1] + GWI_inst_NonRW_Epl[:,2])
  
print(GWI_inst_tot_NonRW_Epl[:,0])



t = np.arange(0,tf-1,1)

#create zero list to highlight the horizontal line for 0
def zerolistmaker(n):
    listofzeros = [0] * (n)
    return listofzeros

#convert to flat list
GWI_inst_tot_NonRW_S2nu = np.array([item for sublist in GWI_inst_tot_NonRW_S2nu for item in sublist])
GWI_inst_tot_NonRW_S2pl = np.array([item for sublist in GWI_inst_tot_NonRW_S2pl for item in sublist])
GWI_inst_tot_NonRW_Enu = np.array([item for sublist in GWI_inst_tot_NonRW_Enu for item in sublist])
GWI_inst_tot_NonRW_Epl = np.array([item for sublist in GWI_inst_tot_NonRW_Epl for item in sublist])



GWI_inst_tot_S2nu = np.array([item for sublist in GWI_inst_tot_S2nu for item in sublist])
GWI_inst_tot_S2pl = np.array([item for sublist in GWI_inst_tot_S2pl for item in sublist])
GWI_inst_tot_Enu = np.array([item for sublist in GWI_inst_tot_Enu for item in sublist])
GWI_inst_tot_Epl = np.array([item for sublist in GWI_inst_tot_Epl for item in sublist])


plt.plot(t, GWI_inst_tot_NonRW_S2nu, color='lightcoral', label='NR_M_nucleus', ls='--')
plt.plot(t, GWI_inst_tot_NonRW_S2pl, color='deeppink', label='NR_M_plasma', ls='--')
plt.plot(t, GWI_inst_tot_NonRW_Enu, color='royalblue', label='NR_E_nucleus', ls='--')
plt.plot(t, GWI_inst_tot_NonRW_Epl, color='deepskyblue', label='NR_E_plasma', ls='--')

plt.plot(t, GWI_inst_tot_S2nu, color='lightcoral', label='M_nucleus')
plt.plot(t, GWI_inst_tot_S2pl, color='deeppink', label='M_plasma')
plt.plot(t, GWI_inst_tot_Enu, color='royalblue', label='E_nucleus')
plt.plot(t, GWI_inst_tot_Epl, color='deepskyblue', label='E_plasma')

plt.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWI_inst_tot_NonRW_Enu, GWI_inst_tot_NonRW_S2pl, color='lightcoral', alpha=0.3)

plt.grid(True)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.xlim(0,200)
#plt.ylim(-0.5e-9,1.4e-9)


plt.title('Instantaneous GWI, PF_PO')

plt.xlabel('Time (year)')
#plt.ylabel('GWI_inst (10$^{-13}$ W/m$^2$)')
#plt.ylabel('GWI_inst (W/m$^2$)')


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_inst_NonRW_PF_PO', dpi=300)

plt.show()




#%%

#Step (17): Calculate cumulative global warming impact (GWI)

##wood-based
GWI_cum_S2nu = np.cumsum(GWI_inst_tot_S2nu)
GWI_cum_S2pl = np.cumsum(GWI_inst_tot_S2pl)
GWI_cum_Enu = np.cumsum(GWI_inst_tot_Enu)
GWI_cum_Epl = np.cumsum(GWI_inst_tot_Epl)


##NonRW
GWI_cum_NonRW_S2nu = np.cumsum(GWI_inst_tot_NonRW_S2nu)
GWI_cum_NonRW_S2pl = np.cumsum(GWI_inst_tot_NonRW_S2pl)
GWI_cum_NonRW_Enu = np.cumsum(GWI_inst_tot_NonRW_Enu)
GWI_cum_NonRW_Epl = np.cumsum(GWI_inst_tot_NonRW_Epl)



plt.xlabel('Time (year)')
#plt.ylabel('GWI_cum (10$^{-11}$ W/m$^2$)')
plt.ylabel('GWI_cum (W/m$^2$)')


plt.xlim(0,200)
#plt.ylim(-0.3e-7,2e-7)


plt.title('Cumulative GWI, PF_PO')


plt.plot(t, GWI_cum_NonRW_S2nu, color='lightcoral', label='NR_M_nucleus', ls='--')
plt.plot(t, GWI_cum_NonRW_S2pl, color='deeppink', label='NR_M_plasma', ls='--')
plt.plot(t, GWI_cum_NonRW_Enu, color='royalblue', label='NR_E_nucleus', ls='--')
plt.plot(t, GWI_cum_NonRW_Epl, color='deepskyblue', label='NR_E_plasma', ls='--')


plt.plot(t, GWI_cum_S2nu, color='lightcoral', label='M_nucleus')
plt.plot(t, GWI_cum_S2pl, color='deeppink', label='M_plasma')
plt.plot(t, GWI_cum_Enu, color='royalblue', label='E_nucleus')
plt.plot(t, GWI_cum_Epl, color='deepskyblue', label='E_plasma')

plt.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWI_cum_NonRW_Enu, GWI_cum_NonRW_S2pl, color='lightcoral', alpha=0.3)

plt.grid(True)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_cum_NonRW_PF_PO', dpi=300)

plt.show()



#%%

#Step (18): Determine the Instantenous and Cumulative GWI for the  emission reference (1 kg CO2 emission at time zero) before performing dynamic GWP calculation


#determine the GWI inst for the emission reference (1 kg CO2 emission at time zero)

t = np.arange(0,tf-1,1)

matrix_GWI_ref = (tf-1,1)
GWI_inst_ref = np.zeros(matrix_GWI_S2nu)

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

#convert the GWPdyn to tCO2 (divided by 1000)

##wood-based
GWP_dyn_cum_S2nu = [x/(y*1000) for x,y in zip(GWI_cum_S2nu, GWI_cum_ref)]
GWP_dyn_cum_S2pl = [x/(y*1000) for x,y in zip(GWI_cum_S2pl, GWI_cum_ref)]
GWP_dyn_cum_Enu = [x/(y*1000) for x,y in zip(GWI_cum_Enu, GWI_cum_ref)]
GWP_dyn_cum_Epl = [x/(y*1000) for x,y in zip(GWI_cum_Epl, GWI_cum_ref)]


##NonRW
GWP_dyn_cum_NonRW_S2nu = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_S2nu, GWI_cum_ref)]
GWP_dyn_cum_NonRW_S2pl = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_S2pl, GWI_cum_ref)]
GWP_dyn_cum_NonRW_Enu = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_Enu, GWI_cum_ref)]
GWP_dyn_cum_NonRW_Epl = [x/(y*1000) for x,y in zip(GWI_cum_NonRW_Epl, GWI_cum_ref)]




fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)

ax.plot(t, GWP_dyn_cum_NonRW_S2nu, color='lightcoral', label='NR_M_nucleus', ls='--')
ax.plot(t, GWP_dyn_cum_NonRW_S2pl, color='deeppink', label='NR_M_plasma', ls='--')
ax.plot(t, GWP_dyn_cum_NonRW_Enu, color='royalblue', label='NR_E_nucleus', ls='--')
ax.plot(t, GWP_dyn_cum_NonRW_Epl, color='deepskyblue', label='NR_E_plasma', ls='--')


ax.plot(t, GWP_dyn_cum_S2nu, color='lightcoral', label='M_nucleus')
ax.plot(t, GWP_dyn_cum_S2pl, color='deeppink', label='M_plasma')
ax.plot(t, GWP_dyn_cum_Enu, color='royalblue', label='E_nucleus')
ax.plot(t, GWP_dyn_cum_Epl, color='deepskyblue', label='E_plasma')

ax.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

#plt.fill_between(t, GWP_dyn_cum_NonRW_Enu, GWP_dyn_cum_NonRW_S2pl, color='lightcoral', alpha=0.3)

plt.grid(True)

ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)


ax.set_xlabel('Time (year)')
ax.set_ylabel('GWP$_{dyn}$ (t-CO$_2$-eq)')


ax.set_xlim(0,200)
#ax.set_ylim(-250,1400)


ax.set_title('Dynamic GWP, PF_PO')


plt.draw()

plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_cum_NonRW_PF_PO', dpi=300)


#%%

#Step (20): Exporting the data behind result graphs to Excel

year = []
for x in range (0, 201): 
    year.append(x) 


### Create Column
    
Col1 = year

##GWI_Inst
#GWI_inst from wood-based scenarios
Col_GI_3  = GWI_inst_tot_S2nu
Col_GI_4  = GWI_inst_tot_S2pl
Col_GI_5  = GWI_inst_tot_Enu
Col_GI_6  = GWI_inst_tot_Epl


#print(Col_GI_1)
#print(np.shape(Col_GI_1))

#GWI_inst from counter use scenarios
Col_GI_9  = GWI_inst_tot_NonRW_S2nu
Col_GI_10  = GWI_inst_tot_NonRW_S2pl
Col_GI_11  = GWI_inst_tot_NonRW_Enu
Col_GI_12  = GWI_inst_tot_NonRW_Epl


#print(Col_GI_7)
#print(np.shape(Col_GI_7))


#create column results
    
##GWI_cumulative
#GWI_cumulative from wood-based scenarios
Col_GC_3 = GWI_cum_S2nu
Col_GC_4 = GWI_cum_S2pl
Col_GC_5 = GWI_cum_Enu
Col_GC_6 = GWI_cum_Epl

#GWI_cumulative from counter use scenarios
Col_GC_9 = GWI_cum_NonRW_S2nu
Col_GC_10 = GWI_cum_NonRW_S2pl
Col_GC_11 = GWI_cum_NonRW_Enu
Col_GC_12 = GWI_cum_NonRW_Epl

#create column results

##GWPdyn
#GWPdyn from wood-based scenarios
Col_GWP_3 = GWP_dyn_cum_S2nu
Col_GWP_4 = GWP_dyn_cum_S2pl
Col_GWP_5 = GWP_dyn_cum_Enu
Col_GWP_6 = GWP_dyn_cum_Epl

#GWPdyn from counter use scenarios
Col_GWP_9 = GWP_dyn_cum_NonRW_S2nu
Col_GWP_10 = GWP_dyn_cum_NonRW_S2pl
Col_GWP_11 = GWP_dyn_cum_NonRW_Enu
Col_GWP_12 = GWP_dyn_cum_NonRW_Epl


#Create colum results
dfM_GI = pd.DataFrame.from_dict({'Year':Col1,'M_nucleus (W/m2)':Col_GI_3, 'M_plasma (W/m2)':Col_GI_4,
                                         'E_nucleus (W/m2)':Col_GI_5, 'E_plasma (W/m2)':Col_GI_6,
                                         'NR_M_nucleus (W/m2)':Col_GI_9, 'NR_M_plasma (W/m2)':Col_GI_10,
                                         'NR_E_nucleus (W/m2)':Col_GI_11, 'NR_E_plasma (W/m2)':Col_GI_12})
    
dfM_GC = pd.DataFrame.from_dict({'Year':Col1,'M_nucleus (W/m2)':Col_GC_3, 'M_plasma (W/m2)':Col_GC_4,
                                         'E_nucleus (W/m2)':Col_GC_5, 'E_plasma (W/m2)':Col_GC_6, 
                                         'NR_M_nucleus (W/m2)':Col_GC_9, 'NR_M_plasma (W/m2)':Col_GC_10,
                                         'NR_E_nucleus (W/m2)':Col_GC_11, 'NR_E_plasma (W/m2)':Col_GC_12})

dfM_GWPdyn = pd.DataFrame.from_dict({'Year':Col1,'M_nucleus (t-CO2eq)':Col_GWP_3, 'M_plasma (t-CO2eq)':Col_GWP_4,
                                          'E_nucleus (t-CO2eq)':Col_GWP_5, 'E_plasma (t-CO2eq)':Col_GWP_6, 
                                          'NR_M_nucleus (t-CO2eq)':Col_GWP_9, 'NR_M_plasma (t-CO2eq)':Col_GWP_10,
                                          'NR_E_nucleus (t-CO2eq)':Col_GWP_11, 'NR_E_plasma (t-CO2eq)':Col_GWP_12})

    
#Export to excel
writer = pd.ExcelWriter('GraphResults_PF_PO_Ret.xlsx', engine = 'xlsxwriter')

#GWI_inst
dfM_GI.to_excel(writer, sheet_name = 'GWI_Inst_PF_PO', header=True, index=False)

#GWI cumulative
dfM_GC.to_excel(writer, sheet_name = 'Cumulative GWI_PF_PO', header=True, index=False)

#GWP_dyn
dfM_GWPdyn.to_excel(writer, sheet_name = 'GWPdyn_PF_PO', header=True, index=False)



writer.save()
writer.close()


#%%

#Step (21): Generate the excel file for the individual carbon emission and sequestration flows

#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)


print(len(year))


division = 1000*44/12
division_CH4 = 1000*16/12



#M_nu
c_firewood_energy_S2nu = [x/division for x in c_firewood_energy_S2nu]
decomp_emissions[:,0] = [x/division for x in decomp_emissions[:,0]]
TestDSM2nu.o = [x/division for x in TestDSM2nu.o]
PH_Emissions_PO_S2nu = [x/division for x in PH_Emissions_PO_S2nu]
PH_Emissions_HWP_S2nu = [x/division for x in PH_Emissions_HWP_S2nu]
#OC_storage_S2nu = [x/division for x in OC_storage_S2nu]
flat_list_nucleus = [x/division for x in flat_list_nucleus]
decomp_tot_CO2_S2nu[:,0] = [x/division for x in decomp_tot_CO2_S2nu[:,0]]

decomp_tot_CH4_S2nu[:,0] = [x/division_CH4 for x in decomp_tot_CH4_S2nu[:,0]]



#M_pl
c_firewood_energy_S2pl = [x/division for x in c_firewood_energy_S2pl]
TestDSM2pl.o = [x/division for x in TestDSM2pl.o]
PH_Emissions_PO_S2pl = [x/division for x in PH_Emissions_PO_S2pl]
PH_Emissions_HWP_S2pl = [x/division for x in PH_Emissions_HWP_S2pl]
#OC_storage_S2pl = [x/division for x in OC_storage_S2pl]
flat_list_plasma = [x/division for x in flat_list_plasma]
decomp_tot_CO2_S2pl[:,0] = [x/division for x in decomp_tot_CO2_S2pl[:,0]]

decomp_tot_CH4_S2pl[:,0] = [x/division_CH4 for x in decomp_tot_CH4_S2pl[:,0]]


#Enu
c_firewood_energy_Enu = [x/division for x in c_firewood_energy_Enu]
c_pellets_Enu = [x/division for x in c_pellets_Enu]
TestDSM3nu.o = [x/division for x in TestDSM3nu.o]
PH_Emissions_PO_Enu = [x/division for x in PH_Emissions_PO_Enu]
PH_Emissions_HWP_Enu = [x/division for x in PH_Emissions_HWP_Enu]
#OC_storage_Enu = [x/division for x in OC_storage_Enu]
decomp_tot_CO2_Enu[:,0] = [x/division for x in decomp_tot_CO2_Enu[:,0]]

decomp_tot_CH4_Enu[:,0] = [x/division_CH4 for x in decomp_tot_CH4_Enu[:,0]]


#Epl
c_firewood_energy_Epl = [x/division for x in c_firewood_energy_Epl]
c_pellets_Epl = [x/division for x in c_pellets_Epl]
TestDSM3pl.o = [x/division for x in TestDSM3pl.o]
PH_Emissions_PO_Epl = [x/division for x in PH_Emissions_PO_Epl]
PH_Emissions_HWP_Epl = [x/division for x in PH_Emissions_HWP_Epl]
#OC_storage_Epl = [x/division for x in OC_storage_Epl]
decomp_tot_CO2_Epl[:,0] = [x/division for x in decomp_tot_CO2_Epl[:,0]]

decomp_tot_CH4_Epl[:,0] = [x/division_CH4 for x in decomp_tot_CH4_Epl[:,0]]


#landfill aggregate flows
Landfill_decomp_PF_PO_S2nu = decomp_tot_CH4_S2nu, decomp_tot_CO2_S2nu
Landfill_decomp_PF_PO_S2pl = decomp_tot_CH4_S2pl, decomp_tot_CO2_S2pl
Landfill_decomp_PF_PO_Enu = decomp_tot_CH4_Enu, decomp_tot_CO2_Enu
Landfill_decomp_PF_PO_Epl = decomp_tot_CH4_Epl, decomp_tot_CO2_Epl


Landfill_decomp_PF_PO_S2nu = [sum(x) for x in zip(*Landfill_decomp_PF_PO_S2nu)]
Landfill_decomp_PF_PO_S2pl = [sum(x) for x in zip(*Landfill_decomp_PF_PO_S2pl)]
Landfill_decomp_PF_PO_Enu = [sum(x) for x in zip(*Landfill_decomp_PF_PO_Enu)]
Landfill_decomp_PF_PO_Epl = [sum(x) for x in zip(*Landfill_decomp_PF_PO_Epl)]

Landfill_decomp_PF_PO_S2nu = [item for sublist in Landfill_decomp_PF_PO_S2nu for item in sublist]
Landfill_decomp_PF_PO_S2pl = [item for sublist in Landfill_decomp_PF_PO_S2pl for item in sublist]
Landfill_decomp_PF_PO_Enu = [item for sublist in Landfill_decomp_PF_PO_Enu for item in sublist]
Landfill_decomp_PF_PO_Epl = [item for sublist in Landfill_decomp_PF_PO_Epl for item in sublist]


#Wood processing aggregate flows
OpProcessing_PF_PO_S2nu = [x + y for x, y in zip(PH_Emissions_PO_S2nu, PH_Emissions_HWP_S2nu)] 
OpProcessing_PF_PO_S2pl = [x + y for x, y in zip(PH_Emissions_PO_S2pl, PH_Emissions_HWP_S2pl)] 
OpProcessing_PF_PO_Enu = [x + y for x, y in zip(PH_Emissions_PO_Enu, PH_Emissions_HWP_Enu)]
OpProcessing_PF_PO_Epl = [x + y for x, y in zip(PH_Emissions_PO_Epl, PH_Emissions_HWP_Epl)]


#M_nu
Column1 = year
Column2 = c_firewood_energy_S2nu
Column3 = decomp_emissions[:,0]
Column4 = TestDSM2nu.o
Column5 = OpProcessing_PF_PO_S2nu
#Column7_1 = OC_storage_S2nu
Column7 = Landfill_decomp_PF_PO_S2nu
Column8 = flat_list_nucleus

#M_pl
Column9 = c_firewood_energy_S2pl
Column10 = TestDSM2pl.o
Column11 = OpProcessing_PF_PO_S2pl
#Column13_1 = OC_storage_S2pl
Column13 = Landfill_decomp_PF_PO_S2pl
Column14 = flat_list_plasma

#E_nu
Column15 = c_firewood_energy_Enu
Column15_1 = c_pellets_Enu
Column16 = TestDSM3nu.o
Column17 = OpProcessing_PF_PO_Enu
#Column19_1 = OC_storage_Enu
Column19 = Landfill_decomp_PF_PO_Enu


#E_pl
Column20 = c_firewood_energy_Epl
Column20_1 = c_pellets_Epl
Column21 = TestDSM3pl.o
Column22 = OpProcessing_PF_PO_Epl
#Column24_1 = OC_storage_Epl
Column24 = Landfill_decomp_PF_PO_Epl



#M
dfM_nu = pd.DataFrame.from_dict({'Year':Column1,'F0-1: Biomass C sequestration (t-C)':Column8,
                                 #'9: Landfill storage (t-C)':Column7_1, 
                                 'F1-0: Residue decomposition (t-C)':Column3,
                                 'F6-0-1: Emissions from firewood/other energy use (t-C)':Column2,
                                 'F8-0: Operational stage/processing emissions (t-C)':Column5,                                 
                                 'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column4,
                                 'F7-0: Landfill gas decomposition (t-C)':Column7})

dfM_pl = pd.DataFrame.from_dict({'Year':Column1,'F0-1: Biomass C sequestration (t-C)':Column14,
                                 #'9: Landfill storage (t-C)':Column13_1, 
                                 'F1-0: Residue decomposition (t-C)':Column3,
                                 'F6-0-1: Emissions from firewood/other energy use (t-C)':Column9,
                                 'F8-0: Operational stage/processing emissions (t-C)':Column11,
                                 'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column10,
                                 'F7-0: Landfill gas decomposition (t-C)':Column13})

#E
dfE_nu = pd.DataFrame.from_dict({'Year':Column1,'F0-1: Biomass C sequestration (t-C)':Column8,
                                 #'9: Landfill storage (t-C)':Column19_1, 
                                 'F1-0: Residue decomposition (t-C)':Column3,
                                 'F6-0-1: Emissions from firewood/other energy use (t-C)':Column15, 
                                 'F8-0: Operational stage/processing emissions (t-C)':Column17,
                                 'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column16,
                                 'F4-0: Emissions from wood pellets use (t-C)':Column15_1,
                                 'F7-0: Landfill gas decomposition (t-C)':Column19})

dfE_pl = pd.DataFrame.from_dict({'Year':Column1, 'F0-1: Biomass C sequestration (t-C)':Column14, 
                                 #'9: Landfill storage (t-C)':Column24_1,
                                 'F1-0: Residue decomposition (t-C)':Column3,
                                 'F6-0-1: Emissions from firewood/other energy use (t-C)':Column20, 
                                 'F8-0: Operational stage/processing emissions (t-C)':Column22,
                                 'F6-0-2: Energy use emissions from in-use stocks outflow (t-C)':Column21,
                                 'F4-0: Emissions from wood pellets use (t-C)':Column20_1,
                                 'F7-0: Landfill gas decomposition (t-C)':Column24})

    
    
    
writer = pd.ExcelWriter('C_flows_PF_PO_Ret.xlsx', engine = 'xlsxwriter')


dfM_nu.to_excel(writer, sheet_name = 'PF_PO_M_nu', header=True, index=False)
dfM_pl.to_excel(writer, sheet_name = 'PF_PO_M_pl', header=True, index=False)
dfE_nu.to_excel(writer, sheet_name = 'PF_PO_E_nu', header=True, index=False)
dfE_pl.to_excel(writer, sheet_name = 'PF_PO_E_pl', header=True, index=False)

writer.save()
writer.close()


#%%

#Step (22): Plot of the individual carbon emission and sequestration flows for normal and symlog-scale graphs


#PF_PO_M_nu

fig=plt.figure()
fig.show()
ax1=fig.add_subplot(111)


# plot
ax1.plot(t, flat_list_nucleus, color='darkkhaki', label='F0-1: Biomass C sequestration') 
#ax1.plot(t, OC_storage_S2nu, color='darkturquoise', label='9: Landfill storage') 
ax1.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax1.plot(t, c_firewood_energy_S2nu, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')  
ax1.plot(t, OpProcessing_PF_PO_S2nu, color='orange', label='F8-0: Operational stage/processing emissions')  
ax1.plot(t, TestDSM2nu.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax1.plot(t, Landfill_decomp_PF_PO_S2nu, color='yellow', label='F7-0: Landfill gas decomposition') 

ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax1.set_xlim(-1,200)



ax1.set_yscale('symlog')
 
ax1.set_xlabel('Time (year)')
ax1.set_ylabel('C flows (t-C) (symlog)')

ax1.set_title('Carbon flow, PF_PO_M_nucleus (symlog-scale)')

plt.show()


#%%

#plotting the individual C flows

#PF_PO_M_nu

f, (ax_a, ax_b) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax_a.plot(t, flat_list_nucleus, color='darkkhaki', label='F0-1: Biomass C sequestration') 
#ax_a.plot(t, OC_storage_S2nu, color='darkturquoise', label='9: Landfill storage') 
ax_a.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax_a.plot(t, c_firewood_energy_S2nu, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use')  
ax_a.plot(t, OpProcessing_PF_PO_S2nu, color='orange', label='F8-0: Operational stage/processing emissions')  
ax_a.plot(t, TestDSM2nu.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax_a.plot(t, Landfill_decomp_PF_PO_S2nu, color='yellow', label='F7-0: Landfill gas decomposition') 


ax_b.plot(t, c_firewood_energy_S2nu, color='mediumseagreen') 
ax_b.plot(t, decomp_emissions[:,0], color='lightcoral')
#ax_b.plot(t, OC_storage_S2nu, color='darkturquoise') 
ax_b.plot(t, TestDSM2nu.o, color='royalblue') 
ax_b.plot(t, OpProcessing_PF_PO_S2nu, color='orange') 
ax_b.plot(t, Landfill_decomp_PF_PO_S2nu, color='yellow') 
ax_b.plot(t, flat_list_nucleus, color='darkkhaki') 

#ax_a.set_yscale('log')
#ax_b.set_yscale('log')

# zoom-in / limit the view to different portions of the data
ax_a.set_xlim(-1,200)

ax_a.set_ylim(60, 75)  
ax_b.set_ylim(-25, 35)  
#ax_b.set_ylim(-0.3, 0.5)  



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

ax_a.set_title('Carbon flow, PF_PO_M_nucleus')
#plt.plot(t, Cflow_PF_SF_S1)
#plt.plot(t, Cflow_PF_SF_S2)
#plt.plot(t, Cflow_PF_SF_E)
#plt.xlim([0, 200])

plt.show()


#%%

#plot for the individual carbon flows - symlog-scale graphs

#PF_PO_M_pl

fig=plt.figure()
fig.show()
ax2=fig.add_subplot(111)



# plot
ax2.plot(t, flat_list_plasma, color='darkkhaki', label='F0-1: Biomass C sequestration') 
#ax2.plot(t, OC_storage_S2pl, color='darkturquoise', label='9: Landfill storage') 
ax2.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax2.plot(t, c_firewood_energy_S2pl, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax2.plot(t, OpProcessing_PF_PO_S2pl, color='orange', label='F8-0: Operational stage/processing emissions') 
ax2.plot(t, TestDSM2pl.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax2.plot(t, Landfill_decomp_PF_PO_S2pl, color='yellow', label='F7-0: Landfill gas decomposition') 

ax2.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax2.set_xlim(-1,200)



ax2.set_yscale('symlog')
 
ax2.set_xlabel('Time (year)')
ax2.set_ylabel('C flows (t-C) (symlog)')

ax2.set_title('Carbon flow, PF_PO_M_plasma (symlog-scale)')

plt.show()


#%%

#plotting the individual C flows

#PF_PO_M_pl

f, (ax_c, ax_d) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax_c.plot(t, flat_list_plasma, color='darkkhaki', label='F0-1: Biomass C sequestration') 
#ax_c.plot(t, OC_storage_S2pl, color='darkturquoise', label='9: Landfill storage') 
ax_c.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax_c.plot(t, c_firewood_energy_S2pl, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax_c.plot(t, OpProcessing_PF_PO_S2pl, color='orange', label='F8-0: Operational stage/processing emissions') 
ax_c.plot(t, TestDSM2pl.o, color='royalblue', label='F6-0-2: Energy use emissions from in-use stocks outflow') 
ax_c.plot(t, Landfill_decomp_PF_PO_S2pl, color='yellow', label='F7-0: Landfill gas decomposition') 


ax_d.plot(t, c_firewood_energy_S2pl, color='mediumseagreen') 
ax_d.plot(t, decomp_emissions[:,0], color='lightcoral')
#ax_d.plot(t, OC_storage_S2pl, color='darkturquoise') 
ax_d.plot(t, TestDSM2pl.o, color='royalblue') 
ax_d.plot(t, OpProcessing_PF_PO_S2pl, color='orange') 
ax_d.plot(t, Landfill_decomp_PF_PO_S2pl, color='yellow') 
ax_d.plot(t, flat_list_plasma, color='darkkhaki') 


# zoom-in / limit the view to different portions of the data
ax_c.set_xlim(-1,200)

ax_c.set_ylim(60, 75)  
ax_d.set_ylim(-25, 35)  

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

ax_c.set_title('Carbon flow, PF_PO_M_plasma')
#plt.plot(t, Cflow_PF_SF_S1)
#plt.plot(t, Cflow_PF_SF_S2)
#plt.plot(t, Cflow_PF_SF_E)
#plt.xlim([0, 200])

plt.show()


#%%

#plot for the individual carbon flows - symlog-scale graphs

#PF_PO_E_nu

fig=plt.figure()
fig.show()
ax3=fig.add_subplot(111)


# plot
ax3.plot(t, flat_list_nucleus, color='darkkhaki', label='F0-1: Biomass C sequestration')  
#ax3.plot(t, OC_storage_Enu, color='darkturquoise', label='9: Landfill storage') 
ax3.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax3.plot(t, c_firewood_energy_Enu, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax3.plot(t, OpProcessing_PF_PO_Enu, color='orange', label='F8-0: Operational stage/processing emissions') 
ax3.plot(t, Landfill_decomp_PF_PO_Enu, color='yellow', label='F7-0: Landfill gas decomposition') 
ax3.plot(t, c_pellets_Enu, color='slategrey', label='F4-0: Emissions from wood pellets use') 
#ax3.plot(t, TestDSM3nu.o, color='royalblue', label='in-use stock output') 

ax3.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax3.set_xlim(-1,200)



ax3.set_yscale('symlog')
 
ax3.set_xlabel('Time (year)')
ax3.set_ylabel('C flows (t-C) (symlog)')

ax3.set_title('Carbon flow, PF_PO_E_nucleus (symlog-scale)')

plt.show()

#%%

#plotting the individual C flows

#PF_PO_E_nu

f, (ax_e, ax_f) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax_e.plot(t, flat_list_nucleus, color='darkkhaki', label='F0-1: Biomass C sequestration')  
#ax_e.plot(t, OC_storage_Enu, color='darkturquoise', label='9: Landfill storage') 
ax_e.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax_e.plot(t, c_firewood_energy_Enu, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax_e.plot(t, OpProcessing_PF_PO_Enu, color='orange', label='F8-0: Operational stage/processing emissions') 
ax_e.plot(t, Landfill_decomp_PF_PO_Enu, color='yellow', label='F7-0: Landfill gas decomposition') 
ax_e.plot(t, c_pellets_Enu, color='slategrey', label='F4-0: Emissions from wood pellets use') 
#ax_e.plot(t, TestDSM3nu.o, color='royalblue', label='in-use stock output') 



ax_f.plot(t, c_firewood_energy_Enu, color='mediumseagreen') 
ax_f.plot(t, decomp_emissions[:,0], color='lightcoral')
ax_f.plot(t, c_pellets_Enu, color='slategrey') 
#ax_f.plot(t, TestDSM3nu.o, color='royalblue') 
#ax_f.plot(t, OC_storage_Enu, color='darkturquoise') 
ax_f.plot(t, OpProcessing_PF_PO_Enu, color='orange') 
ax_f.plot(t, Landfill_decomp_PF_PO_Enu, color='yellow') 
ax_f.plot(t, flat_list_nucleus, color='darkkhaki') 


# zoom-in / limit the view to different portions of the data
ax_e.set_xlim(-1,200)

ax_e.set_ylim(170, 190)  
ax_f.set_ylim(-25, 30)  


# hide the spines between ax and ax2
ax_e.spines['bottom'].set_visible(False)
ax_f.spines['top'].set_visible(False)
ax_e.xaxis.tick_top()
ax_e.tick_params(labeltop=False)  # don't put tick labels at the top
ax_f.xaxis.tick_bottom()

ax_e.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

d = .012  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_e.transAxes, color='k', clip_on=False)
ax_e.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_e.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_f.transAxes)  # switch to the bottom axes
ax_f.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_f.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



ax_f.set_xlabel('Time (year)')
ax_f.set_ylabel('C flows (t-C)')
ax_e.set_ylabel('C flows (t-C)')

ax_e.set_title('Carbon flow, PF_PO_E_nucleus')
#plt.plot(t, Cflow_PF_SF_S1)
#plt.plot(t, Cflow_PF_SF_S2)
#plt.plot(t, Cflow_PF_SF_E)
#plt.xlim([0, 200])

plt.show()

#%%

#plot for the individual carbon flows - symlog-scale graphs

#PF_PO_E_pl

fig=plt.figure()
fig.show()
ax4=fig.add_subplot(111)



# plot
ax4.plot(t, flat_list_plasma, color='darkkhaki', label='F0-1: Biomass C sequestration') 
#ax4.plot(t, OC_storage_Epl, color='darkturquoise', label='9: Landfill storage')
ax4.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax4.plot(t, c_firewood_energy_Epl, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax4.plot(t, OpProcessing_PF_PO_Epl, color='orange', label='F8-0: Operational stage/processing emissions') 
ax4.plot(t, Landfill_decomp_PF_PO_Epl, color='yellow', label='F7-0: Landfill gas decomposition')  
#ax4.plot(t, TestDSM3pl.o, color='royalblue', label='in-use stock output') 
ax4.plot(t, c_pellets_Epl, color='slategrey', label='F4-0: Emissions from wood pellets use')

ax4.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax4.set_xlim(-1,200)



ax4.set_yscale('symlog')
 
ax4.set_xlabel('Time (year)')
ax4.set_ylabel('C flows (t-C) (symlog)')

ax4.set_title('Carbon flow, PF_PO_E_plasma (symlog-scale)')

plt.show()

#%%

#plotting the individual C flows

#PF_PO_E_pl

f, (ax_g, ax_h) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax_g.plot(t, flat_list_plasma, color='darkkhaki', label='F0-1: Biomass C sequestration') 
#ax_g.plot(t, OC_storage_Epl, color='darkturquoise', label='9: Landfill storage')
ax_g.plot(t, decomp_emissions[:,0], color='lightcoral', label='F1-0: Residue decomposition')
ax_g.plot(t, c_firewood_energy_Epl, color='mediumseagreen', label='F6-0-1: Emissions from firewood/other energy use') 
ax_g.plot(t, OpProcessing_PF_PO_Epl, color='orange', label='F8-0: Operational stage/processing emissions') 
ax_g.plot(t, Landfill_decomp_PF_PO_Epl, color='yellow', label='F7-0: Landfill gas decomposition')  
#ax_g.plot(t, TestDSM3pl.o, color='royalblue', label='in-use stock output') 
ax_g.plot(t, c_pellets_Epl, color='slategrey', label='F4-0: Emissions from wood pellets use') 


ax_h.plot(t, c_firewood_energy_Epl, color='mediumseagreen') 
ax_h.plot(t, c_pellets_Epl, color='slategrey')
ax_h.plot(t, decomp_emissions[:,0], color='lightcoral')
#ax_h.plot(t, TestDSM3pl.o, color='royalblue') 
ax_h.plot(t, OpProcessing_PF_PO_Epl, color='orange') 
#ax_h.plot(t, OC_storage_Epl, color='darkturquoise')
ax_h.plot(t, Landfill_decomp_PF_PO_Epl, color='yellow') 
ax_h.plot(t, flat_list_plasma, color='darkkhaki') 


# zoom-in / limit the view to different portions of the data
ax_g.set_xlim(-1,200)

ax_g.set_ylim(170, 190)  
ax_h.set_ylim(-25, 30)  

# hide the spines between ax and ax2
ax_g.spines['bottom'].set_visible(False)
ax_h.spines['top'].set_visible(False)
ax_g.xaxis.tick_top()
ax_g.tick_params(labeltop=False)  # don't put tick labels at the top
ax_h.xaxis.tick_bottom()

ax_g.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

d = .012  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_g.transAxes, color='k', clip_on=False)
ax_g.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_g.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_h.transAxes)  # switch to the bottom axes
ax_h.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_h.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



ax_h.set_xlabel('Time (year)')
ax_h.set_ylabel('C flows (t-C)')
ax_g.set_ylabel('C flows (t-C)')

ax_g.set_title('Carbon flow, PF_PO_E_plasma')
#plt.plot(t, Cflow_PF_SF_S1)
#plt.plot(t, Cflow_PF_SF_S2)
#plt.plot(t, Cflow_PF_SF_E)
#plt.xlim([0, 200])

plt.show()

#%%

#Step (23): Generate the excel file for the net carbon balance


Agg_Cflow_PF_PO_S2nu = [c_firewood_energy_S2nu, decomp_emissions[:,0], TestDSM2nu.o, OpProcessing_PF_PO_S2nu, Landfill_decomp_PF_PO_S2nu, flat_list_nucleus]
Agg_Cflow_PF_PO_S2pl = [c_firewood_energy_S2pl, decomp_emissions[:,0], TestDSM2pl.o, OpProcessing_PF_PO_S2pl, Landfill_decomp_PF_PO_S2pl, flat_list_plasma]
Agg_Cflow_PF_PO_Enu = [c_firewood_energy_Enu, c_pellets_Enu, decomp_emissions[:,0], TestDSM3nu.o, OpProcessing_PF_PO_Enu, Landfill_decomp_PF_PO_Enu, flat_list_nucleus]
Agg_Cflow_PF_PO_Epl = [c_firewood_energy_Epl, c_pellets_Epl, decomp_emissions[:,0], TestDSM3pl.o, OpProcessing_PF_PO_Epl, Landfill_decomp_PF_PO_Epl, flat_list_plasma]


Agg_Cflow_PF_PO_S2nu = [sum(x) for x in zip(*Agg_Cflow_PF_PO_S2nu)]
Agg_Cflow_PF_PO_S2pl = [sum(x) for x in zip(*Agg_Cflow_PF_PO_S2pl)]
Agg_Cflow_PF_PO_Enu = [sum(x) for x in zip(*Agg_Cflow_PF_PO_Enu)]
Agg_Cflow_PF_PO_Epl = [sum(x) for x in zip(*Agg_Cflow_PF_PO_Epl)]


fig=plt.figure()
fig.show()
ax5=fig.add_subplot(111)


# plot
ax5.plot(t, Agg_Cflow_PF_PO_S2nu, color='orange', label='M_nucleus') 
ax5.plot(t, Agg_Cflow_PF_PO_S2pl, color='darkturquoise', label='M_plasma') 
ax5.plot(t, Agg_Cflow_PF_PO_Enu, color='lightcoral', label='E_nucleus')
ax5.plot(t, Agg_Cflow_PF_PO_Epl, color='mediumseagreen', label='E_plasma')  


ax5.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

ax5.set_xlim(-0.3,85)

#ax5.set_yscale('symlog')
 
ax5.set_xlabel('Time (year)')
ax5.set_ylabel('C flows (t-C)')

ax5.set_title('Net carbon balance, PF_PO')

plt.show()

#create column year
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)

#Create colum results
dfM_PF_PO = pd.DataFrame.from_dict({'Year':year,'M_nucleus (t-C)':Agg_Cflow_PF_PO_S2nu, 'M_plasma (t-C)':Agg_Cflow_PF_PO_S2pl,
                                         'E_nucleus (t-C)':Agg_Cflow_PF_PO_Enu, 'E_plasma (t-C)':Agg_Cflow_PF_PO_Epl})

    
#Export to excel
writer = pd.ExcelWriter('AggCFlow_PF_PO_Ret.xlsx', engine = 'xlsxwriter')


dfM_PF_PO.to_excel(writer, sheet_name = 'PF_PO', header=True, index=False)

writer.save()
writer.close()


#%%

#Step (24): Plot the net carbon balance 

##Net carbon balance for M and E (axis break)

f, (ax5a, ax5b) = plt.subplots(2, 1, sharex=True)

ax5a.plot(t, Agg_Cflow_PF_PO_S2nu, color='orange', label='M_nucleus') 
ax5a.plot(t, Agg_Cflow_PF_PO_S2pl, color='darkturquoise', label='M_plasma') 
ax5a.plot(t, Agg_Cflow_PF_PO_Enu, color='lightcoral', label='E_nucleus')
ax5a.plot(t, Agg_Cflow_PF_PO_Epl, color='mediumseagreen', label='E_plasma')  

ax5a.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)

ax5b.plot(t, Agg_Cflow_PF_PO_S2nu, color='orange') 
ax5b.plot(t, Agg_Cflow_PF_PO_S2pl, color='darkturquoise') 
ax5b.plot(t, Agg_Cflow_PF_PO_Enu, color='lightcoral')
ax5b.plot(t, Agg_Cflow_PF_PO_Epl, color='mediumseagreen')  

ax5b.plot(t, zerolistmaker(tf-1), color='black', label='Zero line', ls='--', alpha=0.75)
 

# zoom-in / limit the view to different portions of the data
ax5a.set_xlim(-0.35,85)
#ax5a.set_xlim(-1,200)
ax5a.set_ylim(210, 230)  


ax5b.set_xlim(-0.35,85)
#ax5b.set_xlim(-1,200)
ax5b.set_ylim(-5, 50)  


# hide the spines between ax and ax2
ax5a.spines['bottom'].set_visible(False)
ax5b.spines['top'].set_visible(False)
ax5a.xaxis.tick_top()
ax5a.tick_params(labeltop=False)  # don't put tick labels at the top
ax5b.xaxis.tick_bottom()

ax5a.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

d = .012  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax5a.transAxes, color='k', clip_on=False)
ax5a.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax5a.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax5b.transAxes)  # switch to the bottom axes
ax5b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax5b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



ax5b.set_xlabel('Time (year)')
ax5b.set_ylabel('C flows (t-C)')
ax5a.set_ylabel('C flows (t-C)')

ax5a.set_title('Net carbon balance, PF_PO')

plt.show()


#%%

#Step (25): Generate the excel file for documentation of individual carbon flows in the system definition (Fig. 1)


#print year column
year = []
for x in range (0, 201): 
    year.append(x) 
print (year)


df2nu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2nu')
df2pl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_S2pl')
dfEnu = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Enu')
dfEpl = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_PO.xlsx', 'PF_PO_Epl')



Column1 = year
division = 1000*44/12
division_CH4 = 1000*16/12


## S2nu
## define the input flow for the landfill (F5-7)
OC_storage_S2nu = df2nu['Other_C_storage'].values


OC_storage_S2nu = [x/division for x in OC_storage_S2nu]
OC_storage_S2nu = [abs(number) for number in OC_storage_S2nu]

C_LF_S2nu = [x*1/0.82 for x in OC_storage_S2nu]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_S2nu = [x/division for x in df2nu['Input_PF'].values]
HWP_S2nu_energy =  [x*1/3 for x in c_firewood_energy_S2nu]
HWP_S2nu_landfill = [x*1/0.82 for x in OC_storage_S2nu]

HWP_S2nu_sum = [HWP_S2nu, HWP_S2nu_energy, HWP_S2nu_landfill]
HWP_S2nu_sum = [sum(x) for x in zip(*HWP_S2nu_sum)]

#in-use stocks (S-4)
TestDSM2nu.s = [x/division for x in TestDSM2nu.s]
#TestDSM2nu.i = [x/division for x in TestDSM2nu.i]



# calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_S2nu = (tf,1)
stocks_S2nu = np.zeros(zero_matrix_stocks_S2nu)


i = 0
stocks_S2nu[0] = C_LF_S2nu[0] - Landfill_decomp_PF_PO_S2nu[0]

while i < tf-1:
    stocks_S2nu[i+1] = np.array(C_LF_S2nu[i+1] - Landfill_decomp_PF_PO_S2nu[i+1] + stocks_S2nu[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_S2nu = [x1+x2 for (x1,x2) in zip(HWP_S2nu_sum, [x*2/3 for x in c_firewood_energy_S2nu])] 


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2nu = (tf,1)
ForCstocks_S2nu = np.zeros(zero_matrix_ForCstocks_S2nu)

i = 0
ForCstocks_S2nu[0] = initAGB - flat_list_nucleus[0] - decomp_emissions[0] - HWP_logged_S2nu[0]

while i < tf-1:
    ForCstocks_S2nu[i+1] = np.array(ForCstocks_S2nu[i] - flat_list_nucleus[i+1] - decomp_emissions[i+1] - HWP_logged_S2nu[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
df2nu_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\NonRW_PF_PO.xlsx', 'PF_PO_S2nu')
NonRW_amount_S2nu = df2nu_amount['NonRW_amount'].values

NonRW_amount_S2nu = [x/1000 for x in NonRW_amount_S2nu]



##NonRW emissions (F9-0-2)
emission_NonRW_S2nu = [x/division for x in emission_NonRW_S2nu]
    

#create columns
dfM_nu = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_nucleus,
                                    'F1-0 (t-C)': decomp_emissions[:,0],
                                    #'F1a-2 (t-C)': PF_PO_S2nu,
                                    #'F1c-2 (t-C)': FP_PO_S2nu,
                                    'F1-2 (t-C)': HWP_logged_S2nu, 
                                    'St-1 (t-C)':ForCstocks_S2nu[:,0], 
                                    'F2-3 (t-C)': HWP_S2nu_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_S2nu], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_S2nu_sum, [x*1/0.82 for x in OC_storage_S2nu], [x*1/3 for x in c_firewood_energy_S2nu])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_S2nu],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_S2nu], 
                                   # 'F4-0 (t-C)':,
                                    'St-4 (t-C)': TestDSM2nu.s, 
                                    #'S-4-i (t-C)': TestDSM2nu.i,
                                    'F4-5 (t-C)': TestDSM2nu.o,
                                    'F5-6 (t-C)': TestDSM2nu.o, 
                                    'F5-7 (t-C)': C_LF_S2nu,
                                    'F6-0-1 (t-C)': c_firewood_energy_S2nu,
                                    'F6-0-2 (t-C)': TestDSM2nu.o,
                                    'St-7 (t-C)': stocks_S2nu[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_PF_PO_S2nu,
                                    'F8-0 (t-C)': OpProcessing_PF_PO_S2nu,
                                    'S9-0 (t)': NonRW_amount_S2nu, 
                                    'F9-0 (t-C)': emission_NonRW_S2nu,
                                    })

    
    
    
##S2pl
## define the input flow for the landfill (F5-7)
OC_storage_S2pl = df2pl['Other_C_storage'].values


OC_storage_S2pl = [x/division for x in OC_storage_S2pl]
OC_storage_S2pl = [abs(number) for number in OC_storage_S2pl]

C_LF_S2pl = [x*1/0.82 for x in OC_storage_S2pl]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_S2pl = [x/division for x in df2pl['Input_PF'].values]
HWP_S2pl_energy =  [x*1/3 for x in c_firewood_energy_S2pl]
HWP_S2pl_landfill = [x*1/0.82 for x in OC_storage_S2pl]

HWP_S2pl_sum = [HWP_S2pl, HWP_S2pl_energy, HWP_S2pl_landfill]
HWP_S2pl_sum = [sum(x) for x in zip(*HWP_S2pl_sum)]

#in-use stocks (S-4)
TestDSM2pl.s = [x/division for x in TestDSM2pl.s]
#TestDSM2pl.i = [x/division for x in TestDSM2pl.i]



# calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_S2pl = (tf,1)
stocks_S2pl = np.zeros(zero_matrix_stocks_S2pl)


i = 0
stocks_S2pl[0] = C_LF_S2pl[0] - Landfill_decomp_PF_PO_S2pl[0]

while i < tf-1:
    stocks_S2pl[i+1] = np.array(C_LF_S2pl[i+1] - Landfill_decomp_PF_PO_S2pl[i+1] + stocks_S2pl[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_S2pl = [x1+x2 for (x1,x2) in zip(HWP_S2pl_sum, [x*2/3 for x in c_firewood_energy_S2pl])] 


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_S2pl = (tf,1)
ForCstocks_S2pl = np.zeros(zero_matrix_ForCstocks_S2pl)

i = 0
ForCstocks_S2pl[0] = initAGB - flat_list_plasma[0] - decomp_emissions[0] - HWP_logged_S2pl[0]

while i < tf-1:
    ForCstocks_S2pl[i+1] = np.array(ForCstocks_S2pl[i] - flat_list_plasma[i+1] - decomp_emissions[i+1] - HWP_logged_S2pl[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
df2pl_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\NonRW_PF_PO.xlsx', 'PF_PO_S2pl')
NonRW_amount_S2pl = df2pl_amount['NonRW_amount'].values

NonRW_amount_S2pl = [x/1000 for x in NonRW_amount_S2pl]



##NonRW emissions (F9-0-2)
emission_NonRW_S2pl = [x/division for x in emission_NonRW_S2pl]
    

#create columns
dfM_pl = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_plasma,
                                    'F1-0 (t-C)': decomp_emissions[:,0],
                                    #'F1a-2 (t-C)': PF_PO_S2pl,
                                    #'F1c-2 (t-C)': FP_PO_S2pl,
                                    'F1-2 (t-C)': HWP_logged_S2pl, 
                                    'St-1 (t-C)':ForCstocks_S2pl[:,0], 
                                    'F2-3 (t-C)': HWP_S2pl_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_S2pl], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_S2pl_sum, [x*1/0.82 for x in OC_storage_S2pl], [x*1/3 for x in c_firewood_energy_S2pl])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_S2pl],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_S2pl], 
                                   # 'F4-0 (t-C)':,
                                    'St-4 (t-C)': TestDSM2pl.s, 
                                    #'S-4-i (t-C)': TestDSM2pl.i,
                                    'F4-5 (t-C)': TestDSM2pl.o,
                                    'F5-6 (t-C)': TestDSM2pl.o, 
                                    'F5-7 (t-C)': C_LF_S2pl,
                                    'F6-0-1 (t-C)': c_firewood_energy_S2pl,
                                    'F6-0-2 (t-C)': TestDSM2pl.o,
                                    'St-7 (t-C)': stocks_S2pl[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_PF_PO_S2pl,
                                    'F8-0 (t-C)': OpProcessing_PF_PO_S2pl,
                                    'S9-0 (t)': NonRW_amount_S2pl, 
                                    'F9-0 (t-C)': emission_NonRW_S2pl,
                                    })


##Enu
## define the input flow for the landfill (F5-7)
OC_storage_Enu = dfEnu['Other_C_storage'].values


OC_storage_Enu = [x/division for x in OC_storage_Enu]
OC_storage_Enu = [abs(number) for number in OC_storage_Enu]

C_LF_Enu = [x*1/0.82 for x in OC_storage_Enu]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_Enu = [x/division for x in dfEnu['Wood_pellets'].values]
HWP_Enu_energy =  [x*1/3 for x in c_firewood_energy_Enu]
HWP_Enu_landfill = [x*1/0.82 for x in OC_storage_Enu]

HWP_Enu_sum = [HWP_Enu, HWP_Enu_energy, HWP_Enu_landfill]
HWP_Enu_sum = [sum(x) for x in zip(*HWP_Enu_sum)]

#in-use stocks (S-4)
TestDSM3nu.s = [x/division for x in TestDSM3nu.s]
#TestDSM3nu.i = [x/division for x in TestDSM3nu.i]



# calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_Enu = (tf,1)
stocks_Enu = np.zeros(zero_matrix_stocks_Enu)


i = 0
stocks_Enu[0] = C_LF_Enu[0] - Landfill_decomp_PF_PO_Enu[0]

while i < tf-1:
    stocks_Enu[i+1] = np.array(C_LF_Enu[i+1] - Landfill_decomp_PF_PO_Enu[i+1] + stocks_Enu[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_Enu = [x1+x2 for (x1,x2) in zip(HWP_Enu_sum, [x*2/3 for x in c_firewood_energy_Enu])] 


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_Enu = (tf,1)
ForCstocks_Enu = np.zeros(zero_matrix_ForCstocks_Enu)

i = 0
ForCstocks_Enu[0] = initAGB - flat_list_nucleus[0] - decomp_emissions[0] - HWP_logged_Enu[0]

while i < tf-1:
    ForCstocks_Enu[i+1] = np.array(ForCstocks_Enu[i] - flat_list_nucleus[i+1] - decomp_emissions[i+1] - HWP_logged_Enu[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
dfEnu_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\NonRW_PF_PO.xlsx', 'PF_PO_Enu')
NonRW_amount_Enu = dfEnu_amount['NonRW_amount'].values

NonRW_amount_Enu = [x/1000 for x in NonRW_amount_Enu]



##NonRW emissions (F9-0-2)
emission_NonRW_Enu = [x/division for x in emission_NonRW_Enu]
    

#create columns
dfE_nu = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_nucleus,
                                    'F1-0 (t-C)': decomp_emissions[:,0],
                                    #'F1a-2 (t-C)': PF_PO_Enu,
                                    #'F1c-2 (t-C)': FP_PO_Enu,
                                    'F1-2 (t-C)': HWP_logged_Enu, 
                                    'St-1 (t-C)':ForCstocks_Enu[:,0], 
                                    'F2-3 (t-C)': HWP_Enu_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_Enu], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_Enu_sum, [x*1/0.82 for x in OC_storage_Enu], [x*1/3 for x in c_firewood_energy_Enu])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_Enu],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_Enu], 
                                    'F4-0 (t-C)':c_pellets_Enu,
                                    'St-4 (t-C)': TestDSM3nu.s, 
                                    #'S-4-i (t-C)': TestDSM3nu.i,
                                    'F4-5 (t-C)': TestDSM3nu.o,
                                    'F5-6 (t-C)': TestDSM3nu.o, 
                                    'F5-7 (t-C)': C_LF_Enu,
                                    'F6-0-1 (t-C)': c_firewood_energy_Enu,
                                    'F6-0-2 (t-C)': TestDSM3nu.o,
                                    'St-7 (t-C)': stocks_Enu[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_PF_PO_Enu,
                                    'F8-0 (t-C)': OpProcessing_PF_PO_Enu,
                                    'S9-0 (t)': NonRW_amount_Enu, 
                                    'F9-0 (t-C)': emission_NonRW_Enu,
                                    })
    
    
    

##Epl
## define the input flow for the landfill (F5-7)
OC_storage_Epl = dfEpl['Other_C_storage'].values


OC_storage_Epl = [x/division for x in OC_storage_Epl]
OC_storage_Epl = [abs(number) for number in OC_storage_Epl]

C_LF_Epl = [x*1/0.82 for x in OC_storage_Epl]


## define the input flow from the logging/harvesting to wood materials/pellets processing (F2-3)
HWP_Epl = [x/division for x in dfEpl['Wood_pellets'].values]
HWP_Epl_energy =  [x*1/3 for x in c_firewood_energy_Epl]
HWP_Epl_landfill = [x*1/0.82 for x in OC_storage_Epl]

HWP_Epl_sum = [HWP_Epl, HWP_Epl_energy, HWP_Epl_landfill]
HWP_Epl_sum = [sum(x) for x in zip(*HWP_Epl_sum)]

#in-use stocks (S-4)
TestDSM3pl.s = [x/division for x in TestDSM3pl.s]
#TestDSM3pl.i = [x/division for x in TestDSM3pl.i]



# calculate C stocks in landfill (S-7)
tf = 201

zero_matrix_stocks_Epl = (tf,1)
stocks_Epl = np.zeros(zero_matrix_stocks_Epl)


i = 0
stocks_Epl[0] = C_LF_Epl[0] - Landfill_decomp_PF_PO_Epl[0]

while i < tf-1:
    stocks_Epl[i+1] = np.array(C_LF_Epl[i+1] - Landfill_decomp_PF_PO_Epl[i+1] + stocks_Epl[i])
    i = i + 1



## calculate aggregate flow of logged wood (F1-2)
HWP_logged_Epl = [x1+x2 for (x1,x2) in zip(HWP_Epl_sum, [x*2/3 for x in c_firewood_energy_Epl])] 


## calculate the stocks in the forest (AGB + undecomposed residue) (S-1a+S-1c)
tf = 201

zero_matrix_ForCstocks_Epl = (tf,1)
ForCstocks_Epl = np.zeros(zero_matrix_ForCstocks_Epl)

i = 0
ForCstocks_Epl[0] = initAGB - flat_list_plasma[0] - decomp_emissions[0] - HWP_logged_Epl[0]

while i < tf-1:
    ForCstocks_Epl[i+1] = np.array(ForCstocks_Epl[i] - flat_list_plasma[i+1] - decomp_emissions[i+1] - HWP_logged_Epl[i+1])
    i = i + 1


##NonRW materials/energy amount (F9-0-1)
dfEpl_amount = pd.read_excel('C:\\Work\\Programming\\Practice\\NonRW_PF_PO.xlsx', 'PF_PO_Epl')
NonRW_amount_Epl = dfEpl_amount['NonRW_amount'].values

NonRW_amount_Epl = [x/1000 for x in NonRW_amount_Epl]



##NonRW emissions (F9-0-2)
emission_NonRW_Epl = [x/division for x in emission_NonRW_Epl]
    

#create columns
dfE_pl = pd.DataFrame.from_dict({'Year':Column1, 
                                    'F0-1 (t-C)': flat_list_plasma,
                                    'F1-0 (t-C)': decomp_emissions[:,0],
                                    #'F1a-2 (t-C)': PF_PO_Epl,
                                    #'F1c-2 (t-C)': FP_PO_Epl,
                                    'F1-2 (t-C)': HWP_logged_Epl, 
                                    'St-1 (t-C)':ForCstocks_Epl[:,0], 
                                    'F2-3 (t-C)': HWP_Epl_sum,  
                                    'F2-6 (t-C)': [x*2/3 for x in c_firewood_energy_Epl], 
                                    'SM/E (t-C)': [x1-x2-x3 for (x1,x2,x3) in zip(HWP_Epl_sum, [x*1/0.82 for x in OC_storage_Epl], [x*1/3 for x in c_firewood_energy_Epl])],
                                    'F3-5 (t-C)':[x*1/0.82 for x in OC_storage_Epl],
                                    'F3-6 (t-C)': [x*1/3 for x in c_firewood_energy_Epl], 
                                    'F4-0 (t-C)': c_pellets_Epl,
                                    'St-4 (t-C)': TestDSM3pl.s, 
                                    #'S-4-i (t-C)': TestDSM3pl.i,
                                    'F4-5 (t-C)': TestDSM3pl.o,
                                    'F5-6 (t-C)': TestDSM3pl.o, 
                                    'F5-7 (t-C)': C_LF_Epl,
                                    'F6-0-1 (t-C)': c_firewood_energy_Epl,
                                    'F6-0-2 (t-C)': TestDSM3pl.o,
                                    'St-7 (t-C)': stocks_Epl[:,0],
                                    'F7-0 (t-C)': Landfill_decomp_PF_PO_Epl,
                                    'F8-0 (t-C)': OpProcessing_PF_PO_Epl,
                                    'S9-0 (t)': NonRW_amount_Epl, 
                                    'F9-0 (t-C)': emission_NonRW_Epl,
                                    })



writer = pd.ExcelWriter('C_flows_SysDef_PF_PO_Ret.xlsx', engine = 'xlsxwriter')


dfM_nu.to_excel(writer, sheet_name = 'PF_PO_Mnu', header=True, index=False)
dfM_pl.to_excel(writer, sheet_name = 'PF_PO_Mpl', header=True, index=False)
dfE_nu.to_excel(writer, sheet_name = 'PF_PO_E2nu', header=True, index=False)
dfE_pl.to_excel(writer, sheet_name = 'PF_PO_E2pl', header=True, index=False)


writer.save()
writer.close()
#%%
