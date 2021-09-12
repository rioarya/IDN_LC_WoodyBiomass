# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:21:55 2019

@author: raryapratama
"""

#%%


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
import pandas as pd


#PF_SF Scenario

##Set parameters
#Parameters for primary forest
initAGB = 233            #source: van Beijma et al. (2018)
initAGB_min = 233-72
initAGB_max = 233 + 72

#parameters for secondary forest. Sourc: Busch et al. (2019)
coeff_MF_nonpl = 11.47
coeff_DF_nonpl = 11.24
coeff_GL_nonpl = 9.42
coeff_MF_pl =17.2

tf = 201

a = 0.082
b = 2.53
#%%

#Step(1): C loss from the harvesting/clear cut

df3 = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S1_mat')


t = range(0,tf,1)

c_loss_S1_mat = df3['C_loss'].values


print(c_loss_S1_mat)


#%%

#Step (2) Aboveground biomass (AGB) decomposition



#S1_mat
df = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S1_mat')

tf = 201

t = np.arange(tf)


def decomp_E_trial(t,remainAGB_S1_mat):
    return (1-(1-np.exp(-a*t))**b)*remainAGB_S1_mat



#set zero matrix
output_decomp_S1_mat = np.zeros((len(t),len(df['C_remainAGB'].values)))


for i,remain_part_S1_mat in enumerate(df['C_remainAGB'].values):
    #print(i,remain_part)
    output_decomp_S1_mat[i:,i] = decomp_E_trial(t[:len(t)-i],remain_part_S1_mat)

print(output_decomp_S1_mat[:,:4])



#find the yearly emissions from decomposition by calculating the differences between elements in list 'decomp_tot_S1' 
#(https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
# https://stackoverflow.com/questions/11095892/numpy-difference-between-neighboring-elements

#difference between element, 
subs_matrix_S1_mat = np.zeros((len(t)-1,len(df['C_remainAGB'].values-1)))

i = 0
while i < tf:
    subs_matrix_S1_mat[:,i] = np.diff(output_decomp_S1_mat[:,i])
    i = i + 1 

print(subs_matrix_S1_mat[:,:4])
print(len(subs_matrix_S1_mat))



#since there is no carbon emission from decomposition at the beginning of the year (esp. from 'year 1' onward), 
#we have to replace the positive numbers with 0 values (https://stackoverflow.com/questions/36310897/how-do-i-change-all-negative-numbers-to-zero-in-python/36310913)
subs_matrix_S1_mat = subs_matrix_S1_mat.clip(max=0)

print(subs_matrix_S1_mat[:,:4])

#make the results as absolute values
subs_matrix_S1_mat = abs(subs_matrix_S1_mat)
print(subs_matrix_S1_mat[:,:4])


#insert row of zeros into the first row of the subs_matrix
zero_matrix_S1_mat = np.zeros((len(t)-200,len(df['C_remainAGB'].values)))
print(zero_matrix_S1_mat)

subs_matrix_S1_mat = np.vstack((zero_matrix_S1_mat, subs_matrix_S1_mat))

print(subs_matrix_S1_mat[:,:4])


#sum every column of the subs_matrix into one vector matrix
matrix_tot_S1_mat = (tf,1)
decomp_tot_S1_mat = np.zeros(matrix_tot_S1_mat) 

i = 0
while i < tf:
    decomp_tot_S1_mat[:,0] = decomp_tot_S1_mat[:,0] + subs_matrix_S1_mat[:,i]
    i = i + 1

print(decomp_tot_S1_mat[:,0])



#plotting
t = np.arange(0,tf)

plt.plot(t,decomp_tot_S1_mat,label='test_mat')


plt.xlim(0,200)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)

plt.show()




#%%

#Step (3): Non-renewable (steel) material substitution


t = range(0,tf,1)

#test_mat
df1_mat = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S1_mat')

subs_S1_mat = df1_mat['Subs_NonRW'].values


print(subs_S1_mat)






#%%

#Step (4): Dynamic stock model of wood harvested from primary forest and oil palm lumber

#HWP from primary forest, 35 year-old building materials lifetime



from dynamic_stock_model import DynamicStockModel


df1_mat = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S1_mat')


TestDSM1_mat = DynamicStockModel(t = df1_mat['Year'].values, i = df1_mat['Input_PF'].values, lt = {'Type': 'Normal', 'Mean': np.array([35]), 'StdDev': np.array([10.5])})


CheckStr1_mat, ExitFlag1_mat = TestDSM1_mat.dimension_check()


Stock_by_cohort1_mat, ExitFlag1_mat = TestDSM1_mat.compute_s_c_inflow_driven()



S1_mat, ExitFlag1_mat   = TestDSM1_mat.compute_stock_total()


O_C1_mat, ExitFlag1_mat = TestDSM1_mat.compute_o_c_from_s_c()



O1_mat, ExitFlag1_mat   = TestDSM1_mat.compute_outflow_total()



DS1_mat, ExitFlag1_mat  = TestDSM1_mat.compute_stock_change()



Bal1_mat, ExitFlag1_mat = TestDSM1_mat.check_stock_balance()



#print output flow

print(TestDSM1_mat.o)



#find the yearly emissions from stock outflow by calculating the differences between elements in list 'TestDSM1.s' (https://stackoverflow.com/questions/5314241/difference-between-consecutive-elements-in-list)
#TestDSM1.s = [p - q for q, p in zip(TestDSM1.s, TestDSM1.s[1:])]

#make the results as absolute values (https://stackoverflow.com/questions/20832769/how-to-obtain-absolute-value-of-numbers-of-a-list)
#TestDSM1.s = [abs(number) for number in TestDSM1.s]

#insert 0 value to the list as the first element, because there is no emissions due to the outflow from the stocks in year 0 (https://stackoverflow.com/questions/17911091/append-integer-to-beginning-of-list-in-python)
#var = 0
#TestDSM1.s.insert(0,var)


#print(TestDSM1.s)


#plt.plot(TestDSM1.s)
#plt.xlim([0, 100])
#plt.ylim([0,50])

#plt.show()


#%%

#Step (5): Secondary forest biomass growth (Busch et al. 2019)

t = range(0,tf,1)


#calculate the biomass and carbon content of moist forest
def Cgrowth_1(t):
    return (44/12*1000*coeff_MF_nonpl*(np.sqrt(t)))

flat_list_moist = Cgrowth_1(t)


#calculate the biomass and carbon content of moist forest
def Cgrowth_2(t):
    return (44/12*1000*coeff_DF_nonpl*(np.sqrt(t)))

flat_list_dry = Cgrowth_2(t)

print(flat_list_dry[200])

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


df1_mat = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S1_mat')


t = range(0,tf,1)

PH_Emissions_HWP1_S1_mat = df1_mat['PH_Emissions_HWP'].values





#%%

#Step (7): other C storage (inert-landfill)


t = range(0,tf,1)

#S1_mat
df1_mat = pd.read_excel('C:\\Work\\Programming\\Practice\\PF_SF.xlsx', 'PF_SF_S1_mat')

OC_storage_S1_mat = df1_mat['Other_C_storage'].values

print(OC_storage_S1_mat)


#%%
#Step (8), sum the emissions


#https://stackoverflow.com/questions/52703442/python-sum-values-from-multiple-lists-more-than-two
#C_loss + C_remainAGB + C_remainHWP + PH_Emissions_PO

Emissions_PF_SF_S1_mat = [c_loss_S1_mat, decomp_tot_S1_mat[:,0], subs_S1_mat, TestDSM1_mat.o, PH_Emissions_HWP1_S1_mat, OC_storage_S1_mat]


Emissions_PF_SF_S1_mat = [sum(x) for x in zip(*Emissions_PF_SF_S1_mat)]


plt.plot(t, Emissions_PF_SF_S1_mat)
plt.xlim([0, 100])
plt.ylim([0,100])


#%%


#Step (9), generate the excel/csv file for the emission and sequestration inventory

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
Col2_E_trial = Emissions_PF_SF_S1_mat
Col3 = Emissions_CH4
Col4 = flat_list_moist
Col5 = Emission_ref
Col6 = flat_list_dry


#S1_mat
df1_mat_moi = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_E_trial,'kg_CH4':Col3,'kg_CO2_seq':Col4,'emission_ref':Col5})
df1_mat_dry = pd.DataFrame.from_dict({'Year':Col1,'kg_CO2':Col2_E_trial,'kg_CH4':Col3,'kg_CO2_seq':Col6,'emission_ref':Col5})

writer = pd.ExcelWriter('emissions_seq_PF_SF_test_mat.xlsx', engine = 'xlsxwriter')

df1_mat_moi.to_excel(writer, sheet_name = 'test_mat_moist', header=True, index=False)
df1_mat_dry.to_excel(writer, sheet_name = 'test_mat_dry', header=True, index=False)


writer.save()
writer.close()

#df1.to_excel('test.xlsx', 'nuclues', header=True, index=False)


#df2.to_excel('test.xlsx', 'plasma', header=True, index=False)


#%%

## DYNAMIC LCA

# General Parameters

aCH4 = 0.129957e-12;    # methane - instantaneous radiative forcing per unit mass [W/m2 /kgCH4]
TauCH4 = 12;    # methane - lifetime (years)
aCO2 = 0.0018088e-12;    # CO2 - instantaneous radiative forcing per unit mass [W/m2 /kgCO2]
TauCO2 = [172.9,  18.51,  1.186];    # CO2 parameters according to Bern carbon cycle-climate model
aBern = [0.259, 0.338, 0.186];        # CO2 parameters according to Bern carbon cycle-climate model
a0Bern = 0.217;                     # CO2 parameters according to Bern carbon cycle-climate model
tf = 202                           #until 202 because we want to get the DCF(t-i) until DCF(201) to determine the impact from the emission from the year 200 (There is no DCF(0))


#%%

#Bern 2.5 CC Model

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

#DCF_instant for CO2 and CH4

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

#import emission data


#read test_mat_moist
df = pd.read_excel('emissions_seq_PF_SF_test_mat.xlsx', 'test_mat_moist') # can also index sheet by name or fetch all sheets
emission_CO2_S1_mat_moi = df['kg_CO2'].tolist()
emission_CH4_S1_mat_moi = df['kg_CH4'].tolist()
emission_CO2_seq_S1_mat_moi = df['kg_CO2_seq'].tolist()

emission_CO2_ref = df['emission_ref'].tolist() 


#read test_mat_dry
df = pd.read_excel('emissions_seq_PF_SF_test_mat.xlsx', 'test_mat_dry')
emission_CO2_S1_mat_dry = df['kg_CO2'].tolist()
emission_CH4_S1_mat_dry = df['kg_CH4'].tolist()
emission_CO2_seq_S1_mat_dry = df['kg_CO2_seq'].tolist()


#no_sequestration



#%%

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

#%%

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

#GWI_inst for all gases


#test_mat_moist
t = np.arange(0,tf-1,1)

matrix_GWI_S1_mat_moi = (tf-1,3)
GWI_inst_S1_mat_moi = np.zeros(matrix_GWI_S1_mat_moi)



for t in range(0,tf-1):
    GWI_inst_S1_mat_moi[t,0] = np.sum(np.multiply(emission_CO2_S1_mat_moi,DCF_CO2_ti[:,t]))
    GWI_inst_S1_mat_moi[t,1] = np.sum(np.multiply(emission_CH4_S1_mat_moi,DCF_CO2_ti[:,t]))
    GWI_inst_S1_mat_moi[t,2] = np.sum(np.multiply(emission_CO2_seq_S1_mat_moi,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S1_mat_moi = (tf-1,1)
GWI_inst_tot_S1_mat_moi = np.zeros(matrix_GWI_tot_S1_mat_moi)

GWI_inst_tot_S1_mat_moi[:,0] = np.array(GWI_inst_S1_mat_moi[:,0] + GWI_inst_S1_mat_moi[:,1] + GWI_inst_S1_mat_moi[:,2])
  
print(GWI_inst_tot_S1_mat_moi[:,0])


#test_mat_dry
t = np.arange(0,tf-1,1)

matrix_GWI_S1_mat_dry = (tf-1,3)
GWI_inst_S1_mat_dry = np.zeros(matrix_GWI_S1_mat_dry)



for t in range(0,tf-1):
    GWI_inst_S1_mat_dry[t,0] = np.sum(np.multiply(emission_CO2_S1_mat_dry,DCF_CO2_ti[:,t]))
    GWI_inst_S1_mat_dry[t,1] = np.sum(np.multiply(emission_CH4_S1_mat_dry,DCF_CO2_ti[:,t]))
    GWI_inst_S1_mat_dry[t,2] = np.sum(np.multiply(emission_CO2_seq_S1_mat_dry,DCF_CO2_ti[:,t]))

matrix_GWI_tot_S1_mat_dry = (tf-1,1)
GWI_inst_tot_S1_mat_dry = np.zeros(matrix_GWI_tot_S1_mat_dry)

GWI_inst_tot_S1_mat_dry[:,0] = np.array(GWI_inst_S1_mat_dry[:,0] + GWI_inst_S1_mat_dry[:,1] + GWI_inst_S1_mat_dry[:,2])
  
print(GWI_inst_tot_S1_mat_dry[:,0])


#no sequestration
t = np.arange(0,tf-1,1)

matrix_GWI_S1_mat_nos = (tf-1,1)
GWI_inst_S1_mat_nos = np.zeros(matrix_GWI_S1_mat_nos)



for t in range(0,tf-1):
    GWI_inst_S1_mat_nos[t,0] = np.sum(np.multiply(emission_CO2_S1_mat_dry,DCF_CO2_ti[:,t]))


matrix_GWI_tot_S1_mat_nos = (tf-1,1)
GWI_inst_tot_S1_mat_nos = np.zeros(matrix_GWI_tot_S1_mat_nos)

GWI_inst_tot_S1_mat_nos[:,0] = np.array(GWI_inst_S1_mat_nos[:,0])
  
print(GWI_inst_tot_S1_mat_nos[:,0])


t = np.arange(0,tf-1,1)


plt.plot(t, GWI_inst_tot_S1_mat_moi, color='royalblue', label='mat_moist')
plt.plot(t, GWI_inst_tot_S1_mat_dry, color='deepskyblue', label='mat_dry')
plt.plot(t, GWI_inst_tot_S1_mat_nos, color='lightcoral', label='mat_no_regrowth')

#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
#plt.grid(True)
plt.legend(loc='best', frameon=False)


plt.xlim(0,200)
plt.ylim(-0.2e-9,1.1e-9)

plt.title('Instantaneous GWI, PF_SF_test_mat')

plt.xlabel('Time (year)')
#plt.ylabel('GWI_inst (10$^{-13}$ W/m$^2$)')
plt.ylabel('GWI_inst (W/m$^2$)')

plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_inst_PF_SF_test_mat', dpi=300)

plt.show()

len(GWI_inst_tot_S1_mat_moi)

#%%

#GWI_cumulative  --> check again! try to run it with only materials case

GWI_cum_S1_mat_moi = np.cumsum(GWI_inst_tot_S1_mat_moi)
GWI_cum_S1_mat_dry = np.cumsum(GWI_inst_tot_S1_mat_dry)
GWI_cum_S1_mat_nos = np.cumsum(GWI_inst_tot_S1_mat_nos)




plt.xlabel('Time (year)')
#plt.ylabel('GWI_cum (10$^{-11}$ W/m$^2$)')
plt.ylabel('GWI_cum (W/m$^2$)')


plt.xlim(0,200)
plt.ylim(-0.1e-7,1e-7)

plt.title('Cumulative GWI, PF_SF_test_mat')

plt.plot(t, GWI_cum_S1_mat_moi, color='royalblue', label='mat_moist')
plt.plot(t, GWI_cum_S1_mat_dry, color='deepskyblue', label='mat_dry')
plt.plot(t, GWI_cum_S1_mat_nos, color='lightcoral', label='mat_no_regrowth')


plt.legend(loc='best', frameon=False)
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
#plt.grid(True)


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWI_cum_PF_SF_test_mat')

plt.show()

#%%

#determine the GWI inst for the emission reference (1 kg CO2 emission at time zero)

t = np.arange(0,tf-1,1)

matrix_GWI_ref = (tf-1,1)
GWI_inst_ref = np.zeros(matrix_GWI_S1_mat_moi)

for t in range(0,tf-1):
    GWI_inst_ref[t,0] = np.sum(np.multiply(emission_CO2_ref,DCF_CO2_ti[:,t]))

#print(GWI_inst_ref[:,0])

len(GWI_inst_ref)


#%%

#determine the GWI cumulative for the emission reference

t = np.arange(0,tf-1,1)

GWI_cum_ref = np.cumsum(GWI_inst_ref[:,0])
#print(GWI_cum_ref)

plt.xlabel('Time (year)')
plt.ylabel('GWI_cum_ref (10$^{-13}$ W/m$^2$.kgCO$_2$)')

plt.plot(t, GWI_cum_ref)



len(GWI_cum_ref)

#%%


#determine GWP dyn

GWP_dyn_cum_S1_mat_moi = [x/(y*1000) for x,y in zip(GWI_cum_S1_mat_moi, GWI_cum_ref)]
GWP_dyn_cum_S1_mat_dry = [x/(y*1000) for x,y in zip(GWI_cum_S1_mat_dry, GWI_cum_ref)]
GWP_dyn_cum_S1_mat_nos = [x/(y*1000) for x,y in zip(GWI_cum_S1_mat_nos, GWI_cum_ref)]


#print(GWP_dyn_cum_S1_mat_moi)

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)


#ax.plot(t, GWP_dyn_cum_S1moi, color='mediumseagreen', label='S1_moist')
#ax.plot(t, GWP_dyn_cum_S1dry, color='forestgreen', label='S1_dry')
#ax.plot(t, GWP_dyn_cum_S2moi, color='lightcoral', label='S2_moist')
#ax.plot(t, GWP_dyn_cum_S2dry, color='deeppink', label='S2_dry')
ax.plot(t, GWP_dyn_cum_S1_mat_moi, color='royalblue', label='mat_moist')
ax.plot(t, GWP_dyn_cum_S1_mat_dry, color='deepskyblue', label='mat_dry')
ax.plot(t, GWP_dyn_cum_S1_mat_nos, color='lightcoral', label='mat_no_regrowth')

 
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
#plt.grid(True)
plt.legend(loc='best', frameon=False)


ax.set_xlim(0,200)
ax.set_ylim(-200,800)


ax.set_xlabel('Time (year)')
ax.set_ylabel('GWP$_{dyn}$ (t-CO$_2$)')

ax.set_title('Dynamic GWP, PF_SF_test_mat')


plt.savefig('C:\Work\Data\ID Future Scenarios\Hectare-based\Fig\GWP_dyn_cum_PF_SF_test_mat', dpi=300)


plt.draw()
#%%