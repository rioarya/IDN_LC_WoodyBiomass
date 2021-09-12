This folder is dedicated to the Python script to run the sensitivity analysis for the existing harvesting conversion efficiency scenarios (results shown in Appendix). 
In addition, in this folder we also have the Python script (named with '...dim' for testing the effects of diminishing foregone carbon sequestration
of primary forest for counter-use (NR) scenarios (results also shown in Appendix).

- For the product lifetime sensitivity analysis, the users can simply change the product lifetime in Step (4) in the script (Dynamic stock model) in the 'EC (Existing Conversion Efficiency), Appendix' folder. See Table 20 in Appendix 1
for more information on the product lifetime sensitivity analysis. 
- No DL_FP for the case of testing the effects of diminishing foregone sequestration as no primary forest was involved in this scenario  
- Each python script accompanied by the corresponding Excel Files. Some parts of the script will read the data in Excel files.
- The flow names in Excel files may not so intuitive. I invite the readers to check the Appendix C for a more tidy version of the 
yearly material flow tables. There, you can also find out the corresponding flow codes that can be linked with the system definition (Fig. 1)
- Some scenarios like PF_PO, PF_FP and DL_FP has counter use material flow in separate excel files (name started with 'NonRW...')
- Some steps/modules in the Python 3 are unused (marked with #)