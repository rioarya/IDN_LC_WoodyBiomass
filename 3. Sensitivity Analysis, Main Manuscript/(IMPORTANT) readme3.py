This folder is dedicated to the Python script to run the sensitivity analysis for the improved harvesting conversion efficiency scenarios (results shown in the main manuscript)

- For the product lifetime sensitivity analysis, the users can simply change the product lifetime in Step (4) in the script (Dynamic stock model) 
in the 'Main Results in the Manuscript' folder. See Table A.20 in Appendix A for more information on the product lifetime sensitivity analysis. 
- Please check further guideline/workflow in Appendix A, Table A.22.
- Each python script accompanied by the corresponding Excel Files. Some parts of the script will read the data in Excel files.
- The flow names in Excel files may not so intuitive. I invite the readers to check the Appendix C for a more tidy version of the 
yearly material flow tables. There, you can also find out the corresponding flow codes that can be linked with the system definition (Fig. 1)
- Some scenarios like PF_PO, PF_FP and DL_FP has counter use material flow in separate excel files (name started with 'NonRW...')
- Please replace 'C:\\my_path\\scenario_files.xlsx' with your own path/directory. 