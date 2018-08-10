SUMMER SOLAR REU PROGRAMS - SXM FLIGHT SPARE ANGULAR RESPONSE
AUTHOR: Christian Nieves

----------------------------
Python Scripts
-----------------------------
1) parse-h5_tlm-log_files
    - Used as a terminal command
        parse-h5_tlm-log_files [directory containing .tlm.lo files] [directory that will contain all the product files] 
    - Copies '.tlm.log' files to the destination folder
    - Creates a folder for each file
    - Creates a folder for .h5 files
    - Uses rex_raw_parser and rex_hdf5_gen to generate '.h5' files and saves them in their corresponding directories
    
2) flight_spare_data_plots.py
    - Contains all the functions for the analysis of data taken form the flight spare measurements
    
----------------------------
Folders
-----------------------------
1) plotFigs
    - All the figures plotted with the flight spare data

----------------------------
Text files
-----------------------------
1) data_table.txt
    - Contain all the filenames along with the azimuth and elevation angles for each measurement