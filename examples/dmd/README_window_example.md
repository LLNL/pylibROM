#Steps for running parametric_tw_csv_rdim_window.py

#generate heat conduction data via:
    sh heat_conducrtion_csv.sh
#edit rdim_window_example.csv file to user's liking; each row contains the window number and corresponding reduced basis size for the given window
#copy rdim_window_example.csv to the dmd_list directory
#run parametric_tw_csv_rdim_window.py via:
    python3 parametric_tw_csv_rdim_window.py -o hc_parametric_tw -nwinsamp 25 -dtc 0.01 -rdim_window_file rdim_window_example.csv -offline
    python3 parametric_tw_csv_rdim_window.py -o hc_parametric_tw -nwinsamp 25 -dtc 0.01 -rdim_window_file rdim_window_example.csv -online
#NOTE: -nwinsamp must be set such that the number of windows generated is the same as the number of windows specified in -rdim_window_file for consistency