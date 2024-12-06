#  ******************************************************************************
#  *
#  * Copyright (c) 2013-2023, Lawrence Livermore National Security, LLC
#  * and other libROM project developers. See the top-level COPYRIGHT
#  * file for details.
#  *
#  * SPDX-License-Identifier: (Apache-2.0 OR MIT)
#  *
#  *****************************************************************************
# // Compile with: make parametric_tw_csv
# //
# // Generate CSV or HDF database on heat conduction with either
# // heat_conduction_csv.sh or heat_conduction_hdf.sh (HDF is more efficient).
# //
# // =============================================================================
# //
# // Parametric serial DMD command (for HDF version, append -hdf):
# //   python3 parametric_tw_csv.py -o hc_parametric_serial -rdim 16 -dtc 0.01 -offline
# //   python3 parametric_tw_csv.py -o hc_parametric_serial -rdim 16 -dtc 0.01 -online
# //
# // Final-time prediction error (Last line in run/hc_parametric_serial/dmd_par5_prediction_error.csv):
# //   0.0012598331433506
# //
# // Parametric time windowing DMD command (for HDF version, append -hdf):
# //   python3 parametric_tw_csv.py -o hc_parametric_tw -nwinsamp 25 -dtc 0.01 -offline
# //   python3 parametric_tw_csv.py -o hc_parametric_tw -nwinsamp 25 -dtc 0.01 -online
# //
# // Final-time prediction error (Last line in run/hc_parametric_tw/dmd_par5_prediction_error.csv):
# //   0.0006507358659606
# //
# // Parametric time windowing DMD command with custom reduced dimension size for each window (for HDF version, append -hdf):
# //   python3 parametric_tw_csv.py -o hc_parametric_tw -nwinsamp 25 -dtc 0.01 -rdim_window_file rdim_window_example.csv -offline
# //   python3 parametric_tw_csv.py -o hc_parametric_tw -nwinsamp 25 -dtc 0.01 -rdim_window_file rdim_window_example.csv -online
# //
# // =============================================================================
# //
# // Description: Parametric time windowing DMD on general CSV datasets.
# //
# // User specify file locations and names by -list LIST_DIR -train-set TRAIN_LIST -test-set TEST_LIST -data DATA_DIR -var VAR_NAME -o OUT_DIR
# //
# // File structure:
# // 1. LIST_DIR/TRAIN_LIST.csv             -- each row specifies one training DATASET
# // 2. LIST_DIR/TEST_LIST.csv              -- each row specifies one testing DATASET
# // 3. LIST_DIR/DATASET.csv                -- each row specifies the suffix of one STATE in DATASET
# // 4. DATA_DIR/dim.csv                    -- specifies the dimension of VAR_NAME
# // 5. DATA_DIR/DATASET/tval.csv           -- specifies the time instances
# // 6. DATA_DIR/DATASET/STATE/VAR_NAME.csv -- each row specifies one value of VAR_NAME of STATE
# // 7. DATA_DIR/DATASET/TEMPORAL_IDX.csv   -- (optional) specifies the first and last temporal index in DATASET
# // 8. DATA_DIR/SPATIAL_IDX.csv            -- (optional) each row specifies one spatial index of VAR_NAME
# // 9. run/OUT_DIR/indicator_val.csv       -- (optional) each row specifies one indicator endpoint value

import os
import io
import pathlib
import sys
try:
    import mfem.par as mfem
except ModuleNotFoundError:
    msg = "PyMFEM is not installed yet. Install PyMFEM:\n"
    msg += "\tgit clone https://github.com/mfem/PyMFEM.git\n"
    msg += "\tcd PyMFEM\n"
    msg += "\tpython3 setup.py install --with-parallel\n"
    raise ModuleNotFoundError(msg)

from os.path import expanduser, join, dirname
import numpy as np
from numpy import sin, cos, exp, sqrt, pi, abs, array, floor, log, sum

sys.path.append("../../build")
import pylibROM.algo as algo
import pylibROM.linalg as linalg
import pylibROM.utils as utils
from pylibROM.python_utils import StopWatch

def GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name, myid, mode):
    if (mode == 1):
        filename = "%s/%s%s_%d.hdf" % (data_dir, sim_name, par_dir, myid)
    else:
        filename = "%s/%s%s/%s_%d.hdf" % (data_dir, sim_name, par_dir, hdf_name, myid)

    print(filename)
    return filename

if __name__ == "__main__":
    infty = sys.maxsize
    precision = 16

    # 1. Initialize MPI.
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.Get_rank()
    num_procs = MPI.COMM_WORLD.Get_size()

    # 2. Parse command-line options.
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description="parametric_tw_csv")
    parser.add_argument("-offline", "--offline",
                        action='store_true', dest='offline', default=False,
                        help="Enable or disable the offline phase.")
    parser.add_argument("-no-offline", "--no-offline",
                        action='store_false', dest='offline',
                        help="Enable or disable the offline phase.")
    parser.add_argument("-online", "--online",
                        action='store_true', dest='online', default=False,
                        help="Enable or disable the online phase.")
    parser.add_argument("-no-online", "--no-online",
                        action='store_false', dest='online',
                        help="Enable or disable the online phase.")
    parser.add_argument("-predict", "--predict",
                        action='store_true', dest='predict', default=False,
                        help="Enable or disable DMD prediction.")
    parser.add_argument("-no-predict", "--no-predict",
                        action='store_false', dest='predict',
                        help="Enable or disable DMD prediction.")
    parser.add_argument("-tf", "--t-final",
                        action='store', dest='t_final', default=-1.0, type=float,
                        help="Final time.")
    parser.add_argument("-dtc", "--dtc",
                        action='store', dest='dtc', default=0.0, type=float,
                        help="Fixed (constant) dt.")
    parser.add_argument("-ddt", "--dtime-step",
                        action='store', dest='ddt', default=0.0, type=float,
                        help="Desired Time step.")
    parser.add_argument("-nwin", "--numwindows",
                        action='store', dest='numWindows', default=0, type=int,
                        help="Number of DMD windows.")
    parser.add_argument("-nwinsamp", "--numwindowsamples",
                        action='store', dest='windowNumSamples', default=infty, type=int,
                        help="Number of samples in DMD windows.")
    parser.add_argument("-nwinover", "--numwindowoverlap", dest='windowOverlapSamples',
                        action='store', default=0, type=int,
                        help="Number of samples for DMD window overlap.")
    parser.add_argument("-os", "--offset-indicator",
                        dest='offset_indicator', action='store_true', default=False,
                        help="Enable or distable the option of offset indicator.")
    parser.add_argument("-no-os", "--no-offset-indicator",
                        dest='offset_indicator', action='store_false',
                        help="Enable or distable the option of offset indicator.")
    parser.add_argument("-rbf", "--radial-basis-function", dest='rbf',
                        action='store', default='G', type=str,
                        help="Radial basis function used in interpolation. Options: \"G\", \"IQ\", \"IMQ\".")
    parser.add_argument("-interp", "--interpolation-method", dest='interp_method',
                        action='store', default='LS', type=str,
                        help="Method of interpolation. Options: \"LS\", \"IDW\", \"LP\".")
    parser.add_argument("-acrv", "--admd-crv", dest='admd_closest_rbf_val',
                        action='store', default=0.9, type=float,
                        help="Adaptive DMD closest RBF value.")
    parser.add_argument("-pcrv", "--pdmd-crv", dest='pdmd_closest_rbf_val',
                        action='store', default=0.9, type=float,
                        help="Parametric DMD closest RBF value.")
    parser.add_argument("-ef", "--energy-fraction", dest='ef',
                        action='store', default=0.9999, type=float,
                        help="Energy fraction for DMD.")
    parser.add_argument("-rdim", "--rdim", dest='rdim',
                        action='store', default=-1, type=int,
                        help="Reduced dimension for DMD.")
    parser.add_argument("-list", "--list-directory", dest='list_dir',
                        action='store', default='dmd_list', type=str,
                        help="Location of training and testing data list.")
    parser.add_argument("-data", "--data-directory", dest='data_dir',
                        action='store', default='dmd_data', type=str,
                        help="Location of training and testing data.")
    parser.add_argument("-hdffile", "--hdf-file", dest='hdf_name',
                        action='store', default='dmd', type=str,
                        help="Base of name of HDF file for training and testing data.")
    parser.add_argument("-sim", "--sim-name", dest='sim_name',
                        action='store', default='sim', type=str,
                        help="Name of simulation.")
    parser.add_argument("-var", "--variable-name", dest='var_name',
                        action='store', default='sol', type=str,
                        help="Name of variable.")
    parser.add_argument("-train-set", "--training-set-name", dest='train_list',
                        action='store', default='dmd_train_parametric', type=str,
                        help="Name of the training datasets within the list directory.")
    parser.add_argument("-test-set", "--testing-set-name", dest='test_list',
                        action='store', default='dmd_test', type=str,
                        help="Name of the testing datasets within the list directory.")
    parser.add_argument("-t-idx", "--temporal-index", dest='temporal_idx_list',
                        action='store', default='temporal_idx', type=str,
                        help="Name of the file indicating bound of temporal indices.")
    parser.add_argument("-x-idx", "--spatial-index", dest='spatial_idx_list',
                        action='store', default='spatial_idx', type=str,
                        help="Name of the file indicating spatial indices.")
    parser.add_argument("-snap-pfx", "--snapshot-prefix", dest='snap_pfx',
                        action='store', default='step', type=str,
                        help="Prefix of snapshots.")
    parser.add_argument("-o", "--outputfile-name", dest='basename',
                        action='store', default='', type=str,
                        help="Name of the sub-folder to dump files within the run directory.")
    parser.add_argument("-save", "--save", dest='save_csv',
                        action='store_true', default=False,
                        help="Enable or disable prediction result output (files in CSV format).")
    parser.add_argument("-no-save", "--no-save", dest='save_csv',
                        action='store_false',
                        help="Enable or disable prediction result output (files in CSV format).")
    parser.add_argument("-save-hdf", "--save-hdf", dest='save_hdf',
                        action='store_true', default=False,
                        help="Enable or disable prediction result output (files in HDF format).")
    parser.add_argument("-no-save-hdf", "--no-save-hdf", dest='save_hdf',
                        action='store_false',
                        help="Enable or disable prediction result output (files in HDF format).")
    parser.add_argument("-csv", "--csv", dest='csvFormat',
                        action='store_true', default=True,
                        help="Use CSV or HDF format for input files.")
    parser.add_argument("-hdf", "--hdf", dest='csvFormat',
                        action='store_false',
                        help="Use CSV or HDF format for input files.")
    parser.add_argument("-wdim", "--wdim", dest='useWindowDims',
                        action='store_true', default=False,
                        help="Use DMD dimensions for each window, input from a CSV file.")
    parser.add_argument("-no-wdim", "--no-wdim", dest='useWindowDims',
                        action='store_false',
                        help="Use DMD dimensions for each window, input from a CSV file.")
    parser.add_argument("-subs", "--subsample", dest='subsample',
                        action='store', default=0, type=int,
                        help="Subsampling factor for training snapshots.")
    parser.add_argument("-esubs", "--eval_subsample", dest='eval_subsample',
                        action='store', default=0, type=int,
                        help="Subsampling factor for evaluation.")
    parser.add_argument("-hdfmode", "--hdfmodefilename", dest='fileNameMode',
                        action='store', default=0, type=int,
                        help="HDF filename mode.")
    parser.add_argument("-rdim_window_file", "--rdim_window_filename", dest='RDIMWindowfileName',
                        action='store', default='', type=str,
                        help="CSV file storing rdim for each window.")
    
    args = parser.parse_args()
    if (myid == 0):
        parser.print_options(args)

    offline                 = args.offline
    online                  = args.online
    predict                 = args.predict
    t_final                 = args.t_final
    dtc                     = args.dtc
    ddt                     = args.ddt
    numWindows              = args.numWindows
    windowNumSamples        = args.windowNumSamples
    windowOverlapSamples    = args.windowOverlapSamples
    offset_indicator        = args.offset_indicator
    rbf                     = args.rbf
    interp_method           = args.interp_method
    admd_closest_rbf_val    = args.admd_closest_rbf_val
    pdmd_closest_rbf_val    = args.pdmd_closest_rbf_val
    ef                      = args.ef
    rdim                    = args.rdim
    list_dir                = args.list_dir
    data_dir                = args.data_dir
    sim_name                = args.sim_name
    var_name                = args.var_name
    train_list              = args.train_list
    test_list               = args.test_list
    temporal_idx_list       = args.temporal_idx_list
    spatial_idx_list        = args.spatial_idx_list
    hdf_name                = args.hdf_name
    snap_pfx                = args.snap_pfx
    basename                = args.basename
    save_csv                = args.save_csv
    save_hdf                = args.save_hdf
    csvFormat               = args.csvFormat
    useWindowDims           = args.useWindowDims
    subsample               = args.subsample
    eval_subsample          = args.eval_subsample
    fileNameMode            = args.fileNameMode
    RDIMWindowfileName      = args.RDIMWindowfileName

    assert((not (offline and online)) and (offline or online))
    assert(not ((dtc > 0.0) and (ddt > 0.0)))
    dt_est = max(ddt, dtc)

    if (t_final > 0.0):
        save_csv = True

    outputPath = "run"
    if (basename != ""):
        outputPath += "/" + basename

    if (myid == 0):
        pathlib.Path(outputPath).mkdir(parents=True, exist_ok=True)

    csv_db = utils.CSVDatabase()
    prefix = ""
    suffix = ""
    if (csvFormat):
        db = utils.CSVDatabase()
        suffix = ".csv"
    else:
        db = utils.HDFDatabase()

    if (save_hdf):
        hdf_db = utils.HDFDatabase()
        hdf_db.create("prediction" + (myid) + ".hdf")

    variable = var_name
    nelements = 0

    if (csvFormat):
        nelements = db.getIntegerArray(data_dir + "/dim.csv", 1)[0]
    else:
        db.open(GetFilenameHDF(data_dir, sim_name, "0", hdf_name,
                                myid, fileNameMode), "r")
        nelements = db.getDoubleArraySize("step0sol")
        db.close()

    assert(nelements > 0)
    if (myid == 0):
        print("Variable %s has dimension %d." % (var_name, nelements))

    dim = nelements
    idx_state = csv_db.getIntegerVector("%s/%s.csv" % (data_dir, spatial_idx_list), False)
    if (idx_state.size > 0):
        dim = idx_state.size
        if (myid == 0):
            print("Restricting on %d entries out of %d." % (dim, nelements))

    indicator_val   = []
    if (online or (numWindows > 0)):
        indicator_val = csv_db.getDoubleVector("%s/indicator_val.csv" % outputPath, False)
        if (indicator_val.size > 0):
            if (numWindows > 0):
                assert(numWindows == indicator_val.size)
            else:
                numWindows = indicator_val.size
            if (myid == 0):
                print("Read indicator range partition with %d windows." % numWindows)

    par_dir_list    = []
    par_vectors     = []
    indicator_init  = []
    indicator_last  = []

    #######
    #load window list here
    #######
    use_rdim_windows=False
    window_dim_list=[]
    if RDIMWindowfileName != "":
        if not RDIMWindowfileName.endswith(".csv"):
            RDIMWindowfileName += ".csv"
        window_dim_list = csv_db.getStringVector((list_dir) + "/" + RDIMWindowfileName, False)
        if window_dim_list[0] == 'RDIM':
            use_rdim_windows=True
        elif window_dim_list[0] == 'EF':
            use_rdim_windows=False
        else:
            raise RuntimeError("Expected RDIM or EF as first line in window file")
        for window in range(len(window_dim_list)):
            window_dim_list[window] = window_dim_list[window].split(',')
        window_dim_list = window_dim_list[1:] # chop header from list
        print("Read window file with {} windows:".format(len(window_dim_list)))
        for window in window_dim_list:
            print("  Window {}: {} = {}".format(window[0], "RDIM" if use_rdim_windows else "EF", window[1]))

    training_par_list = csv_db.getStringVector((list_dir) + "/" + train_list + ".csv", False)
    npar = len(training_par_list)
    assert(npar > 0)
    if (myid == 0):
        print("Loading %d training datasets." % npar)

    dpar = -1
    for idx_dataset in range(npar):
        par_info = training_par_list[idx_dataset].split(',') # training DATASET

        dpar = len(par_info) - 1
        curr_par = linalg.Vector(dpar, False)

        if (idx_dataset == 0):
            assert(dpar > 0)
            if (myid == 0):
                print("Dimension of parametric space = %d." % dpar)
        else:
            assert(dpar == len(par_info) - 1)

        par_dir = par_info[0]
        par_dir_list += [par_dir]

        if (not csvFormat):
            db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                    myid, fileNameMode), "r")

        snap_bound_size = 0
        if (csvFormat):
            prefix = (data_dir) + "/" + par_dir + "/"
            snap_bound_size = csv_db.getLineCount(prefix + temporal_idx_list + suffix)
        else:
            db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                    myid, fileNameMode), "r")
            snap_bound_size = db.getInteger("snap_bound_size")

        num_snap_orig = 0
        if (csvFormat):
            num_snap_orig = csv_db.getLineCount((list_dir) + "/" + par_dir + ".csv")
        else:
            db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                    myid, fileNameMode), "r")
            num_snap_orig = db.getInteger("numsnap")

        assert(num_snap_orig > 0)
        if (csvFormat):
            snap_list = csv_db.getStringVector((list_dir) + "/" + par_dir + ".csv", False)
        else:
            snap_index_list = db.getIntegerArray("snap_list", num_snap_orig)

        if (csvFormat): prefix = (data_dir) + "/" + par_dir + "/"
        snap_bound = db.getIntegerArray(prefix + temporal_idx_list + suffix, snap_bound_size)
        if (snap_bound.size > 0):
            snap_bound[0] -= 1
            snap_bound[1] -= 1
            if (myid == 0):
                print("Restricting on snapshot #%d to %d." % (snap_bound[0], snap_bound[1]))
        else:
            snap_bound = np.array([0, num_snap_orig - 1])

        for par_order in range(dpar):
            curr_par[par_order] = float(par_info[par_order+1])
        par_vectors += [curr_par]

        if (offline):
            if (csvFormat): prefix = (data_dir) + "/" + par_dir + "/"
            tvec = db.getDoubleArray(prefix + "tval" + suffix, num_snap_orig)

            if (offset_indicator):
                indicator_init += [0.0]
                indicator_last += [tvec[snap_bound[1]] - tvec[snap_bound[0]]]
            else:
                indicator_init += [tvec[snap_bound[0]]]
                indicator_last += [tvec[snap_bound[1]]]

        db.close()

    assert(windowOverlapSamples < windowNumSamples)
    if (offline and (len(indicator_val) == 0)):
        indicator_min = min(indicator_init)
        indicator_max = max(indicator_last)
        numWindows = round((indicator_max - indicator_min) / (dt_est * windowNumSamples)) if (windowNumSamples < infty) else 1
        for window in range(numWindows):
            indicator_val += [indicator_min + dt_est * windowNumSamples * window]

        if (myid == 0):
            print("Created new indicator range partition with %d windows." % numWindows)
            csv_db.putDoubleVector("%s/indicator_val.csv" % outputPath,
                                       indicator_val, numWindows)

    assert(numWindows > 0)
    if (myid == 0):
        if (numWindows > 1):
            print("Using time windowing DMD with %d windows." % numWindows)
        else:
            print("Using serial DMD.")

    # vector<int> windowDim;
    if (useWindowDims):
        windowDim = csv_db.getIntegerVector("run/maxDim.csv")
        assert(windowDim.size == numWindows)

    dmd_training_timer, dmd_preprocess_timer, dmd_prediction_timer = StopWatch(), StopWatch(), StopWatch()
    dmd = []
    # vector<CAROM::DMD*> dmd_curr_par;
    # double* sample = new double[dim];

    if (offline):
        dmd_training_timer.Start()

        maxDim = [0] * numWindows
        minSamp = [-1] * numWindows

        for idx_dataset in range(npar):
            par_dir = par_dir_list[idx_dataset]
            if (myid == 0):
                print("Loading samples for %s to train DMD." % par_dir)

            dmd_curr_par = [None] * numWindows
            for window in range(numWindows):
                if (ddt > 0.0):
                    dmd_curr_par[window] = algo.AdaptiveDMD(dim, ddt, rbf,
                                                interp_method, admd_closest_rbf_val)
                elif (dtc > 0.0):
                    dmd_curr_par[window] = algo.DMD(dim, dtc, True)
                else:
                    dmd_curr_par[window] = algo.NonuniformDMD(dim)

            dmd += [dmd_curr_par]

            num_snap_orig = 0
            if (csvFormat):
                num_snap_orig = csv_db.getLineCount((list_dir) + "/" + par_dir + ".csv")
            else:
                db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                        myid, fileNameMode), "r")
                num_snap_orig = db.getInteger("numsnap")

            assert(num_snap_orig > 0)
            # vector<string> snap_list(num_snap_orig);
            # vector<int> snap_index_list(num_snap_orig);
            if (csvFormat):
                snap_list = csv_db.getStringVector((list_dir) + "/" + par_dir + ".csv", False)
            else:
                snap_index_list = db.getIntegerArray("snap_list", num_snap_orig)

            # vector<double> tvec(num_snap_orig);
            tvec = db.getDoubleArray(prefix + "tval" + suffix, num_snap_orig)

            snap_bound_size = 0
            if (csvFormat):
                prefix = (data_dir) + "/" + par_dir + "/"
                snap_bound_size = csv_db.getLineCount(prefix + temporal_idx_list + suffix)
            else:
                db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                        myid, fileNameMode), "r")
                snap_bound_size = db.getInteger("snap_bound_size")

            # vector<int> snap_bound(snap_bound_size);
            snap_bound = db.getIntegerArray(prefix + temporal_idx_list + suffix, snap_bound_size)

            if (snap_bound.size > 0):
                snap_bound[0] -= 1
                snap_bound[1] -= 1
                assert(snap_bound.size == 2)
                if (myid == 0):
                    print("Restricting on snapshot #%d to %d." % (snap_bound[0], snap_bound[1]))
            else:
                snap_bound = np.array([0, num_snap_orig - 1])

            curr_window = 0
            overlap_count = 0
            for idx_snap in range(snap_bound[0], snap_bound[1] + 1):
                if ((subsample > 1) and (idx_snap % subsample != 0)
                    and (idx_snap > snap_bound[0]) and (idx_snap < snap_bound[1])):
                    continue

                snap = snap_pfx
                tval = tvec[idx_snap]
                if (csvFormat):
                    snap += snap_list[idx_snap] # STATE
                    data_filename = (data_dir) + "/" + par_dir + "/" + snap + "/" + variable + ".csv" # path to VAR_NAME.csv
                    sample = db.getDoubleArray(data_filename, nelements, idx_state)
                else:
                    snap += str(snap_index_list[idx_snap]) # STATE
                    sample = db.getDoubleArray(snap + "sol", nelements, idx_state)

                dmd[idx_dataset][curr_window].takeSample(sample,
                        tval - offset_indicator * tvec[snap_bound[0]])
                
                if (overlap_count > 0):
                    dmd[idx_dataset][curr_window-1].takeSample(sample,
                        tval - offset_indicator * tvec[snap_bound[0]])
                    overlap_count -= 1

                # a rough estimate to correct the precision of the indicator range partition
                indicator_snap = tval - offset_indicator * tvec[snap_bound[0]] + dt_est / 100.0

                if ((curr_window+1 < numWindows) and (idx_snap+1 <= snap_bound[1])
                    and (indicator_snap > indicator_val[curr_window+1])):
                    overlap_count = windowOverlapSamples
                    curr_window += 1
                    dmd[idx_dataset][curr_window].takeSample(sample,
                        tval - offset_indicator * tvec[snap_bound[0]])

            if (myid == 0):
                print("Loaded %d samples for %s." % (snap_bound[1] - snap_bound[0] + 1, par_dir))

            if len(window_dim_list) > 0:
                # check that window samples is consistent with number of windows in the csv file
                tot_samples=snap_bound[1] - snap_bound[0]
                assert((len(window_dim_list)*windowNumSamples) == tot_samples), "Mismatch between -nwinsamp, number of windows in -rdim_window_file, and total number of samples"

            for window in range(numWindows):
                if (useWindowDims):
                    rdim = windowDim[window]

                if len(window_dim_list) > 0:
                    if use_rdim_windows:
                        rdim = int(window_dim_list[window][1])
                    else:
                        ef = float(window_dim_list[window][1])

                if (rdim != -1):
                    if (myid == 0):
                        print("Creating DMD model #%d with rdim: %d" % (window, rdim))
                    dmd[idx_dataset][window].train(rdim)
                elif (ef > 0.0):
                    if (myid == 0):
                        print("Creating DMD model #%d with energy fraction: %f" % (window, ef))
                    dmd[idx_dataset][window].train(ef)

                if ((window > 0) and predict):
                    if (myid == 0):
                        print("Projecting initial condition at t = %f for DMD model #%d" % (indicator_val[window] + offset_indicator * tvec[snap_bound[0]],
                                                                                            window))
                    init_cond = dmd[idx_dataset][window-1].predict(indicator_val[window])
                    dmd[idx_dataset][window].projectInitialCondition(init_cond)
                    del init_cond

                # Make a directory for this window, only on the first parameter.
                if (idx_dataset == 0):
                    outWindow = outputPath + "/window" + str(window)
                    pathlib.Path(outWindow).mkdir(parents=True, exist_ok=True)

                dmd[idx_dataset][window].save(outputPath + "/window"
                                               + str(window) + "/par"
                                               + str(idx_dataset))

                if (myid == 0):
                    dmd[idx_dataset][window].summary(outputPath + "/window"
                                                      + str(window) + "/par"
                                                      + str(idx_dataset))

                    print("Window %d, DMD %d dim %d" % (window, idx_dataset,
                                                        dmd[idx_dataset][window].getDimension()))

                dim_w = min(dmd[idx_dataset][window].getDimension(),
                            dmd[idx_dataset][window].getNumSamples()-1)
                maxDim[window] = max(maxDim[window], dim_w)

                if ((minSamp[window] < 0) or
                    (dmd[idx_dataset][window].getNumSamples() < minSamp[window])):
                    minSamp[window] = dmd[idx_dataset][window].getNumSamples()
            # escape for-loop over window

            db.close()

            if ((not online) and (not predict)):
                dmd[idx_dataset] = [None] * numWindows
        # escape for-loop over idx_dataset
        dmd_training_timer.Stop()

        # Limit maxDim by minSamp-1
        for window in range(numWindows):
            maxDim[window] = min(maxDim[window], minSamp[window]-1)

        # Write maxDim to a CSV file
        if (myid == 0): csv_db.putIntegerArray("run/maxDim.csv", maxDim, numWindows)
    # escape if-statement of offline

    curr_par = linalg.Vector(dpar, False)

    if (online):
        par_dir_list = []

        dmd_preprocess_timer.Start()
        testing_par_list = csv_db.getStringVector((list_dir) + "/" + test_list + ".csv", False)
        npar = len(testing_par_list)
        assert(npar > 0)

        if (myid == 0):
            print("Loading %d testing datasets." % npar)

        dmd_curr_par = [None] * numWindows
        dmd = [dmd_curr_par] * numWindows

        num_tests = 0
        # vector<double> prediction_time, prediction_error;

        for idx_dataset in range(npar):
            par_info = testing_par_list[idx_dataset].split(',') # testing DATASET
            assert(dpar == len(par_info) - 1)

            par_dir = par_info[0]
            par_dir_list += [par_dir]
            if (myid == 0):
                print("Interpolating DMD models for dataset %s" % par_dir)

            num_snap_orig = 0
            if (csvFormat):
                num_snap_orig = csv_db.getLineCount((list_dir) + "/" + par_dir + ".csv")
            else:
                db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                        myid, fileNameMode), "r")
                num_snap_orig = db.getInteger("numsnap")

            assert(num_snap_orig > 0)
            # vector<string> snap_list(num_snap_orig);
            # vector<int> snap_index_list(num_snap_orig);
            if (csvFormat):
                snap_list = csv_db.getStringVector((list_dir) + "/" + par_dir + ".csv", False)
            else:
                snap_index_list = db.getIntegerArray("snap_list", num_snap_orig)

            for par_order in range(dpar):
                curr_par[par_order] = float(par_info[par_order+1])

            tvec = db.getDoubleArray(prefix + "tval" + suffix, num_snap_orig)

            snap_bound_size = 0
            if (csvFormat):
                prefix = (data_dir) + "/" + par_dir + "/"
                snap_bound_size = csv_db.getLineCount(prefix + temporal_idx_list + suffix)
            else:
                db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                        myid, fileNameMode), "r")
                snap_bound_size = db.getInteger("snap_bound_size")

            snap_bound = db.getIntegerArray(prefix + temporal_idx_list + suffix, snap_bound_size)

            if (snap_bound.size > 0):
                snap_bound[0] -= 1
                snap_bound[1] -= 1
                assert(snap_bound.size == 2)
                if (myid == 0):
                    print("Restricting on snapshot #%d to %d." % (snap_bound[0], snap_bound[1]))
            else:
                snap_bound = np.array([0, num_snap_orig - 1])

            snap = snap_pfx
            if (csvFormat):
                snap += snap_list[snap_bound[0]]; # STATE
                data_filename = (data_dir) + "/" + par_dir + "/" + snap + "/" + variable + ".csv" # path to VAR_NAME.csv
                sample = db.getDoubleArray(data_filename, nelements, idx_state)
                print("sample: ", sample)
                print("sample size: ", sample.size)
            else:
                snap += str(snap_index_list[snap_bound[0]]) # STATE
                sample = db.getDoubleArray(snap + "sol", nelements, idx_state)

            for window in range(numWindows):
                dmd_paths = []
                for idx_trainset in range(len(par_vectors)):
                    dmd_paths += [outputPath + "/window" + str(window) + "/par" + str(idx_trainset)]

                if (len(par_vectors) > 1):
                    if (myid == 0):
                        print("Interpolating DMD model #%d" % window)

                    dmd[idx_dataset][window] = algo.getParametricDMD(algo.DMD, par_vectors,
                                                                     dmd_paths, curr_par, (rbf),
                                                                     (interp_method), pdmd_closest_rbf_val)
                elif ((len(par_vectors) == 1) and (dmd_paths.size() == 1)):
                    if (myid == 0):
                        print("Loading local DMD model #%d" % window)
                    dmd[idx_dataset][window] = algo.DMD(dmd_paths[0])

                if (myid == 0):
                    print("Projecting initial condition at t = %f for DMD model #%d"
                          % (indicator_val[window] + offset_indicator * tvec[snap_bound[0]], window))

                if (window == 0):
                    init_cond = linalg.Vector(dim, True)
                    for i in range(dim):
                        init_cond[i] = sample[i]
                else:
                    init_cond = dmd[idx_dataset][window-1].predict(indicator_val[window])
                
                dmd[idx_dataset][window].projectInitialCondition(init_cond)

                norm_init_cond = init_cond.norm()
                if (myid == 0):
                    print("Initial condition norm %.6e for parameter %d, window %d"
                          % (norm_init_cond, idx_dataset, window - 1))

                del init_cond

                if ((window > 0) and (indicator_val[window] < t_final)):
                    # To save memory, delete dmd[idx_dataset][window] for windows
                    # not containing t_final.
                    if (myid == 0):
                        print("Deleting DMD for parameter %d, window %d"
                              % (idx_dataset, window - 1))

                    del dmd[idx_dataset][window-1]
                    dmd[idx_dataset][window-1] = None
            # escape for-loop over window

            db.close()
        # escape for-loop over idx_dataset
        dmd_preprocess_timer.Stop()
    # escape if-statement of online

    if (online or predict):
        num_tests = 0
        prediction_time, prediction_error = [], []

        for idx_dataset in range(npar):
            par_dir = par_dir_list[idx_dataset]
            if (myid == 0):
                print("Predicting solution for %s using DMD." % par_dir)

            num_snap_orig = 0
            if (csvFormat):
                num_snap_orig = csv_db.getLineCount("%s/%s.csv" % (list_dir, par_dir))
            else:
                db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                        myid, fileNameMode), "r")
                num_snap_orig = db.getInteger("numsnap")

            assert(num_snap_orig > 0)
            if (csvFormat):
                snap_list = csv_db.getStringVector("%s/%s.csv" % (list_dir, par_dir), False)
            else:
                snap_index_list = db.getIntegerArray("snap_list", num_snap_orig)

            tvec = db.getDoubleArray(prefix + "tval" + suffix, num_snap_orig)

            snap_bound_size = 0
            if (csvFormat):
                prefix = "%s/%s/" % (data_dir, par_dir)
                snap_bound_size = csv_db.getLineCount(prefix + temporal_idx_list + suffix)
            else:
                db.open(GetFilenameHDF(data_dir, sim_name, par_dir, hdf_name,
                                        myid, fileNameMode), "r")
                snap_bound_size = db.getInteger("snap_bound_size")

            snap_bound = db.getIntegerArray(prefix + temporal_idx_list + suffix, snap_bound_size)

            if (snap_bound_size > 0):
                snap_bound[0] -= 1
                snap_bound[1] -= 1
                assert(snap_bound.size == 2)
                if (myid == 0):
                    print("Restricting on snapshot #%d to %d." % (snap_bound[0], snap_bound[1]))
            else:
                snap_bound = np.array([0, num_snap_orig - 1])

            num_snap = snap_bound[1] - snap_bound[0] + 1
            curr_window = 0
            for idx_snap in range(snap_bound[0], snap_bound[1] + 1):
                if ((eval_subsample > 1) and (idx_snap % eval_subsample != 0)
                    and (idx_snap > snap_bound[0]) and (idx_snap < snap_bound[1])):
                    continue

                snap = snap_pfx
                tval = tvec[idx_snap]
                if (csvFormat):
                    snap += snap_list[idx_snap]; # STATE
                    data_filename = "%s/%s/%s/%s.csv" % (data_dir, par_dir, snap, variable) # path to VAR_NAME.csv
                    sample = db.getDoubleArray(data_filename, nelements, idx_state)
                else:
                    snap += "%d" % (snap_index_list[idx_snap]) # STATE
                    sample = db.getDoubleArray(snap + "sol", nelements, idx_state)

                if (myid == 0):
                    print("State %s read." % snap)

                if (t_final > 0.0): # Actual prediction without true solution for comparison
                    num_tests += 1
                    while ((curr_window+1 < numWindows) and
                            (t_final - offset_indicator * tvec[snap_bound[0]] >
                            indicator_val[curr_window+1])):
                        curr_window += 1

                    if (myid == 0):
                        print("Predicting DMD solution at t = %f using DMD model #%d" % (t_final, curr_window))

                    dmd_prediction_timer.Start()
                    result = dmd[idx_dataset][curr_window].predict(t_final - offset_indicator * tvec[snap_bound[0]])
                    dmd_prediction_timer.Stop()

                    if (myid == 0):
                        csv_db.putDoubleArray(outputPath + "/" + par_dir +
                                            "_final_time_prediction.csv",
                                            result.getData(), dim)

                    if (save_hdf):
                        hdf_db.putDoubleArray(snap, result.getData(), dim)

                    idx_snap = snap_bound[1]+1 # escape for-loop over idx_snap
                    del result
                else: # Verify DMD prediction results against dataset
                    while ((curr_window+1 < numWindows) and
                           (tval - offset_indicator * tvec[snap_bound[0]] >
                            indicator_val[curr_window+1])):
                        curr_window += 1

                    if (myid == 0):
                        print("Predicting DMD solution #%d at t = %f using DMD model #%d" % (idx_snap, tval, curr_window))

                    dmd_prediction_timer.Start()
                    result = dmd[idx_dataset][curr_window].predict(tval - offset_indicator * tvec[snap_bound[0]])
                    dmd_prediction_timer.Stop()

                    # Calculate the relative error between the DMD final solution and the true solution.
                    dmd_solution = mfem.Vector(result.getData(), result.dim())
                    true_solution = mfem.Vector(sample, dim)
                    diff = mfem.Vector(true_solution.Size())
                    mfem.subtract_vector(dmd_solution, true_solution, diff)

                    tot_diff_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, diff, diff))
                    tot_true_solution_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, true_solution,
                                                        true_solution))
                    rel_error = tot_diff_norm / tot_true_solution_norm

                    prediction_time += [tval]
                    prediction_error += [rel_error]

                    if (myid == 0):
                        print("Norm of true solution at t = %f is %f" % (tval, tot_true_solution_norm))
                        print("Absolute error of DMD solution at t = %f is %f" % (tval, tot_diff_norm))
                        print("Relative error of DMD solution at t = %f is %f" % (tval, rel_error))

                        if (idx_snap == snap_bound[1]):  # Final step
                            with open("log.txt", "a") as f:
                                np.savetxt(f, [windowNumSamples, subsample, rel_error])

                        if (save_csv):
                            csv_db.putDoubleArray("%s/%s_%s_prediction.csv" % (outputPath, par_dir, snap),
                                                  result.getData(), dim)
                        if (dim < nelements):
                            csv_db.putDoubleArray("%s/%s_%s_state.csv" % (outputPath, par_dir, snap),
                                                  sample, dim)

                    if (save_hdf):
                        hdf_db.putDoubleArray(snap, result.getData(), dim)

                    del result

            if ((myid == 0) and (t_final <= 0.0)):
                csv_db.putDoubleVector(outputPath + "/" + par_dir + "_prediction_time.csv",
                                    prediction_time, num_snap)
                csv_db.putDoubleVector(outputPath + "/" + par_dir + "_prediction_error.csv",
                                    prediction_error, num_snap)
                
            # prediction_time.clear();
            # prediction_error.clear();
            num_tests = num_tests + 1 if (t_final > 0.0) else num_tests + num_snap

            db.close()

        assert(num_tests > 0)

        if (myid == 0):
            print("Elapsed time for training DMD: %e second\n" % dmd_training_timer.duration)
            print("Elapsed time for preprocessing DMD: %e second\n" % dmd_preprocess_timer.duration)
            print("Total elapsed time for predicting DMD: %e second\n" % dmd_prediction_timer.duration)
            print("Average elapsed time for predicting DMD: %e second\n" % (dmd_prediction_timer.duration / num_tests))
    # escape case (online || predict)

    if (save_hdf):
        hdf_db.close()
        del hdf_db

    del sample
    del curr_par
    del dmd
