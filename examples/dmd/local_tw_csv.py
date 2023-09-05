# /******************************************************************************
#  *
#  * Copyright (c) 2013-2023, Lawrence Livermore National Security, LLC
#  * and other libROM project developers. See the top-level COPYRIGHT
#  * file for details.
#  *
#  * SPDX-License-Identifier: (Apache-2.0 OR MIT)
#  *
#  *****************************************************************************/

# // Compile with: make local_tw_csv
# //
# // Generate CSV or HDF database on heat conduction with either
# // heat_conduction_csv.sh or heat_conduction_hdf.sh (HDF is more efficient).
# //
# // =================================================================================
# //
# // Local serial DMD command for CSV or HDF:
# //   mpirun -np 8 local_tw_csv -o hc_local_serial -rdim 16 -dtc 0.01 -csv
# //   mpirun -np 8 local_tw_csv -o hc_local_serial -rdim 16 -dtc 0.01 -hdf
# //
# // Final-time prediction error (last line in run/hc_local_serial/dmd_par5_prediction_error.csv):
# //   0.0004063242226265
# //
# // Local time windowing DMD command for CSV or HDF:
# //   mpirun -np 8 local_tw_csv -o hc_local_tw -rdim 16 -nwinsamp 25 -dtc 0.01 -csv
# //   mpirun -np 8 local_tw_csv -o hc_local_tw -nwinsamp 25 -dtc 0.01 -hdf
# //
# // Final-time prediction error (last line in run/hc_local_tw/dmd_par5_prediction_error.csv):
# //   0.0002458808673544
# //
# // =================================================================================
# //
# // Description: Local time windowing DMD on general CSV datasets.
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

if __name__ == "__main__":
    infty = sys.maxsize
    precision = 16

    # 1. Initialize MPI.
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.Get_rank()
    num_procs = MPI.COMM_WORLD.Get_size()

    # 2. Parse command-line options.
    from mfem.common.arg_parser import ArgParser
    # bool train = true;
    # double t_final = -1.0;
    # double dtc = 0.0;
    # double ddt = 0.0;
    # int numWindows = 0;
    # int windowNumSamples = infty;
    # int windowOverlapSamples = 0;
    # const char *rbf = "G";
    # const char *interp_method = "LS";
    # double admd_closest_rbf_val = 0.9;
    # double ef = 0.9999;
    # int rdim = -1;
    # const char *list_dir = "dmd_list";
    # const char *data_dir = "dmd_data";
    # const char *sim_name = "sim";
    # const char *var_name = "sol";
    # const char *train_list = "dmd_train_local";
    # const char *test_list = "dmd_test";
    # const char *temporal_idx_list = "temporal_idx";
    # const char *spatial_idx_list = "spatial_idx";
    # const char *hdf_name = "dmd.hdf";
    # const char *snap_pfx = "step";
    # const char *basename = "";
    # bool save_csv = false;
    # bool csvFormat = true;

    parser = ArgParser(description="local_tw_csv")
    parser.add_argument('-vis', '--visualization',
                        action='store_true', default=True,
                        help='Enable GLVis visualization')
    parser.add_argument("-train", "--train",
                        action='store_true', dest='train', default=True,
                        help="Enable DMD training.")
    parser.add_argument("-no-train", "--no-train",
                        action='store_false', dest='train',
                        help="Disable DMD training.")
    parser.add_argument("-tf", "--t-final",
                        action='store', default=-1.0, type=float,
                        help="Final time.")
    parser.add_argument("-dtc", "--dtc",
                        action='store', default=0.0, type=float,
                        help="Fixed (constant) dt.")
    parser.add_argument("-ddt", "--dtime-step",
                        action='store', default=0.0, type=float,
                        help="Desired Time step.")
    parser.add_argument("-nwin", "--numwindows",
                        action='store', default=0, type=int,
                        help="Number of DMD windows.")
    parser.add_argument("-nwinsamp", "--numwindowsamples",
                        action='store', default=infty, type=int,
                        help="Number of samples in DMD windows.")
    parser.add_argument("-nwinover", "--numwindowoverlap",
                        action='store', default=0, type=int,
                        help="Number of samples for DMD window overlap.")
    parser.add_argument("-rbf", "--radial-basis-function",
                        action='store', default="G", type=str,
                        help="Radial basis function used in interpolation. Options: \"G\", \"IQ\", \"IMQ\".")
    parser.add_argument("-interp", "--interpolation-method",
                        action='store', default="LS", type=str,
                        help="Method of interpolation. Options: \"LS\", \"IDW\", \"LP\".")
    parser.add_argument("-acrv", "--admd-crv",
                        action='store', default=0.9, type=float,
                        help="Adaptive DMD closest RBF value.")
    parser.add_argument("-ef", "--energy-fraction",
                        action='store', default=0.9999, type=float,
                        help="Energy fraction for DMD.")
    parser.add_argument("-rdim", "--rdim",
                        action='store', default=-1, type=int,
                        help="Reduced dimension for DMD.")
    parser.add_argument("-list", "--list-directory",
                        action='store', default="dmd_list", type=str,
                        help="Location of training and testing data list.")
    parser.add_argument("-data", "--data-directory",
                        action='store', default='dmd_data', type=str,
                        help="Location of training and testing data.")
    parser.add_argument("-hdffile", "--hdf-file",
                        action='store', default='dmd.hdf', type=str,
                        help="Name of HDF file for training and testing data.")
    parser.add_argument("-sim", "--sim-name",
                        action='store', default='sim', type=str,
                        help="Name of simulation.")
    parser.add_argument("-var", "--variable-name",
                        action='store', default='sol', type=str,
                        help="Name of variable.")
    parser.add_argument("-train-set", "--training-set-name",
                        action='store', default='dmd_train_local', type=str,
                        help="Name of the training datasets within the list directory.")
    parser.add_argument("-test-set", "--testing-set-name",
                        action='store', default='dmd_test', type=str,
                        help="Name of the testing datasets within the list directory.")
    parser.add_argument("-t-idx", "--temporal-index",
                        action='store', default='temporal_idx', type=str,
                        help="Name of the file indicating bound of temporal indices.")
    parser.add_argument("-x-idx", "--spatial-index",
                        action='store', default='spatial_idx', type=str,
                        help="Name of the file indicating spatial indices.")
    parser.add_argument("-snap-pfx", "--snapshot-prefix",
                        action='store', default='step', type=str,
                        help="Prefix of snapshots.")
    parser.add_argument("-o", "--outputfile-name",
                        action='store', default='', type=str,
                        help="Name of the sub-folder to dump files within the run directory.")
    parser.add_argument("-save", "--save",
                        action='store_true', dest='save',
                        help="Enable prediction result output (files in CSV format).")
    parser.add_argument("-no-save", "--no-save",
                        action='store_false', dest='save',
                        help="Disable prediction result output (files in CSV format).")
    parser.add_argument("-csv", "--csv",
                        action='store_true', dest='csvFormat',
                        help="Use CSV or HDF format for input files.")
    parser.add_argument("-hdf", "--hdf",
                        action='store_false', dest='csvFormat',
                        help="Use CSV or HDF format for input files.")

    args = parser.parse_args()
    if (myid == 0):
        parser.print_options(args)

    dtc                     = args.dtc
    ddt                     = args.dtime_step
    t_final                 = args.t_final
    basename                = args.outputfile_name
    csvFormat               = args.csvFormat
    var_name                = args.variable_name
    data_dir                = args.data_directory
    sim_name                = args.sim_name
    hdf_name                = args.hdf_file
    spatial_idx_list        = args.spatial_index
    temporal_idx_list       = args.temporal_index
    train                   = args.train
    numWindows              = args.numwindows
    list_dir                = args.list_directory
    train_list              = args.training_set_name
    test_list               = args.testing_set_name
    windowNumSamples        = args.numwindowsamples
    windowOverlapSamples    = args.numwindowoverlap
    rbf                     = args.radial_basis_function
    interp_method           = args.interpolation_method
    admd_closest_rbf_val    = args.admd_crv
    snap_pfx                = args.snapshot_prefix
    rdim                    = args.rdim
    ef                      = args.energy_fraction
    

    assert(not ((dtc > 0.0) and (ddt > 0.0)))

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

    variable = var_name
    nelements = 0

    if (csvFormat):
        nelements = db.getIntegerArray(data_dir + "/dim.csv", 1)
    else:
        db.open("%s/%s0/%s" % (data_dir, sim_name, hdf_name), "r")
        nelements = db.getDoubleArraySize("step0sol")
        db.close()

    assert(nelements > 0)
    if (myid == 0):
        print("Variable %s has dimension %d." % (var_name, nelements))

    dim = nelements
    idx_state_size = csv_db.getInteger("idx_state_size")
    idx_state = csv_db.getIntegerArray("%s/%s.csv" % (data_dir, spatial_idx_list), idx_state_size)
    if (idx_state.size > 0):
        dim = idx_state.size
        if (myid == 0):
            print("Restricting on %d entries out of %d." % (dim, nelements))

    if ((not train) or (numWindows > 0)):
        indicator_val_size = db.getInteger("indicator_val_size")
        indicator_val = db.getDoubleArray("%s/indicator_val.csv" % outputPath,
                                        indicator_val_size)
        if (indicator_val.size > 0):
            if (numWindows > 0):
                assert(numWindows == indicator_val.size)
            else:
                numWindows = indicator_val.size
            if (myid == 0):
                print("Read indicator range partition with %d windows." % numWindows)

    npar = 0
    # int num_train_snap_orig, num_train_snap;
    # string training_par_dir;
    # vector<string> training_par_list;
    # vector<int> training_snap_bound;
    if (train):
        training_par_list = csv_db.getStringVector("%s/%s.csv" % (list_dir, train_list), False)
        npar = training_par_list.size
        assert(npar == 1)

        training_par_dir = training_par_list[0].split(',')[0]

        if (csvFormat):
            num_train_snap_orig = csv_db.getLineCount("%s/%s.csv" % (list_dir, training_par_dir))
        else:
            num_train_snap_orig = db.getInteger("numsnap")
        assert(num_train_snap_orig > 0)

        snap_bound_size = 0
        if (csvFormat):
            prefix = "%s/%s/" % (data_dir, training_par_dir)
            snap_bound_size = csv_db.getLineCount(prefix + temporal_idx_list + suffix)
        else:
            db.open("%s/%s/%s" % (data_dir, training_par_dir, hdf_name), "r")
            snap_bound_size = db.getInteger("snap_bound_size")

        if (snap_bound_size > 0):
            assert(snap_bound_size == 2)
            training_snap_bound = db.getIntegerArray(prefix + temporal_idx_list + suffix, snap_bound_size)
            training_snap_bound[0] -= 1
            training_snap_bound[1] -= 1
            num_train_snap = training_snap_bound[1] - training_snap_bound[0] + 1
            if (myid == 0):
                print("Restricting on snapshot #%d to %d." % (training_snap_bound[0], training_snap_bound[1]))
        else:
            training_snap_bound = np.array([0, num_train_snap_orig - 1])
            num_train_snap = num_train_snap_orig

        db.close()

        assert(windowOverlapSamples < windowNumSamples)
        numWindows = int(round(float(num_train_snap - 1) / float(windowNumSamples))) if (windowNumSamples < infty) else 1

    assert(numWindows > 0)
    if (myid == 0):
        if (numWindows > 1):
            print("Using time windowing DMD with %d windows." % numWindows)
        else:
            print("Using serial DMD.")

    dmd = [None] * numWindows
    for window in range(numWindows):
        if (train):
            if (ddt > 0.0):
                dmd[window] = algo.AdaptiveDMD(dim, ddt, rbf,
                                interp_method, admd_closest_rbf_val)
            elif (dtc > 0.0):
                dmd[window] = algo.DMD(dim, dtc)
            else:
                dmd[window] = algo.NonuniformDMD(dim)
        else:
            if (myid == 0):
                print("Loading DMD model #%d." % window)
            dmd[window] = algo.DMD("%s/window%d" % (outputPath, window))

    dmd_training_timer, dmd_preprocess_timer, dmd_prediction_timer = StopWatch(), StopWatch(), StopWatch()
    # double* sample = new double[dim];

    if (train):
        dmd_training_timer.Start()

        par_dir = training_par_list[0].split(',')[0]

        if (csvFormat):
            snap_list = csv_db.getStringVector("%s/%s.csv" % (list_dir, par_dir), False)
        else:
            snap_index_list = db.getIntegerArray("snap_list", num_train_snap_orig)

        if (myid == 0):
            print("Loading samples for %s to train DMD." % par_dir)

        if (csvFormat): prefix = data_dir + "/" + par_dir + "/"
        tvec = db.getDoubleArray(prefix + "tval" + suffix, num_train_snap_orig)

        curr_window = 0
        overlap_count = 0
        for idx_snap in range(training_snap_bound[0], training_snap_bound[1] + 1):
            snap = snap_pfx
            tval = tvec[idx_snap]

            if (idx_snap == training_snap_bound[0]):
                indicator_val = np.append(indicator_val, tval)

            if (csvFormat):
                snap += snap_list[idx_snap]; # STATE
                data_filename = "%s/%s/%s/%s.csv" % (data_dir, par_dir, snap, variable) # path to VAR_NAME.csv
                sample = db.getDoubleArray(data_filename, nelements, idx_state)
            else:
                snap += "%d" % (snap_index_list[idx_snap]) # STATE
                sample = db.getDoubleArray(snap + "sol", nelements, idx_state)

            dmd[curr_window].takeSample(sample, tval)
            if (overlap_count > 0):
                dmd[curr_window-1].takeSample(sample, tval)
                overlap_count -= 1
            if ((curr_window+1 < numWindows) and (idx_snap+1 <= training_snap_bound[1])):
                new_window = False
                if (windowNumSamples < infty):
                    new_window = (idx_snap >= training_snap_bound[0] + (curr_window+1) * windowNumSamples)
                else:
                    new_window = (tval >= indicator_val[curr_window+1])
                if (new_window):
                    overlap_count = windowOverlapSamples
                    curr_window += 1
                    if (windowNumSamples < infty):
                        indicator_val = np.append(indicator_val, tval)
                    dmd[curr_window].takeSample(sample, tval)

        if (myid == 0):
            print("Loaded %d samples for %s." % (num_train_snap, par_dir))
            if (windowNumSamples < infty):
                print("Created new indicator range partition with %d windows." % numWindows)
                csv_db.putDoubleVector("%s/indicator_val.csv" % outputPath,
                                       indicator_val, numWindows)

        for window in range(numWindows):
            if (rdim != -1):
                if (myid == 0):
                    print("Creating DMD model #%d with rdim: %d" % (window, rdim))
                dmd[window].train(rdim)
            elif (ef != -1):
                if (myid == 0):
                    print("Creating DMD model #%d with energy fraction: %f" % (window, ef))
                dmd[window].train(ef)
            dmd[window].save("%s/window%d" % (outputPath, window))
            if (myid == 0):
                dmd[window].summary("%s/window%d" % (outputPath, window))

        dmd_training_timer.Stop()

        if (ddt > 0.0):
            admd = None
            interp_snap = linalg.Vector(dim, True)
            interp_error = []

            for window in range(numWindows):
                admd = dmd[window]
                assert(admd is not None)
                t_init = dmd[window].getTimeOffset()

                dtc = admd.getTrueDt()
                f_snapshots = admd.getInterpolatedSnapshots()
                if (myid == 0):
                    print("Verifying Adaptive DMD model #%d against interpolated snapshots." % window)

                for k in range(f_snapshots.numColumns()):
                    interp_snap = f_snapshots.getColumn(k)
                    result = admd.predict(t_init + k * dtc)

                    dmd_solution = mfem.Vector(result.getData(), dim)
                    true_solution = mfem.Vector(interp_snap.getData(), dim)
                    diff = mfem.Vector(true_solution.Size())
                    mfem.subtract_vector(dmd_solution, true_solution, diff)

                    tot_diff_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, diff, diff))
                    tot_true_solution_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, true_solution,
                                                         true_solution))
                    rel_error = tot_diff_norm / tot_true_solution_norm
                    interp_error += [rel_error]

                    if (myid == 0):
                        print("Norm of interpolated snapshot #%d is %f" % (k, tot_true_solution_norm))
                        print("Absolute error of DMD prediction for interpolated snapshot #%d is %f" % (k, tot_diff_norm))
                        print("Relative error of DMD prediction for interpolated snapshot #%d is %f" % (k, rel_error))
                    del result

                if (myid == 0):
                    csv_db.putDoubleVector("%s/window%d_interp_error.csv" % (outputPath, window),
                                           interp_error, f_snapshots.numColumns())
                interp_error.clear()

        db.close()

    testing_par_list = csv_db.getStringVector("%s/%s.csv" % (list_dir, test_list), False)
    npar = testing_par_list.size
    assert(npar > 0)

    num_tests = 0
    prediction_time, prediction_error = [], []

    for idx_dataset in range(npar):
        par_dir = testing_par_list[idx_dataset].split(',')[0] # testing DATASET

        if (myid == 0):
            print("Predicting solution for %s using DMD." % par_dir)

        num_snap_orig = 0
        if (csvFormat):
            num_snap_orig = csv_db.getLineCount("%s/%s.csv" % (list_dir, par_dir))
        else:
            db.open(data_dir + "/" + par_dir + "/" + hdf_name, "r")
            num_snap_orig = db.getInteger("numsnap")

        assert(num_snap_orig > 0)
        if (csvFormat):
            snap_list = csv_db.getStringVector("%s/%s.csv" % (list_dir, par_dir), False)
        else:
            snap_index_list = db.getIntegerArray("snap_list", num_snap_orig)

        if (csvFormat): prefix = "%s/%s/" % (data_dir, par_dir)

        tvec = db.getDoubleArray(prefix + "tval" + suffix, num_snap_orig)

        snap_bound_size = 0
        if (csvFormat):
            prefix = "%s/%s/" % (data_dir, par_dir)
            snap_bound_size = csv_db.getLineCount(prefix + temporal_idx_list + suffix)
        else:
            db.open(data_dir + "/" + par_dir + "/" + hdf_name, "r")
            snap_bound_size = db.getInteger("snap_bound_size")

        snap_bound = db.getIntegerArray(prefix + temporal_idx_list + suffix, snap_bound_size)

        if (snap_bound_size > 0):
            assert(snap_bound_size == 2)
            snap_bound = db.getIntegerArray(prefix + temporal_idx_list + suffix, snap_bound_size)
            snap_bound[0] -= 1
            snap_bound[1] -= 1
            if (myid == 0):
                print("Restricting on snapshot #%d to %d." % (snap_bound[0], snap_bound[1]))
        else:
            snap_bound = np.array([0, num_snap_orig - 1])

        num_snap = snap_bound[1] - snap_bound[0] + 1
        curr_window = 0
        for idx_snap in range(snap_bound[0], snap_bound[1] + 1):
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

            if (idx_snap == 0):
                dmd_preprocess_timer.Start()
                init_cond = None
                for window in range(numWindows):
                    if (myid == 0):
                        print("Projecting initial condition at t = %f for DMD model #%d" % (indicator_val[window], window))
                    if (window == 0):
                        init_cond = linalg.Vector(dim, True)
                        for i in range(dim):
                            init_cond[i] = sample[i]
                    else:
                        init_cond = dmd[window-1].predict(indicator_val[window])
                    dmd[window].projectInitialCondition(init_cond)
                    del init_cond
                dmd_preprocess_timer.Stop()

            if (t_final > 0.0): # Actual prediction without true solution for comparison
                num_tests += 1
                while ((curr_window+1 < numWindows) and (t_final > indicator_val[curr_window+1])):
                    curr_window += 1

                if (myid == 0):
                    print("Predicting DMD solution at t = %f using DMD model #%d" % (t_final, curr_window))

                dmd_prediction_timer.Start()
                result = dmd[curr_window].predict(t_final)
                dmd_prediction_timer.Stop()
                if (myid == 0):
                    csv_db.putDoubleArray(outputPath + "/" + par_dir +
                                          "_final_time_prediction.csv",
                                          result.getData(), dim)
                idx_snap = snap_bound[1]+1; # escape for-loop over idx_snap
                del result
            else: # Verify DMD prediction results against dataset
                while ((curr_window+1 < numWindows) and (tval > indicator_val[curr_window+1])):
                    curr_window += 1

                if (myid == 0):
                    print("Predicting DMD solution #%d at t = %f using DMD model #%d" % (idx_snap, tval, curr_window))

                dmd_prediction_timer.Start()
                result = dmd[curr_window].predict(tval)
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
                    if (save_csv):
                        csv_db.putDoubleArray("%s/%s_%s_prediction.csv" % (outputPath, par_dir, snap),
                                              result.getData(), dim)
                        if (dim < nelements):
                            csv_db.putDoubleArray("%s/%s_%s_state.csv" % (outputPath, par_dir, snap),
                                                  sample, dim)
                del result

        if ((myid == 0) and (t_final <= 0.0)):
            csv_db.putDoubleVector(outputPath + "/" + par_dir + "_prediction_time.csv",
                                   prediction_time, num_snap)
            csv_db.putDoubleVector(outputPath + "/" + par_dir + "_prediction_error.csv",
                                   prediction_error, num_snap)

        # prediction_time.clear()
        # prediction_error.clear()
        num_tests = num_tests + 1 if (t_final > 0.0) else num_tests + num_snap

        db.close()

    assert(num_tests > 0)

    if (myid == 0):
        print("Elapsed time for training DMD: %e second\n" % dmd_training_timer.duration)
        print("Elapsed time for preprocessing DMD: %e second\n" % dmd_preprocess_timer.duration)
        print("Total elapsed time for predicting DMD: %e second\n" % dmd_prediction_timer.duration)
        print("Average elapsed time for predicting DMD: %e second\n" % dmd_prediction_timer.duration / num_tests)

    del sample, dmd
