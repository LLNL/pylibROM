import sys
import pytest
sys.path.append("../build")
import _pylibROM.linalg as linalg
import numpy as np 
import _pylibROM.hyperreduction as hyperreduction
from mpi4py import MPI 


def test_s_opt():
    comm = MPI.COMM_WORLD
    d_rank = comm.Get_rank()
    d_num_procs = comm.Get_size()
    
    num_total_rows = 100
    num_cols = 5
    num_samples = 10

    num_rows = int(num_total_rows / d_num_procs)
    if num_total_rows % d_num_procs > d_rank:
        num_rows += 1
    row_offset = np.zeros(d_num_procs + 1, dtype=int)
    row_offset[-1] = num_total_rows
    row_offset[d_rank] = num_rows

    row_offset = comm.allgather(row_offset)
    for i in range(d_num_procs - 1, 0, -1):
        row_offset[i] = row_offset[i + 1] - row_offset[i]

    cols = [np.zeros(int(num_rows)) for _ in range(num_cols)]

    cols[0]=[-0.080347247, -0.080824941, -0.081302121, -0.081778653, -0.082254499, -0.082729563, -0.083203785, -0.083677135, -0.084149517, -0.084620871, -0.085091129, -0.085560232, -0.086028107, -0.086494654, -0.086959846, -0.087423578, -0.087885775, -0.088346355, -0.088805251, -0.089262374, -0.089717634, -0.09017095, -0.090622231, -0.091071382, -0.091518328, -0.091962948, -0.092405185, -0.092844911, -0.093282059, -0.09371648, -0.094148129, -0.094576865, -0.095002592, -0.095425196, -0.095844604, -0.096260659, -0.096673273, -0.097082317, -0.097487703, -0.097889289, -0.098286957, -0.098680593, -0.099070072, -0.099455267, -0.099836059, -0.10021232, -0.10058392, -0.1009507, -0.10131258, -0.10166936, -0.10202095, -0.10236719, -0.10270795, -0.10304309, -0.10337245, -0.1036959, -0.10401326, -0.10432443, -0.10462923, -0.10492751, -0.10521911, -0.10550388, -0.10578166, -0.10605229, -0.10631561, -0.10657145, -0.10681966, -0.10706006, -0.10729248, -0.10751677, -0.10773274, -0.10794021, -0.10813902, -0.10832901, -0.10850994, -0.10868172, -0.10884411, -0.10899694, -0.10914003, -0.10927318, -0.10939625, -0.10950901, -0.10961127, -0.10970286, -0.1097836, -0.10985329, -0.10991171, -0.10995869, -0.10999406, -0.11001757, -0.11002909, -0.11002836, -0.11001521, -0.10998946, -0.10995089, -0.10989931, -0.10983452, -0.10975632, -0.10966451, -0.1095589]
    cols[1]=[0.16577394, 0.16335766, 0.16091171, 0.15843585, 0.15593028, 0.15339521, 0.15083075, 0.14823714, 0.14561449, 0.14296299, 0.14028285, 0.13757427, 0.1348374, 0.13207248, 0.12927973, 0.12645936, 0.12361159, 0.12073667, 0.11783486, 0.11490637, 0.11195149, 0.10897047, 0.1059636, 0.10293115, 0.099873409, 0.096790671, 0.093683265, 0.090551496, 0.087395683, 0.08421617, 0.081013277, 0.077787377, 0.074538819, 0.071267962, 0.067975201, 0.064660914, 0.061325502, 0.057969365, 0.054592922, 0.051196568, 0.047780793, 0.044345986, 0.040892627, 0.037421178, 0.033932116, 0.030425919, 0.026903089, 0.023364101, 0.019809507, 0.016239811, 0.012655558, 0.009057289, 0.0054455632, 0.0018209502, -0.0018159689, -0.0054646088, -0.0091243582, -0.012794618, -0.016474755, -0.02016414, -0.023862133, -0.027568055, -0.031281237, -0.035001017, -0.03872668, -0.04245754, -0.046192877, -0.049931943, -0.053674027, -0.057418343, -0.061164159, -0.064910673, -0.068657115, -0.072402671, -0.076146543, -0.079887874, -0.083625838, -0.087359622, -0.09108834, -0.094811104, -0.098527029, -0.10223524, -0.10593481, -0.1096248, -0.11330433, -0.11697239, -0.12062807, -0.12427036, -0.12789832, -0.1315109, -0.13510716, -0.13868606, -0.14224654, -0.14578757, -0.1493081, -0.15280713, -0.15628348, -0.15973617, -0.16316403, -0.16656601]
    cols[2]=[0.17970525, 0.17156506, 0.16346279, 0.1554046, 0.14739309, 0.13943158, 0.13152277, 0.12366994, 0.11587607, 0.10814466, 0.10047892, 0.092881918, 0.085356973, 0.077908114, 0.070538476, 0.063251555, 0.056051292, 0.048940923, 0.041924898, 0.035006601, 0.028190091, 0.021479465, 0.014878601, 0.0083916532, 0.0020227535, -0.0042235223, -0.010343327, -0.016332217, -0.022185223, -0.027898202, -0.033466496, -0.038885534, -0.044150256, -0.049256705, -0.054199141, -0.058972981, -0.063573271, -0.06799484, -0.072232634, -0.0762816, -0.080136321, -0.083791472, -0.087241642, -0.09048152, -0.093505576, -0.096307948, -0.098883197, -0.10122582, -0.10332985, -0.10518932, -0.10679857, -0.10815176, -0.10924243, -0.11006482, -0.11061303, -0.11088023, -0.11086062, -0.11054792, -0.10993535, -0.10901707, -0.10778636, -0.10623648, -0.10436113, -0.10215379, -0.099607565, -0.096715726, -0.093471497, -0.089868546, -0.085899509, -0.081557475, -0.076836079, -0.07172785, -0.066225693, -0.060323235, -0.054012626, -0.047287457, -0.040140413, -0.032564275, -0.024552036, -0.016096218, -0.0071898461, 0.0021743081, 0.01200313, 0.022304369, 0.033084646, 0.044351578, 0.056112174, 0.068373777, 0.081143118, 0.094428174, 0.10823533, 0.12257186, 0.13744484, 0.1528616, 0.16882883, 0.18535329, 0.2024428, 0.22010295, 0.23834157, 0.25716436]
    cols[3]=[0.28032145, 0.24957921, 0.22029646, 0.19246724, 0.16607673, 0.14107944, 0.11744556, 0.095142469, 0.074139498, 0.054404691, 0.035903051, 0.018604863, 0.0024785693, -0.012502356, -0.026368668, -0.039152525, -0.050883375, -0.06158939, -0.071300052, -0.08004047, -0.087844491, -0.094733283, -0.10074241, -0.10589745, -0.11021838, -0.11374335, -0.11649536, -0.11850194, -0.11978894, -0.12038445, -0.12031612, -0.1196097, -0.11829152, -0.11639512, -0.11393934, -0.11095797, -0.10747339, -0.10351679, -0.099114455, -0.094291791, -0.089079134, -0.083502203, -0.077586457, -0.071369395, -0.064870395, -0.058119785, -0.051145002, -0.043979399, -0.036648117, -0.029182721, -0.021608554, -0.013964453, -0.0062712179, 0.001441621, 0.0091359029, 0.016785355, 0.024359601, 0.031827055, 0.039152235, 0.046306755, 0.053257871, 0.059972771, 0.066413671, 0.072555423, 0.078359023, 0.083794571, 0.088822588, 0.093407162, 0.097518355, 0.10111759, 0.10416858, 0.10663764, 0.10848565, 0.1096708, 0.11016409, 0.10992322, 0.10890646, 0.1070776, 0.10439669, 0.10082465, 0.096319415, 0.090843625, 0.084349707, 0.076800086, 0.068151481, 0.058358833, 0.047379136, 0.035172291, 0.02169859, 0.0068927137, -0.0092670191, -0.026840456, -0.045869015, -0.06639801, -0.088478565, -0.1121584, -0.13748048, -0.16450156, -0.19326067, -0.22381401]
    cols[4]=[-0.19527201, -0.16434221, -0.13427129, -0.10615309, -0.07985004, -0.055393722, -0.032901697, -0.012218075, 0.0067121238, 0.023830034, 0.039134681, 0.052733436, 0.064722776, 0.075113587, 0.08388216, 0.091114402, 0.096780472, 0.10109665, 0.10404827, 0.10575347, 0.10611742, 0.10527392, 0.10343203, 0.10052377, 0.09656097, 0.091740102, 0.086089775, 0.079710215, 0.072633594, 0.064911105, 0.056622911, 0.047927659, 0.038828827, 0.029457323, 0.019857464, 0.010070637, 0.00030847918, -0.009599966, -0.019352004, -0.029026438, -0.038344849, -0.047509905, -0.056339417, -0.064674459, -0.072539873, -0.07993228, -0.086732931, -0.092825502, -0.098256141, -0.10285441, -0.10685418, -0.1098156, -0.11199083, -0.11326706, -0.11361062, -0.112961, -0.11141182, -0.10881485, -0.1053227, -0.100847, -0.095438726, -0.089022659, -0.081690922, -0.073525988, -0.064379916, -0.054469947, -0.043819919, -0.032419704, -0.020443646, -0.0078809233, 0.0050634136, 0.01833046, 0.031886131, 0.045477595, 0.059011169, 0.072447062, 0.08536046, 0.097745836, 0.10951488, 0.12020078, 0.12977803, 0.13803452, 0.14448762, 0.14898524, 0.1513482, 0.1511752, 0.14807439, 0.14184049, 0.13198961, 0.11820251, 0.099918507, 0.0771074, 0.048986863, 0.014982483, -0.025133438, -0.071965843, -0.12620743, -0.18812096, -0.258641, -0.33815652]
    orthonormal_mat = np.empty((num_rows, num_cols))
    S_OPT_true_ans = np.array([
        -0.5918106, 1.141427, 1.142373, 1.477501, -0.7002456,
        -1.204534, 1.608694, 0.5777631, -0.8686062, 0.9839489,
        -1.21845, 1.569369, 0.4668517, -0.958101, 0.9462786,
        -1.023713, 0.2118963, -1.029118, -0.3487639, -0.740034,
        -1.016572, 0.1513296, -1.060517, -0.2849657, -0.774013,
        -1.009973, 0.0911333, -1.088985, -0.2200843, -0.8032426,
        -1.107097, -0.8899746, -0.7112781, 1.030203, 0.2788396,
        -1.173239, -1.002324, -0.1872992, 1.061846, 0.969148,
        -1.040098, -1.004302, 0.8125958, 0.2427455, 0.9231714,
        -0.570251, -0.9721371, 1.327513, -1.113124, -1.083388
    ])
    
    for i in range(num_rows):
        for j in range(num_cols):
            orthonormal_mat[i][j] = cols[j][i]
            
    u= linalg.Matrix(orthonormal_mat,True,False)
  
    f_sampled_row= [0] * num_samples
    f_sampled_row_true_ans = [0, 19, 21, 48, 49, 50, 72, 79, 90, 97]
    f_sampled_rows_per_proc = [0] * d_num_procs
    f_basis_sampled_inv = linalg.Matrix(num_samples, num_cols,False)

    f_sampled_row,f_sampled_rows_per_proc= hyperreduction.S_OPT(u, num_cols,f_basis_sampled_inv, d_rank, d_num_procs, num_samples)
    curr_index = 0
    for i in range(1, len(f_sampled_rows_per_proc)):
        curr_index += f_sampled_rows_per_proc[i - 1]
        for j in range(curr_index, curr_index + f_sampled_rows_per_proc[i]):
            f_sampled_row[j] += row_offset[i]

    assert np.all(f_sampled_row == f_sampled_row_true_ans)

    l2_norm_diff = 0.0
    for i in range(num_samples):
        for j in range(num_cols):
            l2_norm_diff += np.abs(S_OPT_true_ans[i * num_cols + j] -
                                    f_basis_sampled_inv[i, j]) ** 2
    l2_norm_diff = np.sqrt(l2_norm_diff)
    assert l2_norm_diff < 1e-4

def test_s_opt_less_basis_vectors():
    comm = MPI.COMM_WORLD
    d_rank = comm.Get_rank()
    d_num_procs = comm.Get_size()
   
    num_total_rows = 100
    num_cols = 5
    num_basis_vectors = 3
    num_samples = 5

    num_rows = int(num_total_rows / d_num_procs)
    if num_total_rows % d_num_procs > d_rank:
        num_rows += 1
    row_offset = np.zeros(d_num_procs + 1, dtype=int)
    row_offset[-1] = num_total_rows
    row_offset[d_rank] = num_rows

    row_offset = comm.allgather(row_offset)
    for i in range(d_num_procs - 1, 0, -1):
        row_offset[i] = row_offset[i + 1] - row_offset[i]

    cols = [np.zeros(int(num_rows)) for _ in range(num_cols)]

    cols[0]=[-0.080347247, -0.080824941, -0.081302121, -0.081778653, -0.082254499, -0.082729563, -0.083203785, -0.083677135, -0.084149517, -0.084620871, -0.085091129, -0.085560232, -0.086028107, -0.086494654, -0.086959846, -0.087423578, -0.087885775, -0.088346355, -0.088805251, -0.089262374, -0.089717634, -0.09017095, -0.090622231, -0.091071382, -0.091518328, -0.091962948, -0.092405185, -0.092844911, -0.093282059, -0.09371648, -0.094148129, -0.094576865, -0.095002592, -0.095425196, -0.095844604, -0.096260659, -0.096673273, -0.097082317, -0.097487703, -0.097889289, -0.098286957, -0.098680593, -0.099070072, -0.099455267, -0.099836059, -0.10021232, -0.10058392, -0.1009507, -0.10131258, -0.10166936, -0.10202095, -0.10236719, -0.10270795, -0.10304309, -0.10337245, -0.1036959, -0.10401326, -0.10432443, -0.10462923, -0.10492751, -0.10521911, -0.10550388, -0.10578166, -0.10605229, -0.10631561, -0.10657145, -0.10681966, -0.10706006, -0.10729248, -0.10751677, -0.10773274, -0.10794021, -0.10813902, -0.10832901, -0.10850994, -0.10868172, -0.10884411, -0.10899694, -0.10914003, -0.10927318, -0.10939625, -0.10950901, -0.10961127, -0.10970286, -0.1097836, -0.10985329, -0.10991171, -0.10995869, -0.10999406, -0.11001757, -0.11002909, -0.11002836, -0.11001521, -0.10998946, -0.10995089, -0.10989931, -0.10983452, -0.10975632, -0.10966451, -0.1095589]
    cols[1]=[0.16577394, 0.16335766, 0.16091171, 0.15843585, 0.15593028, 0.15339521, 0.15083075, 0.14823714, 0.14561449, 0.14296299, 0.14028285, 0.13757427, 0.1348374, 0.13207248, 0.12927973, 0.12645936, 0.12361159, 0.12073667, 0.11783486, 0.11490637, 0.11195149, 0.10897047, 0.1059636, 0.10293115, 0.099873409, 0.096790671, 0.093683265, 0.090551496, 0.087395683, 0.08421617, 0.081013277, 0.077787377, 0.074538819, 0.071267962, 0.067975201, 0.064660914, 0.061325502, 0.057969365, 0.054592922, 0.051196568, 0.047780793, 0.044345986, 0.040892627, 0.037421178, 0.033932116, 0.030425919, 0.026903089, 0.023364101, 0.019809507, 0.016239811, 0.012655558, 0.009057289, 0.0054455632, 0.0018209502, -0.0018159689, -0.0054646088, -0.0091243582, -0.012794618, -0.016474755, -0.02016414, -0.023862133, -0.027568055, -0.031281237, -0.035001017, -0.03872668, -0.04245754, -0.046192877, -0.049931943, -0.053674027, -0.057418343, -0.061164159, -0.064910673, -0.068657115, -0.072402671, -0.076146543, -0.079887874, -0.083625838, -0.087359622, -0.09108834, -0.094811104, -0.098527029, -0.10223524, -0.10593481, -0.1096248, -0.11330433, -0.11697239, -0.12062807, -0.12427036, -0.12789832, -0.1315109, -0.13510716, -0.13868606, -0.14224654, -0.14578757, -0.1493081, -0.15280713, -0.15628348, -0.15973617, -0.16316403, -0.16656601]
    cols[2]=[0.17970525, 0.17156506, 0.16346279, 0.1554046, 0.14739309, 0.13943158, 0.13152277, 0.12366994, 0.11587607, 0.10814466, 0.10047892, 0.092881918, 0.085356973, 0.077908114, 0.070538476, 0.063251555, 0.056051292, 0.048940923, 0.041924898, 0.035006601, 0.028190091, 0.021479465, 0.014878601, 0.0083916532, 0.0020227535, -0.0042235223, -0.010343327, -0.016332217, -0.022185223, -0.027898202, -0.033466496, -0.038885534, -0.044150256, -0.049256705, -0.054199141, -0.058972981, -0.063573271, -0.06799484, -0.072232634, -0.0762816, -0.080136321, -0.083791472, -0.087241642, -0.09048152, -0.093505576, -0.096307948, -0.098883197, -0.10122582, -0.10332985, -0.10518932, -0.10679857, -0.10815176, -0.10924243, -0.11006482, -0.11061303, -0.11088023, -0.11086062, -0.11054792, -0.10993535, -0.10901707, -0.10778636, -0.10623648, -0.10436113, -0.10215379, -0.099607565, -0.096715726, -0.093471497, -0.089868546, -0.085899509, -0.081557475, -0.076836079, -0.07172785, -0.066225693, -0.060323235, -0.054012626, -0.047287457, -0.040140413, -0.032564275, -0.024552036, -0.016096218, -0.0071898461, 0.0021743081, 0.01200313, 0.022304369, 0.033084646, 0.044351578, 0.056112174, 0.068373777, 0.081143118, 0.094428174, 0.10823533, 0.12257186, 0.13744484, 0.1528616, 0.16882883, 0.18535329, 0.2024428, 0.22010295, 0.23834157, 0.25716436]
    cols[3]=[0.28032145, 0.24957921, 0.22029646, 0.19246724, 0.16607673, 0.14107944, 0.11744556, 0.095142469, 0.074139498, 0.054404691, 0.035903051, 0.018604863, 0.0024785693, -0.012502356, -0.026368668, -0.039152525, -0.050883375, -0.06158939, -0.071300052, -0.08004047, -0.087844491, -0.094733283, -0.10074241, -0.10589745, -0.11021838, -0.11374335, -0.11649536, -0.11850194, -0.11978894, -0.12038445, -0.12031612, -0.1196097, -0.11829152, -0.11639512, -0.11393934, -0.11095797, -0.10747339, -0.10351679, -0.099114455, -0.094291791, -0.089079134, -0.083502203, -0.077586457, -0.071369395, -0.064870395, -0.058119785, -0.051145002, -0.043979399, -0.036648117, -0.029182721, -0.021608554, -0.013964453, -0.0062712179, 0.001441621, 0.0091359029, 0.016785355, 0.024359601, 0.031827055, 0.039152235, 0.046306755, 0.053257871, 0.059972771, 0.066413671, 0.072555423, 0.078359023, 0.083794571, 0.088822588, 0.093407162, 0.097518355, 0.10111759, 0.10416858, 0.10663764, 0.10848565, 0.1096708, 0.11016409, 0.10992322, 0.10890646, 0.1070776, 0.10439669, 0.10082465, 0.096319415, 0.090843625, 0.084349707, 0.076800086, 0.068151481, 0.058358833, 0.047379136, 0.035172291, 0.02169859, 0.0068927137, -0.0092670191, -0.026840456, -0.045869015, -0.06639801, -0.088478565, -0.1121584, -0.13748048, -0.16450156, -0.19326067, -0.22381401]
    cols[4]=[-0.19527201, -0.16434221, -0.13427129, -0.10615309, -0.07985004, -0.055393722, -0.032901697, -0.012218075, 0.0067121238, 0.023830034, 0.039134681, 0.052733436, 0.064722776, 0.075113587, 0.08388216, 0.091114402, 0.096780472, 0.10109665, 0.10404827, 0.10575347, 0.10611742, 0.10527392, 0.10343203, 0.10052377, 0.09656097, 0.091740102, 0.086089775, 0.079710215, 0.072633594, 0.064911105, 0.056622911, 0.047927659, 0.038828827, 0.029457323, 0.019857464, 0.010070637, 0.00030847918, -0.009599966, -0.019352004, -0.029026438, -0.038344849, -0.047509905, -0.056339417, -0.064674459, -0.072539873, -0.07993228, -0.086732931, -0.092825502, -0.098256141, -0.10285441, -0.10685418, -0.1098156, -0.11199083, -0.11326706, -0.11361062, -0.112961, -0.11141182, -0.10881485, -0.1053227, -0.100847, -0.095438726, -0.089022659, -0.081690922, -0.073525988, -0.064379916, -0.054469947, -0.043819919, -0.032419704, -0.020443646, -0.0078809233, 0.0050634136, 0.01833046, 0.031886131, 0.045477595, 0.059011169, 0.072447062, 0.08536046, 0.097745836, 0.10951488, 0.12020078, 0.12977803, 0.13803452, 0.14448762, 0.14898524, 0.1513482, 0.1511752, 0.14807439, 0.14184049, 0.13198961, 0.11820251, 0.099918507, 0.0771074, 0.048986863, 0.014982483, -0.025133438, -0.071965843, -0.12620743, -0.18812096, -0.258641, -0.33815652]
    orthonormal_mat = np.empty((num_rows, num_cols))
    S_OPT_true_ans = np.array([
       -1.433786,  2.925154,   2.209402,
        -1.861405,  0.6665424, -1.247851,
        -1.890357,  0.5250462, -1.304832,
        -1.935721,  0.3061151, -1.365598,
        -2.835807, -3.503679,   1.97353
    ])
   
    for i in range(num_rows):
        for j in range(num_cols):
            orthonormal_mat[i][j] = cols[j][i]
            
    u= linalg.Matrix(orthonormal_mat,True,False)
  
    f_sampled_row= [0] * num_samples
    f_sampled_row_true_ans = [0, 44, 46, 49, 90]
    f_sampled_rows_per_proc = [0] * d_num_procs
    f_basis_sampled_inv = linalg.Matrix(num_samples, num_basis_vectors,False)

    f_sampled_row, f_sampled_rows_per_proc = hyperreduction.S_OPT(u, num_basis_vectors, f_basis_sampled_inv, d_rank, d_num_procs, num_samples)
    curr_index = 0
    for i in range(1, len(f_sampled_rows_per_proc)):
        curr_index += f_sampled_rows_per_proc[i - 1]
        for j in range(curr_index, curr_index + f_sampled_rows_per_proc[i]):
            f_sampled_row[j] += row_offset[i]

    assert np.all(f_sampled_row == f_sampled_row_true_ans)

    l2_norm_diff = 0.0
    for i in range(num_samples):
        for j in range(num_basis_vectors):
            l2_norm_diff += np.abs(S_OPT_true_ans[i * num_basis_vectors + j] -
                                    f_basis_sampled_inv[i, j]) ** 2
    l2_norm_diff = np.sqrt(l2_norm_diff)
    assert l2_norm_diff < 1e-4

def test_s_opt_init_vector():
    comm = MPI.COMM_WORLD
    d_rank = comm.Get_rank()
    d_num_procs = comm.Get_size()
   
    num_total_rows = 100
    num_cols = 5
    num_basis_vectors = 3
    num_samples = 5

    num_rows = int(num_total_rows / d_num_procs)
    if num_total_rows % d_num_procs > d_rank:
        num_rows += 1
    row_offset = np.zeros(d_num_procs + 1, dtype=int)
    row_offset[-1] = num_total_rows
    row_offset[d_rank] = num_rows

    row_offset = comm.allgather(row_offset)
    for i in range(d_num_procs - 1, 0, -1):
        row_offset[i] = row_offset[i + 1] - row_offset[i]

    cols = [np.zeros(int(num_rows)) for _ in range(num_cols)]

    cols[0]=[-0.080347247, -0.080824941, -0.081302121, -0.081778653, -0.082254499, -0.082729563, -0.083203785, -0.083677135, -0.084149517, -0.084620871, -0.085091129, -0.085560232, -0.086028107, -0.086494654, -0.086959846, -0.087423578, -0.087885775, -0.088346355, -0.088805251, -0.089262374, -0.089717634, -0.09017095, -0.090622231, -0.091071382, -0.091518328, -0.091962948, -0.092405185, -0.092844911, -0.093282059, -0.09371648, -0.094148129, -0.094576865, -0.095002592, -0.095425196, -0.095844604, -0.096260659, -0.096673273, -0.097082317, -0.097487703, -0.097889289, -0.098286957, -0.098680593, -0.099070072, -0.099455267, -0.099836059, -0.10021232, -0.10058392, -0.1009507, -0.10131258, -0.10166936, -0.10202095, -0.10236719, -0.10270795, -0.10304309, -0.10337245, -0.1036959, -0.10401326, -0.10432443, -0.10462923, -0.10492751, -0.10521911, -0.10550388, -0.10578166, -0.10605229, -0.10631561, -0.10657145, -0.10681966, -0.10706006, -0.10729248, -0.10751677, -0.10773274, -0.10794021, -0.10813902, -0.10832901, -0.10850994, -0.10868172, -0.10884411, -0.10899694, -0.10914003, -0.10927318, -0.10939625, -0.10950901, -0.10961127, -0.10970286, -0.1097836, -0.10985329, -0.10991171, -0.10995869, -0.10999406, -0.11001757, -0.11002909, -0.11002836, -0.11001521, -0.10998946, -0.10995089, -0.10989931, -0.10983452, -0.10975632, -0.10966451, -0.1095589]
    cols[1]=[0.16577394, 0.16335766, 0.16091171, 0.15843585, 0.15593028, 0.15339521, 0.15083075, 0.14823714, 0.14561449, 0.14296299, 0.14028285, 0.13757427, 0.1348374, 0.13207248, 0.12927973, 0.12645936, 0.12361159, 0.12073667, 0.11783486, 0.11490637, 0.11195149, 0.10897047, 0.1059636, 0.10293115, 0.099873409, 0.096790671, 0.093683265, 0.090551496, 0.087395683, 0.08421617, 0.081013277, 0.077787377, 0.074538819, 0.071267962, 0.067975201, 0.064660914, 0.061325502, 0.057969365, 0.054592922, 0.051196568, 0.047780793, 0.044345986, 0.040892627, 0.037421178, 0.033932116, 0.030425919, 0.026903089, 0.023364101, 0.019809507, 0.016239811, 0.012655558, 0.009057289, 0.0054455632, 0.0018209502, -0.0018159689, -0.0054646088, -0.0091243582, -0.012794618, -0.016474755, -0.02016414, -0.023862133, -0.027568055, -0.031281237, -0.035001017, -0.03872668, -0.04245754, -0.046192877, -0.049931943, -0.053674027, -0.057418343, -0.061164159, -0.064910673, -0.068657115, -0.072402671, -0.076146543, -0.079887874, -0.083625838, -0.087359622, -0.09108834, -0.094811104, -0.098527029, -0.10223524, -0.10593481, -0.1096248, -0.11330433, -0.11697239, -0.12062807, -0.12427036, -0.12789832, -0.1315109, -0.13510716, -0.13868606, -0.14224654, -0.14578757, -0.1493081, -0.15280713, -0.15628348, -0.15973617, -0.16316403, -0.16656601]
    cols[2]=[0.17970525, 0.17156506, 0.16346279, 0.1554046, 0.14739309, 0.13943158, 0.13152277, 0.12366994, 0.11587607, 0.10814466, 0.10047892, 0.092881918, 0.085356973, 0.077908114, 0.070538476, 0.063251555, 0.056051292, 0.048940923, 0.041924898, 0.035006601, 0.028190091, 0.021479465, 0.014878601, 0.0083916532, 0.0020227535, -0.0042235223, -0.010343327, -0.016332217, -0.022185223, -0.027898202, -0.033466496, -0.038885534, -0.044150256, -0.049256705, -0.054199141, -0.058972981, -0.063573271, -0.06799484, -0.072232634, -0.0762816, -0.080136321, -0.083791472, -0.087241642, -0.09048152, -0.093505576, -0.096307948, -0.098883197, -0.10122582, -0.10332985, -0.10518932, -0.10679857, -0.10815176, -0.10924243, -0.11006482, -0.11061303, -0.11088023, -0.11086062, -0.11054792, -0.10993535, -0.10901707, -0.10778636, -0.10623648, -0.10436113, -0.10215379, -0.099607565, -0.096715726, -0.093471497, -0.089868546, -0.085899509, -0.081557475, -0.076836079, -0.07172785, -0.066225693, -0.060323235, -0.054012626, -0.047287457, -0.040140413, -0.032564275, -0.024552036, -0.016096218, -0.0071898461, 0.0021743081, 0.01200313, 0.022304369, 0.033084646, 0.044351578, 0.056112174, 0.068373777, 0.081143118, 0.094428174, 0.10823533, 0.12257186, 0.13744484, 0.1528616, 0.16882883, 0.18535329, 0.2024428, 0.22010295, 0.23834157, 0.25716436]
    cols[3]=[0.28032145, 0.24957921, 0.22029646, 0.19246724, 0.16607673, 0.14107944, 0.11744556, 0.095142469, 0.074139498, 0.054404691, 0.035903051, 0.018604863, 0.0024785693, -0.012502356, -0.026368668, -0.039152525, -0.050883375, -0.06158939, -0.071300052, -0.08004047, -0.087844491, -0.094733283, -0.10074241, -0.10589745, -0.11021838, -0.11374335, -0.11649536, -0.11850194, -0.11978894, -0.12038445, -0.12031612, -0.1196097, -0.11829152, -0.11639512, -0.11393934, -0.11095797, -0.10747339, -0.10351679, -0.099114455, -0.094291791, -0.089079134, -0.083502203, -0.077586457, -0.071369395, -0.064870395, -0.058119785, -0.051145002, -0.043979399, -0.036648117, -0.029182721, -0.021608554, -0.013964453, -0.0062712179, 0.001441621, 0.0091359029, 0.016785355, 0.024359601, 0.031827055, 0.039152235, 0.046306755, 0.053257871, 0.059972771, 0.066413671, 0.072555423, 0.078359023, 0.083794571, 0.088822588, 0.093407162, 0.097518355, 0.10111759, 0.10416858, 0.10663764, 0.10848565, 0.1096708, 0.11016409, 0.10992322, 0.10890646, 0.1070776, 0.10439669, 0.10082465, 0.096319415, 0.090843625, 0.084349707, 0.076800086, 0.068151481, 0.058358833, 0.047379136, 0.035172291, 0.02169859, 0.0068927137, -0.0092670191, -0.026840456, -0.045869015, -0.06639801, -0.088478565, -0.1121584, -0.13748048, -0.16450156, -0.19326067, -0.22381401]
    cols[4]=[-0.19527201, -0.16434221, -0.13427129, -0.10615309, -0.07985004, -0.055393722, -0.032901697, -0.012218075, 0.0067121238, 0.023830034, 0.039134681, 0.052733436, 0.064722776, 0.075113587, 0.08388216, 0.091114402, 0.096780472, 0.10109665, 0.10404827, 0.10575347, 0.10611742, 0.10527392, 0.10343203, 0.10052377, 0.09656097, 0.091740102, 0.086089775, 0.079710215, 0.072633594, 0.064911105, 0.056622911, 0.047927659, 0.038828827, 0.029457323, 0.019857464, 0.010070637, 0.00030847918, -0.009599966, -0.019352004, -0.029026438, -0.038344849, -0.047509905, -0.056339417, -0.064674459, -0.072539873, -0.07993228, -0.086732931, -0.092825502, -0.098256141, -0.10285441, -0.10685418, -0.1098156, -0.11199083, -0.11326706, -0.11361062, -0.112961, -0.11141182, -0.10881485, -0.1053227, -0.100847, -0.095438726, -0.089022659, -0.081690922, -0.073525988, -0.064379916, -0.054469947, -0.043819919, -0.032419704, -0.020443646, -0.0078809233, 0.0050634136, 0.01833046, 0.031886131, 0.045477595, 0.059011169, 0.072447062, 0.08536046, 0.097745836, 0.10951488, 0.12020078, 0.12977803, 0.13803452, 0.14448762, 0.14898524, 0.1513482, 0.1511752, 0.14807439, 0.14184049, 0.13198961, 0.11820251, 0.099918507, 0.0771074, 0.048986863, 0.014982483, -0.025133438, -0.071965843, -0.12620743, -0.18812096, -0.258641, -0.33815652]
    orthonormal_mat = np.empty((num_rows, num_cols))
    S_OPT_true_ans = np.array([
       -1.433786,  2.925154,   2.209402,
       -1.861405,  0.6665424, -1.247851,
       -1.890357,  0.5250462, -1.304832,
       -1.935721,  0.3061151, -1.365598,
       -2.835807, -3.503679,   1.97353
    ])
   
    for i in range(num_rows):
        for j in range(num_cols):
            orthonormal_mat[i][j] = cols[j][i]
            
    u= linalg.Matrix(orthonormal_mat,True,False)
  
    f_sampled_row= [0] * num_samples
    f_sampled_row_true_ans = [0, 44, 46, 49, 90]
    f_sampled_rows_per_proc = [0] * d_num_procs
    f_basis_sampled_inv = linalg.Matrix(num_samples, num_basis_vectors,False)
    init_samples = []
    print(row_offset[d_rank])
    if np.any(row_offset[d_rank] <= 90):
     if np.any(row_offset[d_rank + 1]) > 90:
       init_samples.append(90 - row_offset[d_rank])

    f_basis_sampled_inv = linalg.Matrix(num_samples, num_basis_vectors, False)
    f_sampled_row, f_sampled_rows_per_proc = hyperreduction.S_OPT(u, num_basis_vectors, f_basis_sampled_inv, d_rank, d_num_procs, num_samples, init_samples)
    curr_index = 0
    for i in range(1, len(f_sampled_rows_per_proc)):
        curr_index += f_sampled_rows_per_proc[i - 1]
        for j in range(curr_index, curr_index + f_sampled_rows_per_proc[i]):
            f_sampled_row[j] += row_offset[i]

    assert np.all(f_sampled_row == f_sampled_row_true_ans)

    l2_norm_diff = 0.0
    for i in range(num_samples):
        for j in range(num_basis_vectors):
            l2_norm_diff += np.abs(S_OPT_true_ans[i * num_basis_vectors + j] -
                                    f_basis_sampled_inv[i, j]) ** 2
    l2_norm_diff = np.sqrt(l2_norm_diff)
    assert l2_norm_diff < 1e-4

    if (d_rank == 0):
        init_samples.append(0)
    
    f_sampled_row, f_sampled_rows_per_proc = hyperreduction.S_OPT(u, num_basis_vectors, f_basis_sampled_inv, d_rank, d_num_procs, num_samples)
    curr_index = 0
    for i in range(1, len(f_sampled_rows_per_proc)):
        curr_index += f_sampled_rows_per_proc[i - 1]
        for j in range(curr_index, curr_index + f_sampled_rows_per_proc[i]):
            f_sampled_row[j] += row_offset[i]

    assert np.all(f_sampled_row == f_sampled_row_true_ans)

    l2_norm_diff = 0.0
    for i in range(num_samples):
        for j in range(num_basis_vectors):
            l2_norm_diff += np.abs(S_OPT_true_ans[i * num_basis_vectors + j] -
                                    f_basis_sampled_inv[i, j]) ** 2
    l2_norm_diff = np.sqrt(l2_norm_diff)
    assert l2_norm_diff < 1e-4

def test_s_opt_qr():
    comm = MPI.COMM_WORLD
    d_rank = comm.Get_rank()
    d_num_procs = comm.Get_size()
   
    num_total_rows = 100
    num_cols = 5
    num_basis_vectors = 3
    num_samples = 5

    num_rows = int(num_total_rows / d_num_procs)
    if num_total_rows % d_num_procs > d_rank:
        num_rows += 1
    row_offset = np.zeros(d_num_procs + 1, dtype=int)
    row_offset[-1] = num_total_rows
    row_offset[d_rank] = num_rows

    row_offset = comm.allgather(row_offset)
    for i in range(d_num_procs - 1, 0, -1):
        row_offset[i] = row_offset[i + 1] - row_offset[i]

    cols = [np.zeros(int(num_rows)) for _ in range(num_cols)]

    cols[0]=[-0.080347247, -0.080824941, -0.081302121, -0.081778653, -0.082254499, -0.082729563, -0.083203785, -0.083677135, -0.084149517, -0.084620871, -0.085091129, -0.085560232, -0.086028107, -0.086494654, -0.086959846, -0.087423578, -0.087885775, -0.088346355, -0.088805251, -0.089262374, -0.089717634, -0.09017095, -0.090622231, -0.091071382, -0.091518328, -0.091962948, -0.092405185, -0.092844911, -0.093282059, -0.09371648, -0.094148129, -0.094576865, -0.095002592, -0.095425196, -0.095844604, -0.096260659, -0.096673273, -0.097082317, -0.097487703, -0.097889289, -0.098286957, -0.098680593, -0.099070072, -0.099455267, -0.099836059, -0.10021232, -0.10058392, -0.1009507, -0.10131258, -0.10166936, -0.10202095, -0.10236719, -0.10270795, -0.10304309, -0.10337245, -0.1036959, -0.10401326, -0.10432443, -0.10462923, -0.10492751, -0.10521911, -0.10550388, -0.10578166, -0.10605229, -0.10631561, -0.10657145, -0.10681966, -0.10706006, -0.10729248, -0.10751677, -0.10773274, -0.10794021, -0.10813902, -0.10832901, -0.10850994, -0.10868172, -0.10884411, -0.10899694, -0.10914003, -0.10927318, -0.10939625, -0.10950901, -0.10961127, -0.10970286, -0.1097836, -0.10985329, -0.10991171, -0.10995869, -0.10999406, -0.11001757, -0.11002909, -0.11002836, -0.11001521, -0.10998946, -0.10995089, -0.10989931, -0.10983452, -0.10975632, -0.10966451, -0.1095589]
    cols[1]=[0.16577394, 0.16335766, 0.16091171, 0.15843585, 0.15593028, 0.15339521, 0.15083075, 0.14823714, 0.14561449, 0.14296299, 0.14028285, 0.13757427, 0.1348374, 0.13207248, 0.12927973, 0.12645936, 0.12361159, 0.12073667, 0.11783486, 0.11490637, 0.11195149, 0.10897047, 0.1059636, 0.10293115, 0.099873409, 0.096790671, 0.093683265, 0.090551496, 0.087395683, 0.08421617, 0.081013277, 0.077787377, 0.074538819, 0.071267962, 0.067975201, 0.064660914, 0.061325502, 0.057969365, 0.054592922, 0.051196568, 0.047780793, 0.044345986, 0.040892627, 0.037421178, 0.033932116, 0.030425919, 0.026903089, 0.023364101, 0.019809507, 0.016239811, 0.012655558, 0.009057289, 0.0054455632, 0.0018209502, -0.0018159689, -0.0054646088, -0.0091243582, -0.012794618, -0.016474755, -0.02016414, -0.023862133, -0.027568055, -0.031281237, -0.035001017, -0.03872668, -0.04245754, -0.046192877, -0.049931943, -0.053674027, -0.057418343, -0.061164159, -0.064910673, -0.068657115, -0.072402671, -0.076146543, -0.079887874, -0.083625838, -0.087359622, -0.09108834, -0.094811104, -0.098527029, -0.10223524, -0.10593481, -0.1096248, -0.11330433, -0.11697239, -0.12062807, -0.12427036, -0.12789832, -0.1315109, -0.13510716, -0.13868606, -0.14224654, -0.14578757, -0.1493081, -0.15280713, -0.15628348, -0.15973617, -0.16316403, -0.16656601]
    cols[2]=[0.17970525, 0.17156506, 0.16346279, 0.1554046, 0.14739309, 0.13943158, 0.13152277, 0.12366994, 0.11587607, 0.10814466, 0.10047892, 0.092881918, 0.085356973, 0.077908114, 0.070538476, 0.063251555, 0.056051292, 0.048940923, 0.041924898, 0.035006601, 0.028190091, 0.021479465, 0.014878601, 0.0083916532, 0.0020227535, -0.0042235223, -0.010343327, -0.016332217, -0.022185223, -0.027898202, -0.033466496, -0.038885534, -0.044150256, -0.049256705, -0.054199141, -0.058972981, -0.063573271, -0.06799484, -0.072232634, -0.0762816, -0.080136321, -0.083791472, -0.087241642, -0.09048152, -0.093505576, -0.096307948, -0.098883197, -0.10122582, -0.10332985, -0.10518932, -0.10679857, -0.10815176, -0.10924243, -0.11006482, -0.11061303, -0.11088023, -0.11086062, -0.11054792, -0.10993535, -0.10901707, -0.10778636, -0.10623648, -0.10436113, -0.10215379, -0.099607565, -0.096715726, -0.093471497, -0.089868546, -0.085899509, -0.081557475, -0.076836079, -0.07172785, -0.066225693, -0.060323235, -0.054012626, -0.047287457, -0.040140413, -0.032564275, -0.024552036, -0.016096218, -0.0071898461, 0.0021743081, 0.01200313, 0.022304369, 0.033084646, 0.044351578, 0.056112174, 0.068373777, 0.081143118, 0.094428174, 0.10823533, 0.12257186, 0.13744484, 0.1528616, 0.16882883, 0.18535329, 0.2024428, 0.22010295, 0.23834157, 0.25716436]
    cols[3]=[0.28032145, 0.24957921, 0.22029646, 0.19246724, 0.16607673, 0.14107944, 0.11744556, 0.095142469, 0.074139498, 0.054404691, 0.035903051, 0.018604863, 0.0024785693, -0.012502356, -0.026368668, -0.039152525, -0.050883375, -0.06158939, -0.071300052, -0.08004047, -0.087844491, -0.094733283, -0.10074241, -0.10589745, -0.11021838, -0.11374335, -0.11649536, -0.11850194, -0.11978894, -0.12038445, -0.12031612, -0.1196097, -0.11829152, -0.11639512, -0.11393934, -0.11095797, -0.10747339, -0.10351679, -0.099114455, -0.094291791, -0.089079134, -0.083502203, -0.077586457, -0.071369395, -0.064870395, -0.058119785, -0.051145002, -0.043979399, -0.036648117, -0.029182721, -0.021608554, -0.013964453, -0.0062712179, 0.001441621, 0.0091359029, 0.016785355, 0.024359601, 0.031827055, 0.039152235, 0.046306755, 0.053257871, 0.059972771, 0.066413671, 0.072555423, 0.078359023, 0.083794571, 0.088822588, 0.093407162, 0.097518355, 0.10111759, 0.10416858, 0.10663764, 0.10848565, 0.1096708, 0.11016409, 0.10992322, 0.10890646, 0.1070776, 0.10439669, 0.10082465, 0.096319415, 0.090843625, 0.084349707, 0.076800086, 0.068151481, 0.058358833, 0.047379136, 0.035172291, 0.02169859, 0.0068927137, -0.0092670191, -0.026840456, -0.045869015, -0.06639801, -0.088478565, -0.1121584, -0.13748048, -0.16450156, -0.19326067, -0.22381401]
    cols[4]=[-0.19527201, -0.16434221, -0.13427129, -0.10615309, -0.07985004, -0.055393722, -0.032901697, -0.012218075, 0.0067121238, 0.023830034, 0.039134681, 0.052733436, 0.064722776, 0.075113587, 0.08388216, 0.091114402, 0.096780472, 0.10109665, 0.10404827, 0.10575347, 0.10611742, 0.10527392, 0.10343203, 0.10052377, 0.09656097, 0.091740102, 0.086089775, 0.079710215, 0.072633594, 0.064911105, 0.056622911, 0.047927659, 0.038828827, 0.029457323, 0.019857464, 0.010070637, 0.00030847918, -0.009599966, -0.019352004, -0.029026438, -0.038344849, -0.047509905, -0.056339417, -0.064674459, -0.072539873, -0.07993228, -0.086732931, -0.092825502, -0.098256141, -0.10285441, -0.10685418, -0.1098156, -0.11199083, -0.11326706, -0.11361062, -0.112961, -0.11141182, -0.10881485, -0.1053227, -0.100847, -0.095438726, -0.089022659, -0.081690922, -0.073525988, -0.064379916, -0.054469947, -0.043819919, -0.032419704, -0.020443646, -0.0078809233, 0.0050634136, 0.01833046, 0.031886131, 0.045477595, 0.059011169, 0.072447062, 0.08536046, 0.097745836, 0.10951488, 0.12020078, 0.12977803, 0.13803452, 0.14448762, 0.14898524, 0.1513482, 0.1511752, 0.14807439, 0.14184049, 0.13198961, 0.11820251, 0.099918507, 0.0771074, 0.048986863, 0.014982483, -0.025133438, -0.071965843, -0.12620743, -0.18812096, -0.258641, -0.33815652]
    orthonormal_mat = np.empty((num_rows, num_cols))
    S_OPT_true_ans = np.array([
        -1.433785, -2.925153, -2.209402,
        -1.861404, -0.6665415, 1.24785,
        -1.890357, -0.5250456, 1.304833,
        -1.93572, -0.3061142, 1.365598,
        -2.835807,  3.503679, -1.97353
    ])
   
    for i in range(num_rows):
        for j in range(num_cols):
            orthonormal_mat[i][j] = cols[j][i]
            
    u= linalg.Matrix(orthonormal_mat,True,False)
  
    f_sampled_row= [0] * num_samples
    f_sampled_row_true_ans = [0, 44, 46, 49, 90]
    f_sampled_rows_per_proc = [0] * d_num_procs
    f_basis_sampled_inv = linalg.Matrix(num_samples, num_basis_vectors,False)
    init_samples=[]
    f_sampled_row, f_sampled_rows_per_proc = hyperreduction.S_OPT(u, num_basis_vectors, f_basis_sampled_inv, d_rank, d_num_procs, num_samples, init_samples, True)
    curr_index = 0
    for i in range(1, len(f_sampled_rows_per_proc)):
        curr_index += f_sampled_rows_per_proc[i - 1]
        for j in range(curr_index, curr_index + f_sampled_rows_per_proc[i]):
            f_sampled_row[j] += row_offset[i]

    assert np.all(f_sampled_row == f_sampled_row_true_ans)

    l2_norm_diff = 0.0
    for i in range(num_samples):
        for j in range(num_basis_vectors):
            l2_norm_diff += np.abs(S_OPT_true_ans[i * num_basis_vectors + j] -
                                    f_basis_sampled_inv[i, j]) ** 2
    l2_norm_diff = np.sqrt(l2_norm_diff)
    assert l2_norm_diff < 1e-4

if __name__ == "__main__":
    pytest.main()
