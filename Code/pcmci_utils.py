
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.gpdc import GPDC
from tigramite import data_processing as pp



def run_pcmci(data):
    pp_data = pp.DataFrame(data)
    gpdc = GPDC(significance="shuffle_test")  # analytic 或 'shuffle_test'/'bootstrap' 用于非线性显著性计算
    pcmci = PCMCI(dataframe=pp_data, cond_ind_test=gpdc, verbosity=1)
    results = pcmci.run_pcmci(tau_max=3, pc_alpha=None, alpha_level=0.1)
    return results


# def run_pcmci(data):
#     cond_ind_test = CMIknn(significance='shuffle_test')
#     pcmci = PCMCI(dataframe=data, cond_ind_test=cond_ind_test)
#     results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05)
#     return results
