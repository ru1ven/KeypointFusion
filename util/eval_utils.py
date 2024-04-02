import numpy as np


def eval_auc(joint_error_list,sample_number):

    num_kp=21

    data = list()
    num_kp = num_kp
    for _ in range(num_kp):
        data.append(list())

    for i in range(len(joint_error_list)):
        for ii in range(joint_error_list[i][0].shape[0]):
            for j in range(num_kp):
                data[j].append(joint_error_list[i][0][ii][j])

    auc,pck_curve_all, threshs = get_measures(data,0, 50, 20)
    print('stage:' ,0)
    print('Area under curve: %.3f' % auc)
    pck_curve_all, threshs = pck_curve_all[8:], threshs[8:] * 1000.0
    auc_subset = calc_auc(threshs, pck_curve_all)
    print('Area under curve between 20mm - 50mm: %.3f' % auc_subset)

    for i in range(len(joint_error_list)):
        for ii in range(joint_error_list[i][-1].shape[0]):
            for j in range(num_kp):
                data[j].append(joint_error_list[i][-1][ii][j])

    auc,pck_curve_all, threshs = get_measures(data,0, 50, 20)
    print('stage:' ,-1)
    print('Area under curve: %.3f' % auc)
    pck_curve_all, threshs = pck_curve_all[8:], threshs[8:] * 1000.0
    auc_subset = calc_auc(threshs, pck_curve_all)
    print('Area under curve between 20mm - 50mm: %.3f' % auc_subset)


def get_measures(data,val_min=0.0, val_max=0.050, steps=20):
    """ Outputs the average mean and median error as well as the pck score. """
    thresholds = np.linspace(val_min, val_max, steps)
    thresholds = np.array(thresholds)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)


    auc_all = list()
    pck_curve_all = list()

    # Create one plot for each part
    for part_id in range(21):

        # pck/auc
        pck_curve = list()
        for t in thresholds:
            pck = _get_pck(data,part_id, t)
            pck_curve.append(pck)

        pck_curve = np.array(pck_curve)
        pck_curve_all.append(pck_curve)
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        auc_all.append(auc)

    auc_all = np.mean(np.array(auc_all))
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

    return auc_all, pck_curve_all, thresholds

def _get_pck(data, kp_id, threshold):
    """ Returns pck for one keypoint for the given threshold. """
    if len(data[kp_id]) == 0:
        return None

    data = np.array(data[kp_id])
    pck = np.mean((data <= threshold).astype('float'))
    return pck

def calc_auc(x, y):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)

    return integral / norm