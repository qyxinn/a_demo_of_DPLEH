# The following codes adapted are from https://scikit-survival.readthedocs.io/en/stable/index.html
# S. Pölsterl, “scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn,”
# Journal of Machine Learning Research, vol. 21, no. 212, pp. 1–6, 2020.
from util import check_y_survival
import numpy
from sklearn.utils import check_consistent_length, check_array
from mynonparametric import CensoringDistributionEstimator


def _check_estimate_1d(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {:d}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate

def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate_1d(estimate, event_time)

    if not numpy.issubdtype(event_indicator.dtype, numpy.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time, times):
    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    # times = numpy.unique(times)

    if times.max() > 50 or times.min() < 0:
        raise ValueError(
            'all times must be within follow-up time of test data: [{}; {}['.format(
                test_time.min(), test_time.max()))

    return times


def _check_estimate_2d(estimate, test_time, time_points):
    # estimate = check_array(estimate, ensure_2d=False, allow_nd=False)
    # time_points = _check_times(test_time, time_points)
    # check_consistent_length(test_time, estimate)
    #
    # if estimate.ndim == 2 and estimate.shape[1] != time_points.shape[0]:
    #     raise ValueError("expected estimate with {} columns, but got {}".format(
    #         time_points.shape[0], estimate.shape[1]))

    return estimate, time_points


def brier_score(survival_train, survival_test, estimate, times):
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(estimate, test_time, times)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = numpy.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = numpy.inf

    # Calculating the brier scores at each time point
    brier_scores = numpy.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        if i<len(estimate[:,0]):   #无
            est = estimate[:, i]
            is_case = (test_time <= t) & test_event
            is_control = test_time > t

            brier_scores[i] = numpy.mean(numpy.square(est) * is_case.astype(int) / prob_cens_y
                                         + numpy.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i])

    return times, brier_scores


def integrated_brier_score(survival_train, survival_test, estimate, times):
    # Computing the brier scores
    times, brier_scores = brier_score(survival_train, survival_test, estimate, times)

    if times.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    # Computing the IBS
    ibs_value = numpy.trapz(brier_scores, times) / (times[-1] - times[0])

    return ibs_value