#The following codes are adapted from https://scikit-survival.readthedocs.io/en/stable/index.html
#S. Pölsterl, “scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn,” 
#Journal of Machine Learning Research, vol. 21, no. 212, pp. 1–6, 2020.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted
from util import check_y_survival

__all__ = [
    'CensoringDistributionEstimator',
    'kaplan_meier_estimator',
    'nelson_aalen_estimator',
    'ipc_weights',
    'SurvivalFunctionEstimator',
]


def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.
    Parameters
    ----------
    event : array
        Boolean event indicator.
    time : array
        Survival time or time of censoring.
    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.
    Returns
    -------
    times : array
        Unique time points.
    n_events : array
        Number of events at each time point.
    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.
    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0] # shape[0]就是读取矩阵第一维度的长度

    if order is None:
        order = numpy.argsort(time, kind="mergesort") #argsort()函数的作用是将数组按照从小到大的顺序排序，并按照对应的索引值输出。

    uniq_times = numpy.empty(n_samples, dtype=time.dtype) # 根据给定的维度和数值类型返回一个新的数组，元素随机
    uniq_events = numpy.empty(n_samples, dtype=numpy.int_)
    uniq_counts = numpy.empty(n_samples, dtype=numpy.int_)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:  # =1代表右删失
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = numpy.resize(uniq_times, j) #uniq_times前j个值
    n_events = numpy.resize(uniq_events, j)
    total_count = numpy.resize(uniq_counts, j)
    n_censored = total_count - n_events  #删失=总数-死亡

    # offset cumulative sum by one
    total_count = numpy.r_[0, total_count]
    n_at_risk = n_samples - numpy.cumsum(total_count)  # cumsum的作用主要就是计算轴向的累加和。

    return times, n_events, n_at_risk[:-1], n_censored


def _compute_counts_truncated(event, time_enter, time_exit):
    """Compute counts for left truncated and right censored survival data.
    Parameters
    ----------
    event : array
        Boolean event indicator.
    time_start : array
        Time when a subject entered the study.
    time_exit : array
        Time when a subject left the study due to an
        event or censoring.
    Returns
    -------
    times : array
        Unique time points.
    n_events : array
        Number of events at each time point.
    n_at_risk : array
        Number of samples that are censored or have an event at each time point.
    """
    if (time_enter > time_exit).any():
        raise ValueError("exit time must be larger start time for all samples")

    n_samples = event.shape[0]

    uniq_times = numpy.sort(numpy.unique(numpy.r_[time_enter, time_exit]), kind="mergesort") #sort对每一行进行排序 r_连接列表为一行9）numpy.unique() 函数接受一个数组，去除其中重复元素，并按元素由小到大返回一个新的无元素重复的元组或者列表。
    total_counts = numpy.empty(len(uniq_times), dtype=numpy.int_)
    event_counts = numpy.empty(len(uniq_times), dtype=numpy.int_)

    order_enter = numpy.argsort(time_enter, kind="mergesort") # argsort()函数的作用是将数组按照从小到大的顺序排序，并按照对应的索引值输出
    order_exit = numpy.argsort(time_exit, kind="mergesort")
    s_time_enter = time_enter[order_enter]
    s_time_exit = time_exit[order_exit]

    t0 = uniq_times[0]
    # everything larger is included
    idx_enter = numpy.searchsorted(s_time_enter, t0, side="right") # 在数组a中插入数组v（并不执行插入操作），返回一个下标列表，这个列表指明了v中对应元素应该插入在a中那个位置上，left:返回所在位置 right:返回下一个位置
    # everything smaller is excluded
    idx_exit = numpy.searchsorted(s_time_exit, t0, side="left")

    total_counts[0] = idx_enter
    # except people die on the day they enter
    event_counts[0] = 0

    for i in range(1, len(uniq_times)):
        ti = uniq_times[i]

        while idx_enter < n_samples and s_time_enter[idx_enter] <= ti:
            idx_enter += 1

        while idx_exit < n_samples and s_time_exit[idx_exit] < ti:
            idx_exit += 1

        risk_set = numpy.setdiff1d(order_enter[:idx_enter], order_exit[:idx_exit], assume_unique=True) # setdiff1d的作用是求两个数组的集合差。返回' ar1 '中不在' ar2 '中的唯一值。 True:不进行去重复
        total_counts[i] = len(risk_set)

        count_event = 0
        k = idx_exit
        while k < n_samples and s_time_exit[k] == ti:
            if event[order_exit[k]]:
                count_event += 1
            k += 1
        event_counts[i] = count_event

    return uniq_times, event_counts, total_counts


def kaplan_meier_estimator(event, time_exit, time_enter=None, time_min=None, reverse=False):
    """Kaplan-Meier estimator of survival function.
    See [1]_ for further description.
    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Contains binary event indicators.
    time_exit : array-like, shape = (n_samples,)
        Contains event/censoring times.
    time_enter : array-like, shape = (n_samples,), optional
        Contains time when each individual entered the study for
        left truncated survival data.
    time_min : float, optional
        Compute estimator conditional on survival at least up to
        the specified time.
    reverse : bool, optional, default: False
        Whether to estimate the censoring distribution.
        When there are ties between times at which events are observed,
        then events come first and are subtracted from the denominator.
        Only available for right-censored data, i.e. `time_enter` must
        be None.
    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.
    prob_survival : array, shape = (n_times,)
        Survival probability at each unique time point.
        If `time_enter` is provided, estimates are conditional probabilities.
    Examples
    --------
    Creating a Kaplan-Meier curve:
    >>> x, y = kaplan_meier_estimator(event, time)
    >>> plt.step(x, y, where="post")
    >>> plt.ylim(0, 1)
    >>> plt.show()
    References
    ----------
    .. [1] Kaplan, E. L. and Meier, P., "Nonparametric estimation from incomplete observations",
           Journal of The American Statistical Association, vol. 53, pp. 457-481, 1958.
    """
    event, time_enter, time_exit = check_y_survival(event, time_enter, time_exit, allow_all_censored=True)
    check_consistent_length(event, time_enter, time_exit)

    if time_enter is None:
        uniq_times, n_events, n_at_risk, n_censored = _compute_counts(event, time_exit)

        if reverse:
            n_at_risk -= n_events
            n_events = n_censored
    else:
        if reverse:
            raise ValueError("The censoring distribution cannot be estimated from left truncated data")

        uniq_times, n_events, n_at_risk = _compute_counts_truncated(event, time_enter, time_exit)

    # account for 0/0 = nan
    ratio = numpy.divide(n_events, n_at_risk,
                         out=numpy.zeros(uniq_times.shape[0], dtype=float),
                         where=n_events != 0) # numpy.divide(arr1，arr2，out = None，where = True，cast ='same_kind'，order ='K'，dtype = None)：将第一个数组中的数组元素除以第二个元素中的元素(所有情况均逐个元素发生)。 arr1和arr2必须具有相同的形状，并且arr2中的元素不能为零；否则会引发错误。
    values = 1.0 - ratio

    if time_min is not None:
        mask = uniq_times >= time_min
        uniq_times = numpy.compress(mask, uniq_times) # numpy.compress()沿着相应的axis进行切片，并提取condition=1对应位置上的切片重组成新的数组
        values = numpy.compress(mask, values)

    y = numpy.cumprod(values) # numpy.cumprod返回沿给定轴的元素的累积乘积
    return uniq_times, y


def nelson_aalen_estimator(event, time):
    """Nelson-Aalen estimator of cumulative hazard function.
    See [1]_, [2]_ for further description.
    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Contains binary event indicators.
    time : array-like, shape = (n_samples,)
        Contains event/censoring times.
    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.
    cum_hazard : array, shape = (n_times,)
        Cumulative hazard at each unique time point.
    References
    ----------
    .. [1] Nelson, W., "Theory and applications of hazard plotting for censored failure data",
           Technometrics, vol. 14, pp. 945-965, 1972.
    .. [2] Aalen, O. O., "Nonparametric inference for a family of counting processes",
           Annals of Statistics, vol. 6, pp. 701–726, 1978.
    """
    event, time = check_y_survival(event, time)
    check_consistent_length(event, time)
    uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time)

    y = numpy.cumsum(n_events / n_at_risk) # 累加

    return uniq_times, y


def ipc_weights(event, time):
    """Compute inverse probability of censoring weights
    Parameters
    ----------
    event : array, shape = (n_samples,)
        Boolean event indicator.
    time : array, shape = (n_samples,)
        Time when a subject experienced an event or was censored.
    Returns
    -------
    weights : array, shape = (n_samples,)
        inverse probability of censoring weights
    See also
    --------
    CensoringDistributionEstimator
        An estimator interface for estimating inverse probability
        of censoring weights for unseen time points.
    """
    if event.all():
        return numpy.ones(time.shape[0])

    unique_time, p = kaplan_meier_estimator(event, time, reverse=True)

    idx = numpy.searchsorted(unique_time, time[event]) # 在数组a中插入数组v（并不执行插入操作），返回一个下标列表，这个列表指明了v中对应元素应该插入在a中那个位置上
    Ghat = p[idx]

    assert (Ghat > 0).all()

    weights = numpy.zeros(time.shape[0])
    weights[event] = 1.0 / Ghat

    return weights


class SurvivalFunctionEstimator(BaseEstimator):
    """Kaplan–Meier estimate of the survival function."""

    def __init__(self):
        pass

    def fit(self, y):
        """Estimate survival distribution from training data.
        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        Returns
        -------
        self
        """
        event, time = check_y_survival(y, allow_all_censored=True)

        unique_time, prob = kaplan_meier_estimator(event, time)
        self.unique_time_ = numpy.r_[-numpy.infty, unique_time] # numpy.infty正无穷大的浮点表示。
        self.prob_ = numpy.r_[1., prob]

        return self

    def predict_proba(self, time):
        """Return probability of an event after given time point.
        :math:`\\hat{S}(t) = P(T > t)`
        Parameters
        ----------
        time : array, shape = (n_samples,)
            Time to estimate probability at.
        Returns
        -------
        prob : array, shape = (n_samples,)
            Probability of an event.
        """
        check_is_fitted(self, "unique_time_")
        time = check_array(time, ensure_2d=False)

        # K-M is undefined if estimate at last time point is non-zero
        extends = time > self.unique_time_[-1]
        # if self.prob_[-1] > 0 and extends.any():
        #     raise ValueError("time must be smaller than largest "
        #                      "observed time point: {}".format(self.unique_time_[-1]))

        # beyond last time point is zero probability
        Shat = numpy.empty(time.shape, dtype=float)
        Shat[extends] = 0.0

        valid = ~extends
        time = time[valid]
        idx = numpy.searchsorted(self.unique_time_, time)
        # for non-exact matches, we need to shift the index to left
        eps = numpy.finfo(self.unique_time_.dtype).eps
        exact = numpy.absolute(self.unique_time_[idx] - time) < eps
        idx[~exact] -= 1
        Shat[valid] = self.prob_[idx]

        return Shat


class CensoringDistributionEstimator(SurvivalFunctionEstimator):
    """Kaplan–Meier estimator for the censoring distribution."""

    def fit(self, y):
        """Estimate censoring distribution from training data.
        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        Returns
        -------
        self
        """
        event, time = check_y_survival(y)
        if event.all():
            self.unique_time_ = numpy.unique(time)
            self.prob_ = numpy.ones(self.unique_time_.shape[0])
        else:
            unique_time, prob = kaplan_meier_estimator(event, time, reverse=True)
            self.unique_time_ = numpy.r_[-numpy.infty, unique_time]
            self.prob_ = numpy.r_[1., prob]

        return self

    def predict_ipcw(self, y):
        """Return inverse probability of censoring weights at given time points.
        :math:`\\omega_i = \\delta_i / \\hat{G}(y_i)`
        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        Returns
        -------
        ipcw : array, shape = (n_samples,)
            Inverse probability of censoring weights.
        """
        event, time = check_y_survival(y)
        Ghat = self.predict_proba(time[event])

        if (Ghat == 0.0).any():
            raise ValueError("censoring survival function is zero at one or more time points")

        weights = numpy.zeros(time.shape[0])
        weights[event] = 1.0 / Ghat

        return weights