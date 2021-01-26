"""
Chromaticity
----------

Functions to calculate chromaticity and chromatic beating.

TODO: Clean this mess up
"""


def calc_linear_chromaticity(self):
    """ Calculate the Linear Chromaticity

    Eq. 31 in [#FranchiAnalyticformulasrapid2017]_
    """
    res = self._results_df

    if 'CHROMX' not in res:
        self._calc_chromatic_term()

    LOG.debug("Calculating Linear Chromaticity")
    with timeit(lambda t: LOG.debug("  Time needed: {:f}".format(t))):
        DQ1 = - 1/(4 * np.pi) * res['CHROMX'].dropna().sum(axis="index")
        DQ2 = 1/(4 * np.pi) * res['CHROMY'].dropna().sum(axis="index")

    self._results_df.DQ1 = DQ1
    self._results_df.DQ2 = DQ2
    LOG.debug("  Q'x: {:f}".format(DQ1))
    LOG.debug("  Q'y: {:f}".format(DQ2))

    self._log_added('DQ1', 'DQ2')

def get_linear_chromaticity(self):
    """ Return the Linear Chromaticity

    Available after calc_linear_chromaticity
    """
    try:
        return self._results_df.DQ1, self._results_df.DQ2
    except AttributeError:
        self.calc_linear_chromaticity()
        return self._results_df.DQ1, self._results_df.DQ2

################################
#     Chromatic Beating
################################

def calc_chromatic_beating(self):
    """ Calculate the Chromatic Beating

    Eq. 36 in [#FranchiAnalyticformulasrapid2017]_
    """
    tw = self.twiss_df
    res = self._results_df
    phs_adv = self.get_phase_adv()

    if 'CHROMX' not in res:
        self._calc_chromatic_term()

    LOG.debug("Calculating Chromatic Beating")
    with timeit(lambda t: LOG.debug("  Time needed: {:f}".format(t))):
        chromx = res['CHROMX'].dropna()
        chromy = res['CHROMY'].dropna()
        res['DBEATX'] = self._chromatic_beating(
            chromx,
            tau(phs_adv['X'].loc[chromx.index, :], tw.Q1),
            tw.Q1).transpose() - 1
        res['DBEATY'] = - self._chromatic_beating(
            chromy,
            tau(phs_adv['Y'].loc[chromy.index, :], tw.Q2),
            tw.Q2).transpose() - 1

    LOG.debug("  Pk2Pk chromatic beating DBEATX: {:g}".format(
        res['DBEATX'].max() - res['DBEATX'].min()))
    LOG.debug("  Pk2Pk chromatic beating DBEATY: {:g}".format(
        res['DBEATY'].max() - res['DBEATY'].min()))
    self._log_added('DBEATX', 'DBEATY')

def get_chromatic_beating(self):
    """ Return the Chromatic Beating

     Available after calc_chromatic_beating
     """
    if "DBEATX" not in self._results_df or "DBEATY" not in self._results_df:
        self.calc_chromatic_beating()
    return self._results_df.loc[:, ["S", "DBEATX", "DBEATY"]]

def plot_chromatic_beating(self, combined=True):
    """ Plot the Chromatic Beating

    Available after calc_chromatic_beating

    Args:
        combined (bool): If 'True' plots x and y into the same axes.
    """
    LOG.debug("Plotting Chromatic Beating")
    chrom_beat = self.get_chromatic_beating().dropna()
    title = 'Chromatic Beating'
    pstyle.set_style(self._plot_options.style, self._plot_options.manual)

    if combined:
        ax_dx = chrom_beat.plot(x='S')
        ax_dx.set_title(title)
        pstyle.small_title(ax_dx)
        pstyle.set_name(title, ax_dx)
        pstyle.set_yaxis_label('dbetabeat', 'x,y', ax_dx)
        ax_dy = ax_dx
    else:
        ax_dx = chrom_beat.plot(x='S', y='DBEATX')
        ax_dx.set_title(title)
        pstyle.small_title(ax_dx)
        pstyle.set_name(title, ax_dx)
        pstyle.set_yaxis_label('dbetabeat', 'x', ax_dx)

        ax_dy = chrom_beat.plot(x='S', y='DBEATY')
        ax_dy.set_title(title)
        pstyle.small_title(ax_dy)
        pstyle.set_name(title, ax_dy)
        pstyle.set_yaxis_label('dbetabeat', 'y', ax_dy)

    for ax in (ax_dx, ax_dy):
        self._nice_axes(ax)
        ax.legend()

@staticmethod
def _chromatic_beating(chrom_term, tau, q):
    return 1 / (2 * np.sin(2 * np.pi * q)) * \
           np.cos(4 * np.pi * tau).mul(chrom_term, axis='index').sum(axis='index')


    def _calc_chromatic_term(self):
        """ Calculates the chromatic term which is common to all chromatic equations """
        LOG.debug("Calculating Chromatic Term.")
        self._check_k_columns(["K1L", "K2L", "K2SL"])
        res = self._results_df
        tw = self.twiss_df

        with timeit(lambda t: LOG.debug("  Time needed: {:f}".format(t))):
            mask = (tw['K1L'] != 0) | (tw['K2L'] != 0) | (tw['K2SL'] != 0)
            if "DX" in tw and "DY" in tw:
                LOG.debug("Dispersion values found in model. Used for chromatic calculations")
                sum_term = tw.loc[mask, 'K1L'] - \
                           (tw.loc[mask, 'K2L'] * tw.loc[mask, 'DX']) + \
                           (tw.loc[mask, 'K2SL'] * tw.loc[mask, 'DY'])
            else:
                LOG.info("Dispersion values NOT found in model. Using analytic values.")
                if "DX" not in res or "DY" not in res:
                    self.calc_linear_dispersion()
                sum_term = tw.loc[mask, 'K1L'] - \
                           (tw.loc[mask, 'K2L'] * res.loc[mask, 'DX']) + \
                           (tw.loc[mask, 'K2SL'] * res.loc[mask, 'DY'])

            res['CHROMX'] = sum_term * tw.loc[mask, 'BETX']
            res['CHROMY'] = sum_term * tw.loc[mask, 'BETY']

        LOG.debug("Chromatic Term Calculated.")
