"""
Dispersion
----------

Functions to calculate the (linear) dispersion.

TODO: Clean this mess up
"""

def calc_linear_dispersion(self):
    """ Calculate the Linear Disperion.

    Eq. 24 in [#FranchiAnalyticformulasrapid2017]_
    """
    self._check_k_columns(["K0L", "K0SL", "K1SL"])
    tw = self.twiss_df
    phs_adv = self.get_phase_adv()
    res = self._results_df
    coeff_fun = self._linear_dispersion_coeff
    sum_fun = self._linear_dispersion_sum

    # Calculate
    LOG.debug("Calculate Linear Dispersion")
    with timeit(lambda t: LOG.debug("  Time needed: {:f}".format(t))):
        # sources
        k0_mask = tw['K0L'] != 0
        k0s_mask = tw['K0SL'] != 0
        k1s_mask = tw['K1SL'] != 0

        mx_mask = k0_mask | k1s_mask  # magnets contributing to Dx,j (-> Dy,m)
        my_mask = k0s_mask | k1s_mask  # magnets contributing to Dy,j (-> Dx,m)

        if not any(mx_mask | my_mask):
            LOG.warning("  No linear dispersion contributions found. Values will be zero.")
            res['DX'] = 0.
            res['DY'] = 0.
            self._log_added('DX', 'DY')
            return

        # create temporary DataFrame for magnets with coefficients already in place
        df = tfs.TfsDataFrame(index=tw.index).join(
            coeff_fun(tw.loc[:, 'BETX'], tw.Q1)).join(
            coeff_fun(tw.loc[:, 'BETY'], tw.Q2))
        df.columns = ['COEFFX', 'COEFFY']

        LOG.debug("  Calculate uncoupled linear dispersion")
        df.loc[my_mask, 'DX'] = df.loc[my_mask, 'COEFFX'] * \
                                sum_fun(tw.loc[mx_mask, 'K0L'],
                                        0,
                                        0,
                                        tw.loc[mx_mask, 'BETX'],
                                        tau(phs_adv['X'].loc[mx_mask, my_mask], tw.Q1)
                                        ).transpose()
        df.loc[mx_mask, 'DY'] = df.loc[mx_mask, 'COEFFY'] * \
                                sum_fun(-tw.loc[my_mask, 'K0SL'],  # MINUS!
                                        0,
                                        0,
                                        tw.loc[my_mask, 'BETY'],
                                        tau(phs_adv['Y'].loc[my_mask, mx_mask], tw.Q2)
                                        ).transpose()

        LOG.debug("  Calculate full linear dispersion values")
        res.loc[:, 'DX'] = df.loc[:, 'COEFFX'] * \
                           sum_fun(tw.loc[mx_mask, 'K0L'],
                                   tw.loc[mx_mask, 'K1SL'],
                                   df.loc[mx_mask, 'DY'],
                                   tw.loc[mx_mask, 'BETX'],
                                   tau(phs_adv['X'].loc[mx_mask, :], tw.Q1)
                                   ).transpose()
        res.loc[:, 'DY'] = df.loc[:, 'COEFFY'] * \
                           sum_fun(-tw.loc[my_mask, 'K0SL'],  # MINUS!
                                   tw.loc[my_mask, 'K1SL'],
                                   df.loc[my_mask, 'DX'],
                                   tw.loc[my_mask, 'BETY'],
                                   tau(phs_adv['Y'].loc[my_mask, :], tw.Q2)
                                   ).transpose()

    LOG.debug("  Average linear dispersion Dx: {:g}".format(
        np.mean(res['DX'])))
    LOG.debug("  Average linear dispersion Dy: {:g}".format(
        np.mean(res['DY'])))
    self._log_added('DX', 'DY')

def get_linear_dispersion(self):
    """ Return the Linear Dispersion.

    Available after calc_linear_dispersion!
    """
    if "DX" not in self._results_df or "DY" not in self._results_df:
        self.calc_linear_dispersion()
    return self._results_df.loc[:, ["S", "DX", "DY"]]

def plot_linear_dispersion(self, combined=True):
    """ Plot the Linear Dispersion.

    Available after calc_linear_dispersion!

    Args:
        combined (bool): If 'True' plots x and y into the same axes.
    """
    LOG.debug("Plotting Linear Dispersion")
    lin_disp = self.get_linear_dispersion().dropna()
    title = 'Linear Dispersion'
    pstyle.set_style(self._plot_options.style, self._plot_options.manual)

    if combined:
        ax_dx = lin_disp.plot(x='S')
        ax_dx.set_title(title)
        pstyle.set_yaxis_label('dispersion', 'x,y', ax_dx)
        ax_dy = ax_dx
    else:
        ax_dx = lin_disp.plot(x='S', y='DX')
        ax_dx.set_title(title)
        pstyle.set_yaxis_label('dispersion', 'x', ax_dx)

        ax_dy = lin_disp.plot(x='S', y='DY')
        ax_dy.set_title(title)
        pstyle.set_yaxis_label('dispersion', 'y', ax_dy)

    for ax in (ax_dx, ax_dy):
        self._nice_axes(ax)
        ax.legend()

# helpers
@staticmethod
def _linear_dispersion_coeff(beta, q):
    """ Helper to calculate the coefficient """
    return np.sqrt(beta) / (2 * np.sin(np.pi * q))

@staticmethod
def _linear_dispersion_sum(k, j, d, beta, tau):
    """ Helper to calculate the sum """
    # k, j, d , beta = columns -> convert to Series -> broadcasted
    # tau = Matrix as Frame
    calc_column = (k + j * d) * np.sqrt(beta)
    if isinstance(calc_column, pd.DataFrame):
        calc_column = calc_column.squeeze()
    return np.cos(2 * np.pi * tau).mul(calc_column, axis='index').sum(axis='index')
