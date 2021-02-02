"""
Not at all needed for package, but helps for debugging.
"""
from matplotlib import pyplot as plt  # not in requirements
import numpy as np
from optics_functions.constants import S, REAL, IMAG


def plot_rdts_vs_ptc(df_rdt, df_ptc_rdt, df_twiss, rdt_names):
    for rdt in rdt_names:
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(df_twiss[S], df_ptc_rdt[f"{rdt}{REAL}"], color="C0", label="PTC")
        axs[0].plot(df_twiss[S], np.real(df_rdt[f"{rdt}"]), color="C1", label="Analytical")
        axs[0].legend(loc="upper right")
        axs[0].set_ylabel(f"real {rdt}")

        axs[1].plot(df_twiss[S], df_ptc_rdt[f"{rdt}{IMAG}"], color="C0", label="PTC")
        axs[1].plot(df_twiss[S], np.imag(df_rdt[f"{rdt}"]), color="C1", label="Analytical")
        axs[1].set_ylabel(f"imag {rdt}")

        axs[2].plot(df_twiss[S], np.sqrt(df_ptc_rdt[f"{rdt}{IMAG}"]**2 + df_ptc_rdt[f"{rdt}{REAL}"]**2), color="C0", label="PTC")
        axs[2].plot(df_twiss[S], np.abs(df_rdt[f"{rdt}"]), color="C1", label="Analytical")
        axs[2].set_ylabel(f"abs {rdt}")
        axs[2].set_xlabel(f"Location [m]")
    plt.show()


def plot_rdts_vs(df_rdt1, label1, df_rdt2,  label2, df_twiss, rdt_names):
    for rdt in rdt_names:
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(df_twiss[S], np.real(df_rdt1[f"{rdt}"]), color="C0", label=label1)
        axs[0].plot(df_twiss[S], np.real(df_rdt2[f"{rdt}"]), color="C1", label=label2)
        axs[0].legend(loc="upper right")
        axs[0].set_ylabel(f"real {rdt}")

        axs[1].plot(df_twiss[S], np.imag(df_rdt1[f"{rdt}"]), color="C0", label=label1)
        axs[1].plot(df_twiss[S], np.imag(df_rdt2[f"{rdt}"]), color="C1", label=label2)
        axs[1].set_ylabel(f"imag {rdt}")

        axs[2].plot(df_twiss[S], np.abs(df_rdt1[f"{rdt}"]), color="C0", label=label1)
        axs[2].plot(df_twiss[S], np.abs(df_rdt2[f"{rdt}"]), color="C1", label=label2)
        axs[2].set_ylabel(f"abs {rdt}")
        axs[2].set_xlabel(f"Location [m]")
    plt.show()
