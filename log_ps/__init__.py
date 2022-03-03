# coding: utf-8

"""
This script watches `ps` output, parses it for the given process, logs the stas
for the given interval and writes it to a file after termination, so that we can
create plots or whatever from the data.
This is intended to replace the constant staring at `htop` for probing process
ressource usage.
"""

import os
import sys
import time
import json
import subprocess
import argparse
import threading
import atexit
import signal

import numpy as np
import matplotlib.pyplot as plt


_INTERVAL_MIN = 0.1  # Minimum sampling interval
# These are constant, no need to always append
_CONST_FIELDS = ["USER", "PID", "TT", "STARTED", "COMMAND"]


def pkwargs(p, **kwargs):
    """ Extracts prefixed keyword arguments from keyword arguments. """
    return {k.replace(p, ""): v for k, v in kwargs.items() if k.startswith(p)}


def load_log(fname):
    """
    Loads a logfile produced by this module.

    Parameters
    ----------
    fname : str
        Logfile to load.

    Returns
    -------
    logs : dict
        Logging stats as dictionary, values are numpy arrays.
    """
    with open(fname) as fp:
        logs = json.load(fp)
    return{k: np.array(v) for k, v in logs.items()}


def summary_plot_ram(
        log, plotname=None, ram_unit="MiB", time_unit="s", **kwargs):
    """
    Creates a summary plot from a logging file or dict showing the RAM (RSS)
    usage over logged time.

    Parameters
    ----------
    log : str or dict or list of str or list of dict
        If str, is interpreted as a logging filename and tries to load the log
        stats from there. Otherwise expects a logging dict directly.
        If encapsulated in a list, plots all logs into the same plot with the
        filenames as legend entries.
    plotname : str or None, optional (default: None)
        If str, saves the plot under that filename using `plt.savefig`.
        Otherwise the plot is shown interactively with `plt.show()`.
    ram_unit : str, optional (default: 'MiB')
        Unit to plot the RSS usage in. Can be 'K(i)B', 'M(i)B', 'G(i)B'.
    time_unit : str, optional (default: 's')
        Unit to plot the runtime in. Can be 's' ,'m', 'h'.
    kwargs : Plot arguments
        These are passed to the plot instances via a prefix:
        - 'fig_<name>' is passed to `plt.subplots(1, 1, ...)`. Example:
          `fig_figsize(10, 6)` makes a larger figure.
        - 'plot_<name>' is passed to the time vs RSS plot line. Example:
          `plot_ls=':', plot_lw=4` makes the line dotted and wide.
        - 'axvline_<name>' is passed to the tag line axvline plot. Example:
          `axvline_c='k', axvline_ls='--'` makes them black and dashed.
          You can't specify 'axvline_label' because that is reserved for the
          tagnames in the log stats.
        - 'grid_<name>' is passed to `plt.grid`.
        - 'legend_<name>' is passed to the legend. Example:
          `legend_loc='upper right' places the legend.
        - 'savefig_<name>' is passed `plt.savefig(plotname, ...)`. Example:
          `savefig_dpi=200` for increasing the resolution.
        Others are:
        - 'labels' sets a label for each line in an extra legend if mutiple
          logs are given. Must be same length as `log` in that case.
        - 'cmap' to set the colormap from which the tag lines are taken if no
          explicit 'axvline_c' key is given.
        - 'xlim', 'ylim' tuple to set the plot limits, passed to
          `ax.set_[x,y]lim`.
        - 'title' to set the plot title.
        - 'ax', a matplotlib axis instance to draw on. If not given, a new
          figure and axis is made.

    Example
    -------
    ```
    summary_plot_ram(
        "./test_log.json", plotname="./test_log.png", ram_unit="GiB",
        axvline_lw=4, axvline_ls=":", plot_c="C6", cmap="copper",
        savefig_dpi=200)
    ```
    """
    if isinstance(log, str):
        stats = [load_log(log)]
    elif isinstance(log, dict):
        stats = [log]
    elif isinstance(log, list):
        stats = []
        for j, log_ in enumerate(log):
            if isinstance(log_, str):
                stats.append(load_log(log_))
            elif isinstance(log_, dict):
                stats.append(log_)
            else:
                raise TypeError(
                    "`log` entry at {} must be either str or dict.".format(j))
        if "labels" in kwargs and kwargs["labels"] is not None:
            if len(kwargs["labels"]) != len(stats):
                raise ValueError("Need a label for each log or set it to None.")
        else:
            kwargs["labels"] = None
    else:
        raise TypeError("`log` must be either str or dict.")

    units = {"kb": 1.024, "kib": 1., "mb": 1.024e-3, "mib": 1. / 1024,
             "gb": 1.024e-6, "gib": 1. / 1024**2}
    if ram_unit.lower() not in units:
        raise ValueError("`ram_unit` can be {}".format(
            ", ".join([k for k in units])))

    t_units = {"s": 1., "m": 1. / 60., "h": 1. / 3600.}
    if time_unit.lower() not in t_units:
        raise ValueError("`time_unit` can be {}".format(
            ", ".join([k for k in t_units])))

    # Put each one in overview plot
    label_coll = []
    if "ax" in kwargs:
        ax = kwargs["ax"]
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(1, 1, **pkwargs("fig_", **kwargs))

    for k, stats_ in enumerate(stats):
        rss = np.array(stats_["RSS"], dtype=float) * units[ram_unit.lower()]
        times = np.array(
            stats_["sample_time"], dtype=float) * t_units[time_unit.lower()]
        tags = stats_["tags"]

        _l = ax.plot(times, rss, **pkwargs("plot_", **kwargs))
        if kwargs.get("labels", None) is not None:
            label_coll.append((_l[0], kwargs["labels"][k]))

        if "axvline_c" in kwargs:
            colors = len(tags) * [kwargs.pop("axvline_c")]
        else:
            colors = plt.get_cmap(kwargs.get("cmap", "brg"))(
                np.linspace(0, 1, len(tags)))
        for j, tag in enumerate(tags):
            ax.axvline(tag[0], 0, 1, c=colors[j], label=tag[1],
                       **pkwargs("axvline_", **kwargs))
    ax.set_xlabel("Time in {}".format(time_unit.lower()))
    _unit_str = ram_unit.upper().replace("I", "i")
    ax.set_ylabel("RAM in {}".format(_unit_str))
    if "xlim" in kwargs:
        ax.set_xlim(kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    ax.grid(**pkwargs("grid_", **kwargs))

    # Craft multiple legends. TODO: Independent loc keyword args
    if label_coll:
        leg1 = ax.legend(
            [lbl[0] for lbl in label_coll], [lbl[1] for lbl in label_coll],
            **pkwargs("legend_", **kwargs))
    if len(tags) > 0:
        ax.legend(**pkwargs("legend_", **kwargs))
        if label_coll:
            ax.add_artist(leg1)

    if plotname is not None:
        fig.savefig(plotname, **pkwargs("savefig_", **kwargs))


def plot_ram_runtime_model(
    log, xvals, model="exp", model_p0=None,
        plotname=None, ram_unit="MiB", time_unit="s", **kwargs):
    """
    Creates a summary plot and an exponential runtime and RAM model from a
    list of stats dicts or logfilenames.

    Parameters
    ----------
    log : list of str or list of dict
        If str, is interpreted as a logging filename and tries to load the log
        stats from there. Otherwise expects a logging dict directly. Uses the
        keys 'sample_time' and 'RSS'.
    xvals : array-like
        X values per logged stat. Must have same length as `log` and be numeric.
        For best plot results, `log` and `xvals` should be sorted ascending
        after `xvals`.
    model : str or dict, optional (default: 'exp')
        If str, can be 'exp' or 'log'. Then, the data is linearized by
        `y->log(y)` for 'exp' or `x->log(x)` for 'log' and a linear fit is done.
        If dict, must contain keys 'ram' and 'rts' for the RAM and runtime fits.
        Keys can be either 'exp', 'log', or a callable. If callables are given,
        they are used in `scipy.optimize.curve_fit` and must take the
        independent variable as the first argument and the parameters to fit as
        separate remaining arguments (eg. `func(x, a, b, c)` with `a, b, c`
        being the fit parameters).
    model_p0 : dict of tuples or None, optional (default: None)
        Model seed parameters for for each fit. `model_p0['rts']` is used for
        the runtime and `model_p0['ram']` is used for the RAM model fit. If
        `None` or any value is `None`, then `scipy.optimize.curve_fit` sets all
        to 1 automatically.
    plotname : str or None, optional (default: None)
        If str, saves the plot under that filename using `plt.savefig`.
        Otherwise the plot is shown interactively with `plt.show()`.
    ram_unit : str, optional (default: 'MiB')
        Unit to plot the RSS usage in. Can be 'K(i)B', 'M(i)B', 'G(i)B'.
    time_unit : str, optional (default: 's')
        Unit to plot the runtime in. Can be 's' ,'m', 'h'.
    kwargs : Plot arguments
        These are passed to the plot instances via a prefix:
        - 'fig_<name>' is passed to `plt.subplots(1, 1, ...)`. Example:
          `fig_figsize(10, 6)` makes a larger figure.
        - 'plt_model_<name>' is passed to `plt.plot` of the data point plots.
          Defaults: 'c': 'k', 'ls': '-', 'marker': ''.
        - 'plt_data_<name>' is passed to `plt.plot` of the model curve plots
          Defaults: 'lw': 1, 'ls': ':', 'marker': 'o'.
        - 'fillb_<name>' is passed to `plt.fill_between` of model buffer range
          plot. Defaults: 'color': 'C7', 'alpha': 0.5.
        - 'grid_<name>' is passed to `plt.grid`.
        - 'savefig_<name>' is passed `plt.savefig(plotname, ...)`. Example:
          `savefig_dpi=200` for increasing the resolution.
        Others are:
        - 'ylim_time' sets the ylim for the runtime plot.
        - 'ylim_ram' sets the ylabel for the max(RAM) plot.
        - 'xlabel' sets the xlabel on both plots.
        - 'xlim' sets the xlim on both plots.
        - 'title_time' sets the title for the runtime plot, default:
          'Runtime model'.
        - 'title_ram' sets the title for the runtime plot, default:
          'max(RAM) model'.
        - 'suptitle' sets the figure suptitle.
        - 'ax', a 2-tuple `(ax_rts, ax_ram)` of matplotlib axis instance to draw
          on. If not given, a new figure and axes are made. The first element is
          used for the runtime plot, the second one for the RAM plot.

    Returns
    -------
    f_rts : callable
        Returns the best fit values of the runtime model for given x-value.
        Callable via `f_rts(x)`.
    f_rams : callable
        Returns the best fit values of the RAM model for given x-value.
        Callable via `f_rams(x)`.
    p_rts: tuple
        Parameter for the runtime model. The fit was done using a logarithmic
        linear fit, so get the values with `np.exp(np.polyval(p_rts, x))`.
    p_rams: tuple
        Parameter for the RAM model. The fit was done using a logarithmic
        linear fit, so get the values with `np.exp(np.polyval(p_rams, x))`.
    rts_max : ndarray
        Runtime maximum values.
    rams_max : ndarray
        RAM maximum values.
    """
    stats = []
    for j, log_ in enumerate(log):
        if isinstance(log_, str):
            stats.append(load_log(log_))
        elif isinstance(log_, dict):
            stats.append(log_)
        else:
            raise TypeError(
                "`log` entry at {} must be either str or dict.".format(j))

    # Input checks
    xvals = np.atleast_1d(xvals)
    if len(xvals) != len(stats):
        raise ValueError("Need a unique x value for each log.")

    r_units = {"kb": 1.024, "kib": 1., "mb": 1.024e-3, "mib": 1. / 1024,
               "gb": 1.024e-6, "gib": 1. / 1024**2}
    if ram_unit.lower() not in r_units:
        raise ValueError("`ram_unit` can be {}".format(
            ", ".join([k for k in r_units])))

    t_units = {"s": 1., "m": 1. / 60., "h": 1. / 3600.}
    if time_unit.lower() not in t_units:
        raise ValueError("`time_unit` can be {}".format(
            ", ".join([k for k in t_units])))

    if not isinstance(model, dict):
        model = {"rts": model, "ram": model}
    else:
        if "rts" not in model or "ram" not in model:
            raise ValueError("`model` must contain keys 'rts' and 'ram'.")

    if model_p0 is None:
        model_p0 = {"rts": None, "ram": None}
    else:
        # Fill in missing. Eg, when model['ram'] is 'exp', no need to specify p0
        model_p0["rts"] = model_p0.get("rts", None)
        model_p0["ram"] = model_p0.get("ram", None)

    # Prepare data points
    rts, rams = [], []
    for stat in stats:
        rts.append(np.array(stat["sample_time"]).astype(float))
        rams.append(np.array(stat["RSS"]).astype(float))
    rams_max = np.array([r.max() for r in rams]) * r_units[ram_unit.lower()]
    rts_max = np.array([t[-1] for t in rts]) * t_units[time_unit.lower()]

    # Build the model
    fit_res = {}
    _maxvals = {"rts": rts_max, "ram": rams_max}
    for name, val in model.items():
        if val == "exp":
            # Take log of y and make a linear fit
            _p = np.polyfit(xvals, np.log(_maxvals[name]), deg=1)
            _f = lambda x, p=_p: np.exp(np.polyval(p, x))
        elif val == "log":
            # Take log of x and make a linear fit
            _p = np.polyfit(np.log(xvals), _maxvals[name], deg=1)
            _f = lambda x, p=_p: np.polyval(p, np.log(x))
        else:
            from scipy.optimize import curve_fit
            _p = curve_fit(
                model[name], xvals, _maxvals[name], p0=model_p0[name])[0]
            _f = lambda x, p=_p: model[name](x, *p)
        fit_res.update({"f_" + name: _f, "p_" + name: _p})
    p_rams, f_rams = fit_res["p_ram"], fit_res["f_ram"]
    p_rts, f_rts = fit_res["p_rts"], fit_res["f_rts"]

    # Now for the plot
    if "ax" in kwargs:
        axl, axr = kwargs["ax"]
        fig = axl.get_figure()
    else:
        fig, (axl, axr) = plt.subplots(1, 2, **pkwargs("fig_", **kwargs))

    # Plot model
    x = np.linspace(xvals[0], xvals[-1], 100)
    y_rts = f_rts(x)
    y_rams = f_rams(x)
    _kwargs = pkwargs("plt_model_", **kwargs)
    _kwargs["c"] = _kwargs.get("c", "k")
    _kwargs["ls"] = _kwargs.get("ls", "-")
    _kwargs["marker"] = _kwargs.get("marker", "")
    axl.plot(x, y_rts, **_kwargs)
    axr.plot(x, y_rams, **_kwargs)

    # Plot data
    _kwargs = pkwargs("plt_data_", **kwargs)
    # Set nice defaults
    _kwargs["lw"] = _kwargs.get("lw", 1)
    _kwargs["ls"] = _kwargs.get("ls", ":")
    _kwargs["marker"] = _kwargs.get("marker", "o")
    axl.plot(xvals, rts_max, **_kwargs, label="Runtime")
    axr.plot(xvals, rams_max, **_kwargs, label="max(RAM)")

    # Add 2GB or 15% percent buffer RAM request and use integer vals as estimate
    _kwargs = pkwargs("fillb_", **kwargs)
    _kwargs["color"] = _kwargs.get("color", "C7")
    _kwargs["alpha"] = _kwargs.get("alpha", 0.5)
    axr.fill_between(
        x, y_rams, np.ceil(np.maximum(1 + y_rams, 1.15 * y_rams)), **_kwargs)

    axl.set_ylabel("Time in {}".format(time_unit.lower()))
    _unit_str = ram_unit.upper().replace("I", "i")
    axr.set_ylabel("RAM in {}".format(_unit_str))
    if "ylim_time" in kwargs:
        axl.set_ylim(kwargs["ylim_time"])
    if "ylim_ram" in kwargs:
        axr.set_ylim(kwargs["ylim_ram"])

    for ax in (axl, axr):
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])
        if "xlim" in kwargs:
            ax.set_xlim(kwargs["xlim"])

    axl.set_title(kwargs.get("title_time", "Runtime model"))
    axr.set_title(kwargs.get("title_ram", "max(RAM) model"))
    if "suptitle" in kwargs:
        fig.suptitle(kwargs["suptitle"])

    axl.grid(**pkwargs("grid_", **kwargs))
    axr.grid(**pkwargs("grid_", **kwargs))

    if "ax" not in kwargs:
        fig.tight_layout()

    if plotname is not None:
        fig.savefig(plotname, **pkwargs("savefig_", **kwargs))

    return f_rts, f_rams, p_rts, p_rams, rts_max, rams_max


def sample_ps():
    """
    Sample and parse `ps` output for given PID.

    Returns
    -------
    out_narrow : dict
        Dictionary with header names as keys and information for all processes
        for that key as a list as values.
    out_wide : list
        List of dictionaries, one for each process, where each dict has header
        names as keys and information for the current process as values.
    Note
    ----
    ps columns are (superuser.com/questions/117913):
      USER = User owning the process
      PID = Process ID of the process
      %CPU = CPU time used divided by the time the process has been running
      %MEM = Ratio of the RSS to the physical memory on the machine
      VSZ = Virtual memory usage of the entire process (in KiB)
      RSS = Resident Set Size, the non-swapped physical memory that a task has
            used (in KiB?)
      TTY = Controlling tty (terminal)
      STAT = Multi-character process state
      START = Starting time or date of the process
      TIME = Cumulative CPU time
      COMMAND = Command with all its arguments
    """
    cmd = ["ps", "aux"]
    out = subprocess.check_output(cmd).decode("utf-8").split("\n")
    # Parse columns, be careful about CMD column which may contain extra
    # whitespaces. So split and glue and use header count to get data columns
    header = [c.strip() for c in out[0].split(None)]
    ncols = len(header)
    data = [line.split(None, maxsplit=ncols - 1)
            for line in out[1:] if line.strip()]
    for i, line in enumerate(data):
        if len(line) != ncols:
            raise ValueError(
                "Unexpected output while parsing `ps`. Line {} has not length "
                "{}: {}".format(i + 1, ncols, ", ".join(line)))
    out_narrow = {h: [d[i] for d in data] for i, h in enumerate(header)}
    out_wide = [{h: di for h, di in zip(header, d)} for d in data]
    return out_narrow, out_wide


class PSLogger(threading.Thread):
    """
    Manager class to supervise ps logging.

    Parameters
    ----------
    pid : int
        Process PID to watch. Use as `pid=os.getpid()` in caller.
    fname : str or None, optional (default: None)
        Saves the logged stats to this file after the logging is stopped or in
        the event of a crash. If not given, the log is lost when a crash occurs.
    interval : int, optional (default: 1)
        Update interval in seconds, at least 0.1s, smaller values are ignored.

    Example
    -------
    ```
    import os
    import time

    from log_ps import PSLogger, summary_plot_ram


    if __name__ == "__main__":
        PID = os.getpid()
        print("Start logging PID {}".format(PID))
        log_fname = "./test_log_self.log"
        logger = PSLogger(PID, fname=log_fname, interval=0.1)
        logger.start()

        arr = []
        for i in range(5):
            logger.tag("Pass {}, start allocating".format(i))
            arr.append([i for i in range(int(1e7))])
            logger.tag("Pass {}, sleeping 1s".format(i))
            time.sleep(1)

        logger.tag("Done")
        logs = logger.stop()
        summary_plot_ram(
            logs, plotname=log_fname.replace(".json", ".png"),
            ram_unit="GiB", axvline_lw=4, axvline_ls=":", plot_c="C7",
            cmap="copper", savefig_dpi=200)
    ```
    """

    def __init__(self, pid, fname=None, interval=1):
        threading.Thread.__init__(self)

        self._pid = str(pid)  # ps output is str too
        self._interval = max(_INTERVAL_MIN, int(interval))
        self._stats = {"tags": []}

        if fname is not None:
            path = os.path.dirname(
                os.path.expandvars(os.path.expanduser(fname)))
            if not path:
                path = "."
            if not os.path.isdir(path):
                raise ValueError(
                    "Output directory '{}' does not exist.".format(path))
            fname = os.path.basename(fname)
            if not fname:
                raise ValueError("Given filename is invalid.")
            fname = os.path.join(path, fname)
            if not fname.endswith(".json"):
                fname += ".json"
        self._fname = fname

        # Used for checking if stop() was called.
        # stackoverflow.com/questions/323972
        self._stop_trigger = threading.Event()

        # Register the stop method to properly shutdown the thread on kevboard
        # interrupts.
        # www.netways.de/blog/2016/07/21/atexit-oder-wie-man-python-dienste-nicht-beenden-sollte
        atexit.register(self.stop)  # sys.exit
        signal.signal(signal.SIGINT, self.stop)  # Ctrl-C
        signal.signal(signal.SIGTSTP, self.stop)  # Ctrl-Z
        signal.signal(signal.SIGTERM, self.stop)  # external SIGTERM

    def run(self):
        """ Overwrites `threading.Thread.run` """
        while True:
            _t0 = time.time()
            # Sample ps stats and find given PID
            ps_info_narrow, ps_info_wide = sample_ps()
            try:
                pids = ps_info_narrow["PID"]
            except (ValueError, KeyError):
                self._save_output()
                print("No PID information in ps output. Stopping logging.")
                break
            try:
                idx = pids.index(self._pid)
            except ValueError:
                self._save_output()
                print("Requested PID '{}' not found in `ps` output. "
                      "Stopping logging".format(self._pid))
                break
            # Append or init output data
            proc_stats = ps_info_wide[idx]
            # Init ps keys if first pass
            if "sample_time" not in self._stats:
                for name, v in proc_stats.items():
                    if name in _CONST_FIELDS:
                        self._stats[name] = v
                    else:
                        self._stats[name] = [v]
                self._start_time = _t0  # Init relative time reference
                self._stats["sample_time"] = [0]
            else:
                self._stats["sample_time"].append(_t0 - self._start_time)
                for name in proc_stats.keys():
                    if name in _CONST_FIELDS:
                        continue
                    self._stats[name].append(proc_stats[name])

            # Replaces `time.sleep`. Return False only in case of timeout
            # Using max(0, ...) if something takes too long.
            _dt = time.time() - _t0
            if self._stop_trigger.wait(timeout=max(0, self._interval - _dt)):
                self._save_output()
                break

    def _save_output(self):
        if self._fname:
            with open(self._fname, "w") as outf:
                json.dump(self._stats, fp=outf, indent=2)

    def stop(self, *args):
        """
        Stops the thread on next occasion.
        args only to make it `signal` compatible.
        """
        self._stop_trigger.set()
        return self._stats

    def tag(self, tagname):
        """
        Create a tag at the current timestamp. Use as
        ```
        logger.tag("start loading data")
        load_data()
        logger.tag("done loading data")
        ```
        to connect logged data with program events (eg. RAM spikes).

        Parameters
        ----------
        tagname : str
            Tag.
        """
        if not self.is_alive():
            raise RuntimeError("The logger is not running.")
        if "sample_time" in self._stats:
            self._stats["tags"].append(
                (time.time() - self._start_time, tagname))
        else:  # Before first log point, use 0 as sample time
            self._stats["tags"].append((0, tagname))


def _save_output(output, fname):
    with open(fname, "w") as outf:
        json.dump(output, fp=outf, indent=2)
        print("Saved output to '{}'".format(fname))


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="Process ID to watch.")
    parser.add_argument(
        "-o", "--outf", type=str, required=True,
        help="Filename of the output file. Data is stored in JSON format.")
    parser.add_argument(
        "-n", "--interval", type=int, default=1,
        help="Integer time interval in seconds to sample `ps`. Default: 1.")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="If given, print message for each sampling.")
    args = parser.parse_args()

    # Check input args
    if args.interval < _INTERVAL_MIN:
        sys.exit("Interval must be >= 0.1 second.")

    path = os.path.dirname(os.path.expandvars(os.path.expanduser(args.outf)))
    if not path:
        path = "."
    if not os.path.isdir(path):
        raise ValueError("Output directory '{}' does not exist.".format(path))
    fname = os.path.basename(args.outf)
    if not fname:
        raise ValueError("Given filename is invalid.")
    fname = os.path.join(path, fname)
    if not fname.endswith(".json"):
        fname += ".json"

    stats_out = {"tags": []}  # Output data
    while True:
        try:
            _t0 = time.time()

            # Sample ps stats
            ps_info_narrow, ps_info_wide = sample_ps()
            # Find desired PID
            try:
                pids = ps_info_narrow["PID"]
            except ValueError as err:
                _save_output(stats_out, fname)
                sys.exit(err)
            except KeyError:
                _save_output(stats_out, fname)
                sys.exit("No PID information in parsed output.")
            try:
                idx = pids.index(args.pid)
            except ValueError:
                _save_output(stats_out, fname)
                sys.exit("Requested PID '{}' "
                         "not in `ps` output.".format(args.pid))
            # Append or init output data
            proc_stats = ps_info_wide[idx]
            # Init ps keys if first pass
            if "sample_time" not in stats_out:
                for name, v in proc_stats.items():
                    if name in _CONST_FIELDS:
                        stats_out[name] = v
                    else:
                        stats_out[name] = [v]
                start_time = _t0  # Init relative time reference
                stats_out["sample_time"] = [0]
            else:
                stats_out["sample_time"].append(_t0 - start_time)
                for name in proc_stats.keys():
                    if name in _CONST_FIELDS:
                        continue
                    stats_out[name].append(proc_stats[name])

            _dt = time.time() - _t0
            if args.verbose:
                print(
                    "{:8.1f}s : Successfully sampled `ps`, next sampling "
                    "in {:.3f}s".format(
                        stats_out["sample_time"][-1], args.interval - _dt))
            # Using max(0, ...) if something takes too long
            time.sleep(max(0, args.interval - _dt))
        except KeyboardInterrupt:
            _save_output(stats_out, fname)
            sys.exit("Programm ended by user.")
        except Exception as err:
            _save_output(stats_out, fname)
            sys.exit("Unknown error occured, exiting: {}".format(err))


if __name__ == "__main__":
    _main()
