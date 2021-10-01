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


__all__ = ["sample_ps", "PSLogger", "load_log", "summary_plot_ram"]

_INTERVAL_MIN = 0.1  # Minimum sampling interval
# These are constant, no need to always append
_CONST_FIELDS = ["USER", "PID", "TT", "STARTED", "COMMAND"]


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
        Logging stats as dictionary.
    """
    with open(fname) as fp:
        logs = json.load(fp)
    return logs


def summary_plot_ram(log, plotname=None, ram_unit="MiB", **kwargs):
    """
    Creates a summary plot from a logging file or dict showing the RAM (RSS)
    usage over logged time.

    Parameters
    ----------
    log : str or dict
        If str, is interpreted as a logging filename and tries to load the log
        stats from there. Otherwise expects a logging dict directly.
    plotname : str or None, optional (default: None)
        If str, saves the plot under that filename using `plt.savefig`.
        Otherwise the plot is shown interactively with `plt.show()`.
    ram_unit : str, optional (default: 'MiB')
        Unit to plot the RSS usage in. Can be 'K(i)B', 'M(i)B', 'G(i)B'.
    kwargs : plot arguments
        These are passed to the plot instances via a prefix:
        - 'fig_<name>' is passed to `plt.subplots(1, 1, ...)`. Example:
          `fig_figsize(10, 6)` makes a larger figure.
        - 'plot_<name>' is passed to the time vs RSS plot line. Example:
          `plot_ls=':', plot_lw=4` makes the line dotted and wide.
        - 'axvline_<name>' is passed to the tag line axvline plot. Example:
          `axvline_c='k', axvline_ls='--'` makes them black and dashed.
          You can't specify 'axvline_label' because that is reserved for the
          tagnames in the log stats.
        - 'legend_<name>' is passed to the legend. Example:
          `legend_loc='upper right' places the legend.
        - 'savefig_<name>' is passed `plt.savefig(plotname, ...)`. Example:
          `savefig_dpi=200` for increasing the resolution.
        Others are:
        - 'cmap' to set the colormap from which the tag lines are taken if no
          explicit 'axvline_c' key is given.

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
        stats = load_log(log)
    elif isinstance(log, dict):
        stats = log
    else:
        raise TypeError("`log` must be either str or dict.")

    units = {"kb": 1.024, "kib": 1., "mb": 1.024e-3, "mib": 1. / 1024,
             "gb": 1.024e-6, "gib": 1. / 1024**2}
    if ram_unit.lower() not in units:
        raise ValueError("`ram_unit` can be {}".format(
            ", ".join([k for k in units])))

    rss = np.array(stats["RSS"], dtype=float) * units[ram_unit.lower()]
    times = np.array(stats["sample_time"], dtype=float)
    tags = stats["tags"]

    fig, ax = plt.subplots(
        1, 1, **{k.replace("fig_", ""): v for k, v in kwargs.items()
                 if k.startswith("fig_")})
    ax.plot(
        times, rss,
        **{k.replace("plot_", ""): v for k, v in kwargs.items()
           if k.startswith("plot_")})
    if "axvline_c" in kwargs:
        colors = len(tags) * [kwargs.pop("axvline_c")]
    else:
        colors = plt.get_cmap(kwargs.get("cmap", "brg"))(
            np.linspace(0, 1, len(tags)))
    for j, tag in enumerate(tags):
        ax.axvline(
            tag[0], 0, 1, c=colors[j], label=tag[1],
            **{k.replace("axvline_", ""): v for k, v in kwargs.items()
               if k.startswith("axvline_")})
    ax.set_xlabel("Time in s")
    _unit_str = ram_unit.upper().replace("I", "i")
    ax.set_ylabel("RAM in {}".format(_unit_str))
    if len(tags) > 0:
        ax.legend(**{k.replace("legend_", ""): v for k, v in kwargs.items()
                     if k.startswith("legend_")})
    if plotname is None:
        plt.show()
    else:
        fig.savefig(
            plotname, **{k.replace("savefig_", ""): v for k, v in kwargs.items()
                         if k.startswith("savefig_")})


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
                print("Logging stopped by calling `stop()`.")
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
