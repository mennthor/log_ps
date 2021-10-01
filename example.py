import os
import time

from log_ps import PSLogger, summary_plot_ram


if __name__ == "__main__":
    PID = os.getpid()
    print("Start logging PID {}".format(PID))
    log_fname = "./test_log.json"
    logger = PSLogger(PID, fname=log_fname, interval=0.1)
    logger.start()

    arr = []
    for i in range(5):
        print("Main allocating a few GB, pass {}".format(i))
        logger.tag("Pass {}, start allocating".format(i))
        arr.append([i for i in range(int(1e6))])  # Will be 250MB in the end
        time.sleep(0.5)  # Simulate longer allocation
        print("Main sleeps for 1s")
        logger.tag("Pass {}, sleeping 1s".format(i))
        time.sleep(1)

    print("Main is done")
    logger.tag("Done")
    print("Stopping thread")
    logs = logger.stop()

    print("Showing log results")
    summary_plot_ram(
        logs, plotname=log_fname.replace(".json", ".png"), ram_unit="MiB",
        axvline_lw=4, axvline_ls=":", plot_c="C7", cmap="copper",
        savefig_dpi=200)
