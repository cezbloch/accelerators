from __future__ import division, absolute_import, print_function
import pyopencl as cl
import pyopencl.characterize.performance as perf
from six.moves import range


def main():
    platform = cl.get_platforms()[0]

    ctx = cl.Context(
        dev_type=cl.device_type.ALL, 
        properties=[(cl.context_properties.PLATFORM, platform)])    

    prof_overhead, latency = perf.get_profiling_overhead(ctx)
    print("command latency: %g us" % latency)
    print("profiling overhead: %g us -> %.1f %%" % (
            prof_overhead / 1000000, (100*prof_overhead/latency)/1000000))
    queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print("empty kernel: %g s" % perf.get_empty_kernel_time(queue))
    print("float32 add: %g GOps/s" % (perf.get_add_rate(queue)/1e9))

    for tx_type in [
            perf.HostToDeviceTransfer,
            perf.DeviceToHostTransfer,
            perf.DeviceToDeviceTransfer]:
        print("----------------------------------------")
        print(tx_type.__name__)
        print("----------------------------------------")

        print("latency: %g s" % perf.transfer_latency(queue, tx_type))
        for i in range(6, 31, 2):
            bs = 1 << i
            try:
                result = "%g GB/s" % (perf.transfer_bandwidth(queue, tx_type, bs)/1e9)
            except Exception as e:
                result = "exception: %s" % e.__class__.__name__
            print("bandwidth @ %d bytes: %s" % (bs, result))

if __name__ == "__main__":
    main()