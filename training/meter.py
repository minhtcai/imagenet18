import subprocess
import time


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, average_momentum=0.5):
        self.average_momentum = average_momentum
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0  # running average of whole epoch
        self.smooth_average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n

        if self.count == 0:
            self.smooth_average = value
        else:
            self.smooth_average = self.average * self.average_momentum + value * (1 - self.average_momentum)

        self.average = self.sum / self.count


class TimeMeter:
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()


################################################################################
# Generic utility methods, eventually refactor into separate file
################################################################################
def network_bytes():
    """Returns received bytes, transmitted bytes."""
    proc = subprocess.Popen(['cat', '/proc/net/dev'], stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('ascii')

    recv_bytes = 0
    transmit_bytes = 0
    lines = stdout.strip().split('\n')
    lines = lines[2:]  # strip header
    for line in lines:
        line = line.strip()
        # ignore loopback interface
        if line.startswith('lo'):
            continue
        toks = line.split()

        recv_bytes += int(toks[1])
        transmit_bytes += int(toks[9])
    return recv_bytes, transmit_bytes

################################################################################
