__all__ = ["RecordHook"]


class RecordHook(object):
    def __init__(self, total_step):
        self.total_step = total_step
        self.step = []
        self.loss = []
        self.time = []

    def reset(self):
        self.step = []
        self.loss = []
        self.time = []

    def update(self, current_step, loss_value, duration):
        self.step.append(current_step)
        self.loss.append(loss_value)
        self.time.append(duration)

    def __str__(self):
        if len(self.time) < 10:
            batches_per_sec = len(self.time) / sum(self.time)
        else:
            batches_per_sec = len(self.time[-10:]) / sum(self.time[-10:])

        remaining_time = int(1 / batches_per_sec * (self.total_step - self.step[-1]))
        hours = remaining_time // 3600
        remaining_time = remaining_time % 3600
        mins = remaining_time // 60
        secs = remaining_time % 60

        return '{:d}/{:d}   loss value: {:.4f}   speed: {:.2f} batches/s   ' \
               'time needed: {:d} hours {:d} mins {:d} secs'\
            .format(self.step[-1], self.total_step, self.loss[-1], batches_per_sec, hours, mins, secs)
