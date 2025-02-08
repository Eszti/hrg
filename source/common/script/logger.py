import os


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        if os.path.exists(log_file):
            os.remove(log_file)

    def log(self, line, to_stdout=False):
        new_line = f"{line}\n"
        with open(self.log_file, "a") as f:
            f.write(new_line)
        if to_stdout:
            print(line)
