class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, *args):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            print(*args, file=f)
            print(*args)

    def log_args(self, args):
        for eachArg, value in args.__dict__.items():
            self.log(eachArg + ' : ' + str(value))