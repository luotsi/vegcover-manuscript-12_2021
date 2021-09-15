import os


'''
Utilities for trimming low-level log output from library dependencies.
'''

class suppress_stdout_stderr(object):
    '''
    From https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions:
    
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self, stdout_file=os.devnull, stderr_file=os.devnull):
        # Open a pair of null files
        flags = os.O_APPEND | os.O_WRONLY | os.O_CREAT
        self.null_fds = [os.open(stdout_file, flags), os.open(stderr_file, flags)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


