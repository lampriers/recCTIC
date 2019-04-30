import os
import sys
import shutil
import errno

import pandas as pd


def prs(*args):
    st = ""
    for s in args:
        st += str(s)
    print(st)

# def prt(*args):
#     d=locals().items()
#     st = ""
#     for s in args:
#         for k,v in d:
#             if id(s)==id(v):
#                 st+=str(k)+"="+str(v)+" "
#     print(st)

def printnorm(self, input, output):
    #if DiffusionModel2.verbose > 0:
        # input is a tuple of packed inputs
        # output is a Variable. output.data is the Tensor we are interested
        print('Inside ' + self.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(input))


        for i in range(len(input)):
            print('input[' + str(i) + ']: ', type(input[i]))
        print('output: ', type(output))
        print('')
        for i in range(len(input)):
            if input[i] is not None:
                print('input[' + str(i) + '] size and norm: ', input[i].size(), ' ', input[i].data.float().norm())
        print('output size:', output.data.size())
        print('output norm:', output.data.norm())


def printgradnorm(self, grad_input, grad_output):
    #if DiffusionModel2.verbose > 1:
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        for i in  range(len(grad_input)):
            print('grad_input['+str(i)+']: ', type(grad_input[i]))
        print('grad_output: ', type(grad_output))
        for i in  range(len(grad_output)):
            print('grad_output['+str(i)+']: ', type(grad_output[i]))
        print('')
        for i in  range(len(grad_input)):
            if grad_input[i] is not None:
                print('grad_input['+str(i)+'] size and norm: ', grad_input[i].size(),' ',grad_input[i].data.float().norm())
        for i in  range(len(grad_output)):
            if grad_output[i] is not None:
                print('grad_output['+str(i)+'] size and norm: ', grad_output[i].size(),' ',grad_output[i].data.float().norm())





class LogCSV(object):
    def __init__(self, root_dir, epoch_size=1, filename='log.csv', wipe=True):
        super(LogCSV, self).__init__()
        self.root_dir = root_dir
        self.filename = os.path.join(self.root_dir, filename)
        self.logs = pd.DataFrame()
        self.it = 0
        self.epoch_size = epoch_size
        self.flushed = 0

        if os.path.isdir(self.root_dir) and wipe:
            do_wipe = input("Log file already exists. Wipe ? (y|n)")

            if do_wipe != 'y':
                print("Quit")
                sys.exit(0)
            else:
                shutil.rmtree(self.root_dir)
        try:
            os.makedirs(self.root_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e

    def new_iteration(self):
        self.it += 1

    def log(self, name, value):
        self.logs.loc[self.it, name] = value

    def log_dict(self, d):
        for k, v in d.items():
            self.logs[k] = v

    def get(self, name):
        return self.logs.loc[self.it, name]

    def get_epoch(self, name):
        if self.epoch_size == 1:
            return self.get(name)
        return self.logs.loc[self.it - self.epoch_size+1:self.it, name].sum()

    def flush(self):
        # Code from KCzar
        # url : http://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file
        if self.flushed > 0 and not os.path.isfile(self.filename):  # handle potential network bugs
            return
        if not os.path.isfile(self.filename):
            self.logs.to_csv(self.filename, mode='a', index=False, sep=',')
        elif len(self.logs.columns) != len(pd.read_csv(self.filename, nrows=1, sep=',').columns):
            raise Exception("Columns do not match")
        elif not (self.logs.columns == pd.read_csv(self.filename, nrows=1, sep=',').columns).all():
            raise Exception("Columns and column order of dataframe and csv file do not match")
        else:
            self.logs.to_csv(self.filename, mode='a', index=False, sep=',', header=None)
        #print(str(self.logs.loc))
        #print(str(self.logs.index))

        self.logs.drop(self.logs.index, inplace=True)
        self.flushed += 1
