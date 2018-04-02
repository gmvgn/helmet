# Use heading and gyro_x for turning
# Gyro_x 0 - 40 - 0 on turn
# Right turn: high to low heading
# Left turn: low to high heading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import json
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.signal import butter, lfilter, freqz
from sklearn.preprocessing import MinMaxScaler


df_columns = [
    'MS',
    'Gyro_x', 'Gyro_y', 'Gyro_z', 
    'Accel_x', 'Accel_y', 'Accel_z', 
    'Mag_x', 'Mag_y', 'Mag_z', 
    'Pitch', 'Roll', 'Heading'
]

class DataPlot:
    def __init__(self, file):
        self.data_file = file
        self.base_name = ""
        self.ensure_csv()
        self.plot_dir = self.get_plot_dir()
        self.df = self.get_main_data_frame()

    def file_get_contents(self, filename):
        with open(filename) as f:
            return f.read()

    def ensure_csv(self):
        name = self.data_file.split('/')[-1]
        path = self.data_file.replace(name, '')
        base_name, exten = name.split('.')
        self.base_name = base_name

        # Format to CSV from data logger
        if (exten != 'csv'):
            csv_file = path + base_name + ".csv"
            if (os.path.isfile(csv_file) == False):
                print("\nConverting to CSV...\n")
                contents = self.file_get_contents(self.data_file)
                # Trim
                contents = re.sub(r'^\s*|\s*$', '', contents)
                # Single line
                contents = re.sub("\n", ',', contents)
                # Rows
                contents = re.sub(r',?-----,?', "\n", contents)
                # Trim 2
                contents = re.sub(r'^\s*|\s*$', '', contents)

                file = open(csv_file, "w")
                file.write(contents)
                file.close()

            self.data_file = csv_file


    def get_plot_dir(self):
        plot_dir = "plots/{}".format(self.base_name)
        print("Plot directory: {}".format(plot_dir))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        return plot_dir

    def get_main_data_frame(self):
        print("Reading dataset...")
        df = pd.read_csv(self.data_file)
        df.columns = df_columns
        print("Finished read")
        return df

    def start(self):
        print("\nPlotting")
        rows = len(self.df)
        x_vals = self.df['MS']
        for col in self.df.columns:
            plt.plot(x_vals, self.df[col])
            fig = plt.gcf()
            fig.set_size_inches(20, 10.5)
            plt.savefig("{}/{}.png".format(self.plot_dir, col))
            plt.close()
        print("Done\n")


class ML:
    def __init__(self, file):
        self.json_data = json.load(open(file))
        self.train_cols = self.json_data['train_columns']
        self.plot_cols = self.json_data['plot_columns']
        self.kernels = self.json_data['kernels']
        self.ml_dir = "ml/{}/".format(self.json_data['dir'])
        self.scale_vals = self.json_data['scale_vals']

        self.data_cols = df_columns
        self.lbl_cols = ['label', 'dataset']
        self.use_cols = self.data_cols + self.lbl_cols
        print("\n----------------------------------")
        print("ML TRAINING\n\n")
        print("Dir: ", self.ml_dir)
        print("Train on: ", self.train_cols)
        print("Plot with: ", self.plot_cols)
        print("Kernels: ", self.kernels)

    def get_data(self):
        combined_df = pd.DataFrame([], columns=self.use_cols)
        i = 0
        for file_data in self.json_data['training']:
            df = pd.read_csv(file_data['file'])
            df.columns = df_columns
            labled_df = df[df['MS'] > file_data['start']].copy()
            labled_df.loc[:,'label'] = 0
            labled_df.loc[:,'dataset'] = i
            for stop in file_data['stops']:
                begin, end = stop
                labled_df.loc[(labled_df['MS'] > begin) & (labled_df['MS'] < end), 'label'] = 1

            combined_df = pd.concat([
                combined_df, 
                labled_df
            ])
            i += 1
        return combined_df

    def start(self):
        print("Selecting data...")
        combined_df = self.get_data()
        print("Finished")

        print("Training classifier...")
        self.do_training(combined_df)
        print("\nFinished\n")


    def do_training(self, df):
        train_x = df[self.train_cols].as_matrix()
        # train_x = self.minmax_matrix(train_x)
        train_y = df['label'].values.astype('int')
        datasets = df['dataset'].unique()

        for kernel in self.kernels:
            print("\n-----------------")
            print("Kernel: {}".format(kernel))

            if (kernel == "logistic"):
                clf = LogisticRegression()
            else:
                clf = SVC(kernel=kernel)
            clf.fit(train_x, train_y)
            y_pred = clf.predict(train_x)
            
            print("Score: ", accuracy_score(train_y, y_pred))

            if (kernel in ['linear', 'logistic']):
                print("Weights: ", clf.coef_)
                print("Intercept: ", clf.intercept_)
                # Linear logistic prediction function
                total, correct = len(train_x), 0
                for i, row in enumerate(train_x):
                    row_sum = np.sum( np.multiply(row, clf.coef_) ) + clf.intercept_[0]
                    pred = 1 if row_sum >= 0 else 0
                    correct += 1 if pred == train_y[i] else 0
                print("Prediction test: ", correct / total)

            print()

            for di in datasets:
                plot_name = "{}{}_train_{}.png".format(self.ml_dir, kernel, di)
                dfi = df[df['dataset'] == di]

                self.plot_predictions(dfi, clf, plot_name)

            test_i = 0
            for file_data in self.json_data['testing']:
                plot_name = "{}{}_test_{}.png".format(self.ml_dir, kernel, test_i)
                df_test = pd.read_csv(file_data['file'])
                df_test.columns = df_columns
                test_i += 1

                self.plot_predictions(df_test, clf, plot_name)

    def plot_predictions(self, df, clf, plot_name):
        print(plot_name)
        x = df[self.train_cols].as_matrix()
        # x = self.minmax_matrix(x)
        y = clf.predict(x)

        for col in self.plot_cols:
            vals = self.scale_filter(df[col].values) if self.scale_vals else df[col].values
            plt.plot(df['MS'], vals)
        plt.scatter(df['MS'], y, color='y')

        plt.gcf().set_size_inches(20, 10.5)
        plt.savefig(plot_name)
        plt.close()

    def mod_avg(self, values, reads=8):
        new_y = [0] * len(values)
        count, avg, csum = 0, 0, 0
        for i, v in enumerate(values):
            count += 1
            csum += v
            if (count == reads):
                avg = csum / count
                csum = count = 0
            new_y[i] = avg
        return new_y

    def mod_lowpass(self, values, RC=None):
        dt = 100
        if (RC == None):
            RC = dt
        y = [0] * len(values)
        alpha = dt / (dt + RC)
        y[0] = alpha * values[0]
        for i in range(1, len(values)):
            y[i] = alpha * values[i] + (1 - alpha) * y[i - 1]
        return y

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def scale_filter(self, data, new_min=0, new_max=1):
        m, M = np.min(data), np.max(data)
        d = M - m
        std = [ (v - m) / d for v in data ]
        return std

    def minmax_matrix(self, data):
        scaler = MinMaxScaler()
        x_prime = scaler.fit_transform(data)
        return x_prime

if __name__ == "__main__":
    if (len(sys.argv) == 3 and sys.argv[1] == 'plot'):
        inst = DataPlot(sys.argv[2])
        inst.start()
    elif (len(sys.argv) == 3 and sys.argv[1] == 'ml'):
        inst = ML(sys.argv[2])
        inst.start()
    else:
        print("Missing required parameters")

