import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from imblearn.over_sampling import SMOTE


class Feature_selection:
    '''This class is used to select only features
        of samples written in a csv file.
    
        Attributes:

        - EXTENTION : the extention you're allow to deal with
        - FILEPATH_DATA : path to the file
        - DATA_FRAME : Pandas data frame containing all data including labels
        - DATA : Pandas data frame containing only data (no labels)
        - LABEL : Pandas containing only labels (no data) 
    '''

    EXTENTION = '.csv'
	
    def __init__(self, filepath_data, output_data_file=None, output_label_file=None): 
        self.FILEPATH_DATA = filepath_data

        # If the file has a correct extention
        if self._check_file():
            self._open_read_file()
            self._process_file()

    def _check_file(self):
        '''Check if the file is a .csv
        '''
        if Path(self.FILEPATH_DATA).suffix == self.EXTENTION:
            return True
        else:
            return False			

    def _open_read_file(self):
        '''Open and store the content of a csv files
        '''
        self.DATA_FRAME = pd.read_csv(self.FILEPATH_DATA)

    def _process_file(self):
        '''Make all the modifications or statistiques
            then save the file
        '''
        self._city_acc_year()
        self._killer_month()
        self._damage_to_int()
        self._split_data_label()
 
    def _city_acc_year(self):
        '''Count and display 
            the number of accident per year in a city
        '''
        df = self.DATA_FRAME
        group_city_year = df.groupby(['agg','an'])
        city_year = group_city_year.count()
        
        # Display
        city_year.mois.plot.bar(by='mois')
        plt.xlabel('1 = non city , 2 = city , 15 = 2015, 16 = 2016')
        plt.ylabel('Nb Accident')
        plt.title('Nb accident per year in/out cities')
        plt.show()

    def _killer_month(self):
        '''Count and display
            the number of accident per month
        '''
    
        df = self.DATA_FRAME
        month_death = df[df.gravite == 'Mortel'].mois.value_counts()
        
        # Disay
        month_death = month_death.sort_values()
        month_death.plot.bar()
        plt.xlabel('Month in number')
        plt.ylabel('Nb of death')
        plt.title('Mortal accident per month (2015-2016)')

        plt.show()

    def _damage_to_int(self):
        '''Change: 
            - Grave = 1
            - Leger/Indemnes = 0
            - Mortel = 0
        '''
        df = self.DATA_FRAME
        mapping = {'Grave': 1, 'Leger/Indemnes': 0, 'Mortel': 0}
        self.DATA_FRAME = df.replace(mapping)

    def _split_data_label(self):
        '''Split data and frame
        '''
        df = self.DATA_FRAME
        self.DATA =  df.drop(['gravite', 'Num_Acc', 'lat', 'long', 'adr', 'hrmn', 'jour'], axis=1)
        self.LABEL = df['gravite']
        print(self.DATA)
   
    def balance_class_smote(self, X_train, y_train):
        '''Use SMOTE to balance classes
        '''
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_sample(X_train, y_train)
        return X_train_bal, y_train_bal
 
    def get_data(self):
        '''Pandas frame containing the data
        '''
        return self.DATA.as_matrix()

    def get_label(self):
        '''Pandas frame containing the label
        '''
        return self.LABEL.as_matrix()
