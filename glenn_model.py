import numpy as np 
from math import *
import random
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

import matplotlib
matplotlib.style.use('ggplot')
pd.options.display.max_rows = 100

import warnings
warnings.filterwarnings('ignore')


########################################################
# Define variables which we will use later in the script
########################################################

parse_date_cols = ['dt_prediction_date', 'dt_target_date', 'dt_flight_date']
target_cols = ['num_pax_000_014_mins_before_sdt',
'num_pax_015_029_mins_before_sdt', 'num_pax_030_044_mins_before_sdt',
'num_pax_045_059_mins_before_sdt', 'num_pax_060_074_mins_before_sdt',
'num_pax_075_089_mins_before_sdt', 'num_pax_090_104_mins_before_sdt',
'num_pax_105_119_mins_before_sdt',  'num_pax_120_134_mins_before_sdt',
'num_pax_135_149_mins_before_sdt', 'num_pax_150_164_mins_before_sdt',
'num_pax_165_179_mins_before_sdt',  'num_pax_180_194_mins_before_sdt',
'num_pax_195_209_mins_before_sdt', 'num_pax_210_224_mins_before_sdt',
'num_pax_225_239_mins_before_sdt', 'num_pax_240plus_mins_before_sdt']


###############################################
# Read in csv file and parse dates. Also generate dataframe with the target
# cases ordered by id
###############################################
df = pd.read_csv("./train.csv", parse_dates = parse_date_cols)
df_target = pd.read_csv("./test.csv", parse_dates = parse_date_cols)
df_train=df[df['cat_case_type']=='Expl']


df_train['num_flight_month']=np.sin(np.pi*df_train['num_flight_month']/12)
df_train['num_flight_weekofyear']=np.sin(np.pi*df_train['num_flight_weekofyear']/52)
df_train['num_flight_dayofweek']=np.sin(np.pi*df_train['num_flight_dayofweek']/7)
df_train['cat_sdt_hour']=np.sin(np.pi*df_train['cat_sdt_hour']/24)

df_train['ord_leisure']=df_train['ord_leisure']/5.0
df_train['ord_irish_residents']=df_train['ord_irish_residents']/5.0
df_train['ord_trip_duration']=df_train['ord_trip_duration']/5.0
df_train['ord_irish_residents']=df_train['ord_irish_residents']/5.0
df_train['ord_female']=df_train['ord_female']/5.0
df_train['ord_party_size']=df_train['ord_party_size']/5.0
df_train['ord_bag_checkin']=df_train['ord_bag_checkin']/5.0
df_train['ord_arrive_by_car']=df_train['ord_arrive_by_car']/5.0

pax_list=["num_pax_000_014_mins_before_sdt","num_pax_015_029_mins_before_sdt",\
          "num_pax_030_044_mins_before_sdt","num_pax_045_059_mins_before_sdt",\
          "num_pax_060_074_mins_before_sdt","num_pax_075_089_mins_before_sdt",\
          "num_pax_090_104_mins_before_sdt","num_pax_105_119_mins_before_sdt",\
          "num_pax_120_134_mins_before_sdt","num_pax_135_149_mins_before_sdt",\
          "num_pax_150_164_mins_before_sdt","num_pax_165_179_mins_before_sdt",\
          "num_pax_180_194_mins_before_sdt","num_pax_195_209_mins_before_sdt",\
          "num_pax_210_224_mins_before_sdt","num_pax_225_239_mins_before_sdt",\
          "num_pax_240plus_mins_before_sdt"]

    
timelist=["num_flight_month","num_flight_weekofyear",\
          "num_flight_dayofweek","cat_sdt_hour"]
period=np.array([12,52,7,24])

                     # "cat_sdt_hour","cat_i_airport",\
                     # "cat_i_city","cat_destination_group_id","cat_longhaul_ind",\
                     # "cat_flight_class_type_id",\
            
            
cat_list=["cat_destination_group_id"]    
    
ord_list=["ord_leisure","ord_irish_residents","ord_trip_duration",\
          "ord_female","ord_party_size","ord_bag_checkin","ord_arrive_by_car"]

interesting_cols=["num_flight_month","num_flight_weekofyear",\
                  "num_flight_dayofweek",\
                  "ord_leisure","ord_irish_residents","ord_trip_duration",\
                  "ord_female","ord_party_size","ord_bag_checkin","ord_arrive_by_car"]

nn=200
cols=["id"]+pax_list
df_model=[]

def calculate_score(df_target_cases, df_predictions):
    '''Root-mean-squared error is the chosen error metric. This function
calculates and returns the root-mean-squared error'''
    f_rmse = np.sqrt(mean_squared_error(df_target_cases, df_predictions))
    return f_rmse


def find_nearest_neighbours(test_row,df_sample):
    #Need to filter for training_flight_dates before target_prediction_date
    X=df_sample[(df_sample['s_model_type']==test_row['s_model_type'])&\
         (df_sample['dt_flight_date']<test_row['dt_prediction_date'])]
    X=X[interesting_cols]
    
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(test_row[interesting_cols])

#    model=np.array(df_train.iloc[indices[0]][pax_list].mean())
    w=np.exp(-distances)
    w=w/sum(sum(w))

    model=np.average(df_sample.iloc[indices[0]][pax_list],axis=0,weights=w[0])
    model=pd.Series(model,index=pd.Series(target_cols))
    return model    


############################################################################
print("Start Time:",datetime.datetime.now().time())
df_model=df_target[df_target['cat_case_type']=='Target'].copy()
df_model[target_cols]=df_model.apply(lambda row : find_nearest_neighbours(row,df_train),axis=1)
df_model=pd.DataFrame(df_model, columns=cols).set_index('id')

df_model.to_csv("glenn_test2.csv")
print("End Time:",datetime.datetime.now().time())
############################################################################

print(calculate_score(df_target[df_target['cat_case_type']=='Target'][target_cols].fillna(0),df_model))
