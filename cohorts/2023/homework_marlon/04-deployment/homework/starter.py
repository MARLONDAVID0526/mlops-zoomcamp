import pickle
import pandas as pd
import sys
import numpy as np



def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model



def read_data(filename):

    categorical = ['PULocationID', 'DOLocationID']

    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df




def get_paths(month:int,  year:int, taxi_type:str ='yellow',) -> list: 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    return input_file, output_file

def transform_dataframe(df:pd.DataFrame, year:int, month:str):
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df = df[['ride_id', 'duration']]

    return df


def prepare_dictionaries(df: pd.DataFrame, dv, model):


    dicts = df.to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return dicts, y_pred


def save_results(df, output_file):
    df.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False)

    return None

def run():

    taxi_type = sys.argv[1]  # 'yellow'
    year = int(sys.argv[2])       #2022
    month = int(sys.argv[3])      #2
    input_file, output_file = get_paths(month=month, year=year, taxi_type=taxi_type,)
    print(input_file,output_file )

    df= read_data(input_file)

    df = transform_dataframe(df,year, month )

    dv, model = load_model()

    
    dicts, y_pred = prepare_dictionaries(df,dv, model )

    # Calculate the standard deviation of the predicted duration
    std_dev = np.mean(y_pred)

    # Print the standard deviation
    print("Mean of Predicted Duration:", std_dev)

    save_results(df, output_file=output_file)

if __name__ == '__main__':
    run()
