# coding=utf-8
# Copyright 2018 Majid Mortazavi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd

__author__ = "Majid Mortazavi"
__email__ = "maj.mortazavi@gmail.com"

def read_parquet_to_df(path, filename):

    df = pd.read_parquet(os.path.join(path, filename),engine='pyarrow')
    # df = df.drop(['review_id','user_id','business_id','date','useful','funny','cool'],axis=1)

    return df

def read_json_to_df(path, filename, chunk_size):

    if chunk_size is not None:
        data = pd.read_json(os.path.join(path, filename), lines=True, chunksize = chunk_size)
        df = next(data)
    else:
        data = pd.read_json(os.path.join(path, filename), lines=True)
        df = data.read()

    return df