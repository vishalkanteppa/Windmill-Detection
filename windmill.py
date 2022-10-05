# Databricks notebook source
# MAGIC %pip install laspy[laszip]
# MAGIC %pip install matplotlib
# MAGIC %pip install pyspark_dist_explore

# COMMAND ----------

import sys, os
import laspy
import numpy as np
import pandas as pd
import logging
import random
import time
import functools
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType,StructField, FloatType, StringType
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
sys.path.append('dbfs/mnt/lsde/datasets/ahn3/')
# sys.stdout.fileno = lambda: False  # idk its necessary if you want to enable faulthandler https://github.com/ray-project/ray/issues/15551
# faulthandler.enable()
# las = laspy.read('/dbfs/mnt/lsde/datasets/ahn3/C_01CZ1.LAZ')

# COMMAND ----------

schema = StructType([ \
    StructField("x", FloatType(), True), \
    StructField("y", FloatType(), True), \
    StructField("classification", FloatType(), True), \
    StructField("z", FloatType(), True), \
    StructField("path", StringType(), True), \
])
def get_all_laz(dir):
    file_infos = dbutils.fs.ls(dir)
    laz_file_infos = []
    while file_infos:
        file_info = file_infos.pop(0)
        if file_info.name[-1] == '/':
            [file_infos.append(to_check) for to_check in dbutils.fs.ls(file_info.path)]
        elif file_info.name[-4:].lower() == '.laz':
            laz_file_infos.append(file_info)
    return laz_file_infos

def convert_mount_to_regular_path(mount_path):
    mount_path_prefix = "dbfs:/"
    if(not mount_path.startswith(mount_path_prefix)):
        raise ValueError(f"didn't receive mount path: {mount_path}")
    return "/dbfs/" + mount_path[len(mount_path_prefix):]


def load_map(n_point_sample, n_chunk_iterator):
    def _load_map(f_path):
        xyz = np.empty((n_point_sample, 4), dtype=float)  
        n_point_collected = 0
        with laspy.open(f_path) as f:
            for points in f.chunk_iterator(n_chunk_iterator):
                start = n_point_collected
                points_current = np.column_stack((points.x, points.y, points.classification))
        print('-------',xyz)      
        return [[*xyz, f_path] for xyz in xyz[:n_point_collected,:].tolist()]
    return _load_map

def create_df(file_infos, n_file_info=None, n_point_sample=10_000_000, n_chunk_iterator=2_000_000):
    if(n_file_info):
        file_infos_sampled = random.sample(file_infos, n_file_info)
    else:
        file_infos_sampled = file_infos
    files_rdd = sc.parallelize([convert_mount_to_regular_path(file_info.path) for file_info in file_infos_sampled])
    points_rdd = files_rdd.flatMap(load_map(n_point_sample, n_chunk_iterator))
    df_points = spark.createDataFrame(points_rdd, schema)

    return df_points


def get_data_from_file(ahn,file_names,n_files=10):
    file_names=random.sample(file_names,n_files)
    for file in file_names:
        
ahn3_file_infos = get_all_laz("dbfs:/mnt/lsde/datasets/ahn3/")
ahn2_file_infos = get_all_laz("dbfs:/mnt/lsde/datasets/ahn2/")



# COMMAND ----------

ahn2_file_infos = get_all_laz("dbfs:/mnt/lsde/datasets/ahn2/")
files_rdd = sc.parallelize([convert_mount_to_regular_path(file_info.path) for file_info in ahn2_file_infos])
points_rdd = files_rdd.flatMap(load_map(10000000, 2500000))



# COMMAND ----------

points_rdd.collect()

# COMMAND ----------

def create_and_store_or_load_df(which, file_infos, n_file_info, n_point_sample, n_chunk_iterator):
    df_points_path = f"/mnt/lsde/group02/df_points_{which}_files-{n_file_info}_samples-{n_point_sample}.parquet"
    try:
#       check if file exists
        dbutils.fs.ls(df_points_path)
    except Exception as err:
        df_points = create_df(file_infos, n_file_info, n_point_sample, n_chunk_iterator)
        df_points.write.mode("overwrite").parquet(df_points_path)
        return df_points

    df_points = spark.read.parquet(df_points_path)
    return df_points

def create_and_store_or_load_dfs(ahn2_file_infos, ahn3_file_infos, n_file_info, n_point_sample=10_000_000, n_chunk_iterator=2_500_000):
    start_time = time.time()
    df_points_ahn2 = create_and_store_or_load_df("ahn2", ahn2_file_infos, n_file_info, n_point_sample, n_chunk_iterator)
    df_points_ahn3 = create_and_store_or_load_df("ahn3", ahn3_file_infos, n_file_info, n_point_sample, n_chunk_iterator)
    end_time = time.time()
    print(f"took {end_time - start_time:.2f} for n_file_info={n_file_info} n_point_sample={n_point_sample}") 
    return df_points_ahn2, df_points_ahn3

# COMMAND ----------

df_points_ahn2_1, df_points_ahn3_1 = create_and_store_or_load_dfs(ahn2_file_infos, ahn3_file_infos, 10)
# df_points_ahn2_1.describe().show(), df_points_ahn3_1.show()

# COMMAND ----------

fig, ax = plt.subplots()
print(ax)
# df_points_ahn2_1.select('x').show()
column = 'classification'
df_list=list(df_points_ahn2_1.select(column).toPandas()[column])
# print(df_list)


# COMMAND ----------

# sd_sum=0
# m_sum=0
# for i in df_list:
#     m_sum=m_sum+i
# mean=m_sum/len(df_list)

# for i in df_list:
#     sd_sum=sd_sum+(i-mean)**2
# sd=(sd_sum/len(df_list))**.5
# print(sd,mean)

# for i in range(len(df_list)):
# #     df_list[i]=(df_list[i]-mean)/sd
#     df_list[i] = ((1 / (np.sqrt(2 * np.pi) * sd)) * np.exp(-0.5 * ((1 / sd )* (df_list[i] - mean))**2))



# COMMAND ----------

n,bins,patches = plt.hist(df_list, color=['blue'],bins=15,density=False)
# y = ((1 / (np.sqrt(2 * np.pi) * sd)) * np.exp(-0.5 * ((1 / sd )* (bins - mean))**2))
# plt.plot(bins, y, '--', color ='red')
plt.xlabel("Different Classifications")
plt.ylabel("Count")
plt.show()

# COMMAND ----------

print(df_list)

# COMMAND ----------


