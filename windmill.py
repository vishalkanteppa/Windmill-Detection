# Databricks notebook source
# MAGIC %pip install laspy[laszip]
# MAGIC %pip install matplotlib
# MAGIC %pip install pyspark_dist_explore

# COMMAND ----------

import sys, os
import laspy
import numpy as np
import pandas as pd
from functools import partial, reduce
import logging
import random
import time
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType,StructField, FloatType, StringType, IntegerType
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
sys.path.append('dbfs/mnt/lsde/datasets/ahn3/')
# sys.stdout.fileno = lambda: False  # idk its necessary if you want to enable faulthandler https://github.com/ray-project/ray/issues/15551
# faulthandler.enable()
# las = laspy.read('/dbfs/mnt/lsde/datasets/ahn3/C_01CZ1.LAZ')

# COMMAND ----------

def convert_mount_to_regular_path(mount_path):
    mount_path_prefix = "dbfs:/"
    if(not mount_path.path.startswith(mount_path_prefix)):
        raise ValueError(f"didn't receive mount path: {mount_path}")
    return "/dbfs/" + mount_path.path[len(mount_path_prefix):]

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

# ahn3_file_infos = get_all_laz("dbfs:/mnt/lsde/datasets/ahn3/")
# ahn2_file_infos = get_all_laz("dbfs:/mnt/lsde/datasets/ahn2/")

# COMMAND ----------

schema_1 = StructType([ \
    StructField("x", FloatType(), True), \
    StructField("y", FloatType(), True), \
    StructField("z", FloatType(), True), \
    StructField("path", StringType(), True), \
])
schema_2 = StructType([ \
    StructField("X", IntegerType(), True), \
    StructField("Y", IntegerType(), True), \
    StructField("Z", IntegerType(), True), \
    StructField("intensity", IntegerType(), True), \
    StructField("bit_fields", IntegerType(), True), \
    StructField("raw_classification", IntegerType(), True), \
    StructField("scan_angle_rank", IntegerType(), True), \
    StructField("user_data", IntegerType(), True), \
    StructField("point_source_id", IntegerType(), True), \
    StructField("gps_time", FloatType(), True), \
    StructField("file_path", StringType(), True) \
])
ahn2_ahn3_factor = 41432/1375
schemas = {
    1: schema_1,
    2: schema_2,
}

def load_map(f_path, n_point_sample, n_chunk_iterator, names):
    dfs = []
    with laspy.open(f_path) as f:
        if n_point_sample > f.header.point_count:
            n_point_sample = f.header.point_count
        n_points_read = 0 
        mask_offset = 0
        sample = np.random.choice(f.header.point_count, n_point_sample, replace=False)
        mask = np.full(f.header.point_count, False)
        mask[sample] = True
        for points in f.chunk_iterator(n_chunk_iterator):
            current_mask = mask[n_points_read:n_points_read+len(points)]
            df = pd.DataFrame(points.array[current_mask])
            dfs.append(df)
            n_points_read += len(points)
            if(n_points_read >= n_point_sample):
                break
                
    df = pd.concat(dfs, ignore_index=True, copy=False)
    df['file_path'] = f_path
    if not 'gps_time' in df.columns:
        df['gps_time'] = pd.Series([], dtype='float64')
    df = df[names]
    return df.to_numpy().tolist()

def create_df(f_paths, schema, n_f_paths, n_point_sample, n_chunk_iterator):
    if(n_f_paths):
        f_paths_sampled = random.sample(f_paths, n_f_paths)
    else:
        f_paths_sampled = f_paths
    files_rdd = sc.parallelize([convert_mount_to_regular_path(f_path) for f_path in f_paths_sampled])
    points_rdd = files_rdd.flatMap(partial(load_map, n_point_sample=n_point_sample, n_chunk_iterator=n_chunk_iterator, names=schema.names))
    df_points = spark.createDataFrame(points_rdd, schema)
    return df_points


def create_and_store_or_load_df(which, f_paths, n_f_paths, n_point_sample, n_chunk_iterator, schema=2):
    df_points_path = f"/mnt/lsde/group02/df_points_{which}_files-{n_f_paths}_samples-{n_point_sample}_schema-{schema}.parquet"
    try:
        dbutils.fs.ls(df_points_path)
    except Exception as err:
        df_points = create_df(f_paths, schemas[schema], n_f_paths, n_point_sample, n_chunk_iterator)
        df_points.write.parquet(df_points_path)
        return df_points
    df_points = spark.read.parquet(df_points_path)
    return df_points

def create_and_store_df(which, f_paths, n_f_paths, n_point_sample, n_chunk_iterator, schema=2):
    df_points_path = f"/mnt/lsde/group04/{which}_files-{n_f_paths}_no_of_samples-{n_point_sample}-{schema}.parquet"
    df_points = create_df(f_paths, schemas[schema], n_f_paths, n_point_sample, n_chunk_iterator)
    df_points.write.mode("overwrite").parquet(df_points_path)
    df_points = spark.read.parquet(df_points_path)
    return df_points

def create_and_store_or_load_dfs(ahn2_f_paths, ahn3_f_paths, n_f_paths, n_point_sample=10_000_000, n_chunk_iterator=2_000_000):
    start_time = time.time()
#     df_points_ahn2 = create_and_store_or_load_df("ahn2", ahn2_f_paths, n_f_paths, n_point_sample, n_chunk_iterator)
#     df_points_ahn3 = create_and_store_or_load_df("ahn3", ahn3_f_paths, n_f_paths, n_point_sample, n_chunk_iterator)
    df_points_ahn2 = create_and_store_df("ahn2", ahn2_f_paths, int(n_f_paths * ahn2_ahn3_factor), int(n_point_sample / ahn2_ahn3_factor), n_chunk_iterator)
    df_points_ahn3 = create_and_store_df("ahn3", ahn3_f_paths, n_f_paths, n_point_sample, n_chunk_iterator)
    end_time = time.time()
    print(f"took {end_time - start_time:.2f} for n_f_paths={n_f_paths} n_point_sample={n_point_sample}")  # this isn't very accurate/comparable if loading from disk
    return df_points_ahn2, df_points_ahn3

# COMMAND ----------

ahn3_file_infos = get_all_laz("dbfs:/mnt/lsde/datasets/ahn3/")
ahn2_file_infos = get_all_laz("dbfs:/mnt/lsde/datasets/ahn2/")

# COMMAND ----------

df_points_ahn2_1, df_points_ahn3_1 = create_and_store_or_load_dfs(ahn2_file_infos, ahn3_file_infos, 120)
# df_points_ahn2_1.describe().show(), df_points_ahn3_1.describe().show()

# COMMAND ----------

df_points_ahn2_1.describe().show(), df_points_ahn3_1.describe().show()

# COMMAND ----------

def convert(file):
    image=laspy.read(file)
    return image.X

# COMMAND ----------

file=random.choice(ahn2_file_infos)
file2=random.choice(ahn2_file_infos)
print(file)
files_rdd = sc.parallelize([convert_mount_to_regular_path(file_info.path) for file_info in ahn2_file_infos])

image2=laspy.read(convert_mount_to_regular_path(file2.path))
data=sc.parallelize(np.stack([image2.X, image2.Y, image2.Z, image2.classification], axis=0).transpose((1, 0)))

for file in files_rdd.collect():
    image=laspy.read(file)
    temp=sc.parallelize(np.stack([image.X, image.Y, image.Z, image.classification], axis=0).transpose((1, 0)))
    data=data.union(temp)
# image=laspy.read(convert_mount_to_regular_path(file.path))
# temp=sc.parallelize(np.stack([image.X, image.Y, image.Z, image.classification], axis=0).transpose((1, 0)))
# data=data.union(temp)
# data.collect()    

    
    
# temp = files_rdd.map(lambda x: convert(x))

    

# COMMAND ----------

data.collect()

# COMMAND ----------

a=[[1,2],[3,4],[10,11]]
b=[[5,6],[7,8]]
a=sc.parallelize(a)
b=sc.parallelize(b)
a=a.union(b)
a.collect()

# COMMAND ----------

def get_data_from_file(file_path,n_files=10):
#     file_names=random.sample(file_names,n_files)
#     data=[]
    image=laspy.read(file_path)
    print('-----',image)
    point_data = np.stack([image.X, image.Y, image.Z, image.classification], axis=0).transpose((1, 0))
    return point_data
    
#     for file in file_names:
#         print(file.path)
#         try:
#             image=laspy.read(file.path)
#         except Exception as e:
#             print(e)
#             return
#         point_data = np.stack([image.X, image.Y, image.Z, image.classification], axis=0).transpose((1, 0))
#     print(point_data)

file_paths = []
n_files=10

n_file_info = random.sample(ahn2_file_infos,n_files)
for file in n_file_info:
    file_paths.append(convert_mount_to_regular_path(file.path))

file_rdd = sc.parallelize(file_paths)
# print(file_rdd.collect())
points_rdd = file_rdd.flatMap(load_map(n_point_sample=n_point_sample, n_chunk_iterator=n_chunk_iterator))


# get_data_from_file(ahn2_file_infos)

# COMMAND ----------

df_points = spark.createDataFrame(x, schema_1)
df_points.write.mode("overwrite").parquet('/mnt/lsde/group04/test.parquet')
test=spark.read.parquet("/mnt/lsde/group04/test.parquet")
print(test)

# COMMAND ----------

n_point_sample = 2_000_000
n_chunk_iterator = 1_000_000
dtype = [('X', '<i4'), ('Y', '<i4'), ('Z', '<i4'), ('intensity', '<u2'), ('bit_fields', 'u1'), ('raw_classification', 'u1'), ('scan_angle_rank', 'i1'), ('user_data', 'u1'), ('point_source_id', '<u2'), ('gps_time', '<f8')]
xyz = np.empty(n_point_sample, dtype=dtype)  # we might not completely fill up this xyz if our sample is larger than the number of points in the current file
n_point_collected = 0
f_path = '/dbfs/mnt/lsde/datasets/ahn2/tileslaz/tile_0_2/ahn_017000_363000.laz'
with laspy.open(f_path) as f:
    # im not sure how points are stored within a laz file e.g. all points with low height first which would make this sample kinda biased
    # we could do some math to figure out how much per chunk we need if we think this is important
    # we need this chunk iterator as the workers seems to crash quite often due to running out of memory otherwise
    for points in f.chunk_iterator(n_chunk_iterator):
        start = n_point_collected
        n_point_collected += len(points)
        
        if not 'gps_time' in [dim.name for dim in points.point_format.dimensions]:
            print(type(f))
            print(type(f.chunk_iterator))
            print(type(points))
#             points.add_extra_dim(laspy.ExtraBytesParams(name="gps_time", type="<f8"))
        print('---------')
        print(f.chunk_iterator)
        points_current = points.array
        print(points_current.dtype)
        # I dont think we actually need all this logic if n_point_sample is a multiple of n_chunk_iterator but it seemed more natural to not make them dependent
        if n_point_collected > n_point_sample:
            leftover = (n_point_sample % n_chunk_iterator)
            xyz[start:(start + leftover)] = points_current[:leftover]
            break
        else:
            leftover = (n_point_collected % n_chunk_iterator)
            leftover = n_chunk_iterator if leftover == 0 else leftover
#             xyz[start:(start + leftover)] = points_current

print([[*xyz, f_path] for xyz in xyz[:n_point_collected].tolist()])

# COMMAND ----------

def create_and_store_or_load_df(which, file_infos, n_file_info, n_point_sample, n_chunk_iterator):
    df_points_path = f"/mnt/lsde/group04/test.parquet"
    try:
#       check if file exists
        dbutils.fs.ls(df_points_path)
    except Exception as err:
        df_points = create_df(file_infos, n_file_info, n_point_sample, n_chunk_iterator)
        df_points.write.mode("overwrite").parquet(df_points_path)
        return df_points
    df_points = spark.read.parquet(df_points_path,schema_2)
    print(df_points)
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
df_points_ahn2_1.describe().show(), df_points_ahn3_1.show()

# COMMAND ----------

fig, ax = plt.subplots()
print(ax)
# df_points_ahn2_1.select('x').show()
column = 'classification'
df_list=list(df_points_ahn2_1.select(column).toPandas()[column])
# print(df_list)


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


