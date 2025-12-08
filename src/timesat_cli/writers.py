from __future__ import annotations
from typing import List, Tuple
import rasterio
import boto3
import os
from rasterio.session import AWSSession
from rasterio.windows import Window

__all__ = ["prepare_profiles", "write_vpp_layers", "write_st_layers"]


def prepare_profiles(img_profile, p_nodata: float, scale: float, offset: float):
    import copy
    img_profile_st = copy.deepcopy(img_profile)
    img_profile_st.update(compress='lzw')
    if scale != 1 or offset != 0:
        img_profile_st.update(dtype=rasterio.float32)

    img_profile_vpp = copy.deepcopy(img_profile)
    img_profile_vpp.update(nodata=p_nodata, dtype=rasterio.float32, compress='lzw')

    img_profile_ns = copy.deepcopy(img_profile)
    img_profile_ns.update(nodata=255, compress='lzw')
    return img_profile_st, img_profile_vpp, img_profile_ns


def write_vpp_layers(paths: List[str], arrays, window: Tuple[int, int, int, int], img_profile_vpp):
    x_map, y_map, x, y = window
    for i, arr in enumerate(arrays, 1):
        with rasterio.open(paths[i - 1], 'r+', **img_profile_vpp) as outvppfile:
            outvppfile.write(arr, window=Window(x_map, y_map, x, y), indexes=1)


def write_st_layers(paths: List[str], arrays, window: Tuple[int, int, int, int], img_profile_st):
    x_map, y_map, x, y = window
    for i, arr in enumerate(arrays, 1):
        with rasterio.open(paths[i - 1], 'r+', **img_profile_st) as outstfile:
            outstfile.write(arr, window=Window(x_map, y_map, x, y), indexes=1)


def get_img_profile_from_s3(s3_credentials, flist):
    """
    Retrieve the image profile from the first file in flist stored in S3.
    """
    session = boto3.Session(
        aws_access_key_id=s3_credentials.get("access_key"),
        aws_secret_access_key=s3_credentials.get("secret"),
    )
    endpoint = s3_credentials.get("endpoint")
    # remove https:// from the endpoint if it exists
    if endpoint.startswith("https://"):
        endpoint = endpoint[8:]
    bucket_name = s3_credentials.get("bucket_name")
    s3_path = os.path.join("s3://", bucket_name, flist[0])

    with rasterio.Env(
        AWSSession(session),
        AWS_S3_ENDPOINT=endpoint,
        AWS_HTTPS="YES",
        AWS_VIRTUAL_HOSTING="FALSE",
    ):
        with rasterio.open(s3_path) as temp:
            img_profile = temp.profile
    return img_profile
