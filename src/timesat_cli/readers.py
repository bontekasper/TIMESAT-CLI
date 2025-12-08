from __future__ import annotations

import datetime
import os
import re
import boto3
from botocore.config import Config
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from .qa import assign_qa_weight

try:
    import ray
except Exception:  # optional
    ray = None

__all__ = ["read_file_lists", "open_image_data"]


def _parse_dates_from_name(name: str) -> Tuple[int, int, int]:
    date_regex1 = r"\d{4}-\d{2}-\d{2}"
    date_regex2 = r"\d{4}\d{2}\d{2}"
    try:
        dates = re.findall(date_regex1, name)
        position = name.find(dates[0])
        y = int(name[position : position + 4])
        m = int(name[position + 5 : position + 7])
        d = int(name[position + 8 : position + 10])
        return y, m, d
    except Exception:
        try:
            dates = re.findall(date_regex2, name)
            position = name.find(dates[0])
            y = int(name[position : position + 4])
            m = int(name[position + 4 : position + 6])
            d = int(name[position + 6 : position + 8])
            return y, m, d
        except Exception as e:
            raise ValueError(f"No date found in filename: {name}") from e


def _read_time_vector(tlist: str, filepaths: List[str]):
    """Return (timevector, yr, yrstart, yrend) in YYYYDOY format."""
    flist = [os.path.basename(p) for p in filepaths]
    timevector = np.ndarray(len(flist), order="F", dtype="uint32")
    if tlist == "":
        for i, fname in enumerate(flist):
            y, m, d = _parse_dates_from_name(fname)
            doy = (datetime.date(y, m, d) - datetime.date(y, 1, 1)).days + 1
            timevector[i] = y * 1000 + doy
    else:
        with open(tlist, "r") as f:
            lines = f.read().splitlines()
        for idx, val in enumerate(lines):
            n = len(val)
            if n == 8:  # YYYYMMDD
                dt = datetime.datetime.strptime(val, "%Y%m%d")
                timevector[idx] = int(f"{dt.year}{dt.timetuple().tm_yday:03d}")
            elif n == 7:  # YYYYDOY
                _ = datetime.datetime.strptime(val, "%Y%j")
                timevector[idx] = int(val)
            else:
                raise ValueError(f"Unrecognized date format: {val}")

    yrstart = int(np.floor(timevector.min() / 1000))
    yrend = int(np.floor(timevector.max() / 1000))
    yr = yrend - yrstart + 1
    return timevector, yr, yrstart, yrend


def _unique_by_timevector(flist: List[str], qlist: List[str], timevector):
    tv_unique, indices = np.unique(timevector, return_index=True)
    flist2 = [flist[i] for i in indices]
    qlist2 = [qlist[i] for i in indices] if qlist else []
    return tv_unique, flist2, qlist2


def read_file_lists(
    tlist: str, data_list: str, qa_list: str
) -> Tuple[np.ndarray, List[str], List[str], int, int, int]:
    qlist: List[str] | str = ""
    with open(data_list, "r") as f:
        flist = f.read().splitlines()
    if qa_list != "":
        with open(qa_list, "r") as f:
            qlist = f.read().splitlines()
        if len(flist) != len(qlist):
            raise ValueError("No. of Data and QA are not consistent")

    timevector, yr, yrstart, yrend = _read_time_vector(tlist, flist)
    timevector, flist, qlist = _unique_by_timevector(flist, qlist, timevector)
    return (
        timevector,
        flist,
        (qlist if isinstance(qlist, list) else []),
        yr,
        yrstart,
        yrend,
    )


def list_files_S3_bucket(credentials, collection, prefix=None):
    # write code that via boto3 will extract the information
    # from the S3 bucket
    # intialize a connection to the S3 bucket
    s3 = boto3.client(
        "s3",
        aws_access_key_id=credentials["access_key"],
        aws_secret_access_key=credentials["secret"],
        endpoint_url=credentials["endpoint"],
        config=Config(signature_version="s3v4"),
    )
    # get the bucket name
    bucket_name = credentials["bucket_name"]
    if prefix is None:
        prefix = credentials["prefix"]
    
    # add collection name as subfolder to prefix
    if prefix:
        prefix = os.path.join(prefix, collection) + "/"
    else:
        prefix = collection + "/"
    paginator = s3.get_paginator("list_objects_v2")
    paginator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    return paginator


def find_file_in_S3_bucket(paginator, tile_name=None, year=None, VI=None,
                           file_extension=".tif"):
    """
    Search for the file in the S3 bucket based on the collection name and the LAEA tile name.

    Parameters:
    paginator (boto3.Paginator): The paginator object for the S3 bucket.
    tile_name (str): The name of the LAEA tile.
    year (str): The year for which the file is needed.
    VI (str): The vegetation index name.
    file_extentsion (str): The file extension of the file.

    Returns:
    str: The key of the file if found, otherwise None.
    """
    list_files = []
    for page in paginator:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if tile_name is None:
                    if VI is not None and year is not None:
                        if VI in key and key.endswith(file_extension):
                            if year in key:
                                list_files.append(key)
                elif tile_name and year and VI:
                    if VI in key and tile_name in key and key.endswith(file_extension):
                        # ensure that the file is from the correct year
                        if year in key:
                            list_files.append(key)
                elif tile_name and year and VI is None:
                    if tile_name in key and key.endswith(file_extension):
                        # ensure that the file is from the correct year
                        if year in key:
                            list_files.append(key)
                elif tile_name and VI is None and year is None:
                    if tile_name in key and key.endswith(file_extension):
                        list_files.append(key)
    return list_files


def get_timesat_input_files(
    s3_credentials, args_processing, VI, Q_FLAG, yrstart, yrend, LC_file=""
):
    """
    Retrieve and match VI and Q_FLAG files from S3
    for the specified years and tile.
    Returns:
        flist: list of VI files
        qlist: list of Q_FLAG files
    """
    lst_files_VI = []
    if Q_FLAG is not None:
        lst_files_QFLAG = []
    for year_check in range(yrstart, yrend + 1):
        bucket_files = list_files_S3_bucket(s3_credentials, str(year_check))
        files_tile_year = find_file_in_S3_bucket(
            bucket_files, args_processing.tile, str(year_check)
        )
        files_VI = [item for item in files_tile_year if VI in item]
        if Q_FLAG is not None:
            if Q_FLAG == "QFLAG2":
                files_QFLAG = [item for item in files_tile_year if Q_FLAG in item]
            elif Q_FLAG == "LOS":
                s3_credentials_LOS = s3_credentials.copy()
                s3_credentials_LOS.update(
                    {
                        "prefix": (
                            f"LOS/v100/tiles_utm/"
                            f"{args_processing.tile[0:2]}/"
                            f"{args_processing.tile[2:3]}/"
                            f"{args_processing.tile[3:5]}/"
                        )
                    }
                )
                bucket_files_los = list_files_S3_bucket(
                    s3_credentials_LOS, str(year_check)
                )
                files_QFLAG = find_file_in_S3_bucket(
                    bucket_files_los, args_processing.tile, str(year_check)
                )
                files_QFLAG = [
                    item for item in files_QFLAG if "MASK" in os.path.basename(item)
                ]
            else:
                raise ValueError(f"Unknown Q_FLAG option: {Q_FLAG}")
            if len(files_VI) != len(files_QFLAG):
                # print which files are missing
                vi_dates = sorted(
                    [os.path.basename(f).split("_")[1][0:8] for f in files_VI]
                )
                # remove duplicated dates
                vi_dates = list(set(vi_dates))
                if Q_FLAG == "QFLAG2":
                    pos_date = 1
                elif Q_FLAG == "LOS":
                    pos_date = 3
                qflag_dates = sorted(
                    [os.path.basename(f).split("_")[pos_date][0:8] for f in files_QFLAG]
                )
                qflag_dates = list(set(qflag_dates))
                missing_in_qflag = set(vi_dates) - set(qflag_dates)
                missing_in_vi = set(qflag_dates) - set(vi_dates)
                # if sum of missing is below 5% of total, we can ignore them
                total_dates = max(len(vi_dates), len(qflag_dates))
                
                #TODO remove this check on 2025, but now TS LOS and VI not available same period
                if (len(missing_in_qflag) + len(missing_in_vi)) / total_dates < 0.05 or year_check == 2025:
                    # remove files with missing dates
                    if missing_in_qflag:
                        files_VI = [
                            f
                            for f in files_VI
                            if os.path.basename(f).split("_")[1][0:8]
                            not in missing_in_qflag
                        ]
                    if missing_in_vi:
                        files_QFLAG = [
                            f
                            for f in files_QFLAG
                            if os.path.basename(f).split("_")[pos_date][0:8]
                            not in missing_in_vi
                        ]
                    pass  # proceed to remove missing files
                else:
                    raise ValueError("Number of VI and Q_FLAG files do not match!")

        files_VI.sort()
        lst_files_VI.extend(files_VI)
        if Q_FLAG is not None:
            files_QFLAG.sort()
            lst_files_QFLAG.extend(files_QFLAG)
        else:
            lst_files_QFLAG = []
    # now retrieve the list of dates from the filenames
    timevector = np.ndarray(len(lst_files_VI), order="F", dtype="uint32")
    for i, f in enumerate(lst_files_VI):
        vaitem = os.path.basename(f).split("_")[1][0:8]
        dt_item = datetime.datetime.strptime(vaitem, "%Y%m%d")
        timevector[i] = int(f"{dt_item.year}{dt_item.timetuple().tm_yday:03d}")

    # land cover part if LC_file is provided
    if LC_file != "":
        s3_credentials_LC = s3_credentials.copy()
        s3_credentials_LC.update({"prefix": ""})
        # check if LC_file exists in S3
        bucket_files_lc = list_files_S3_bucket(s3_credentials_LC, LC_file)
        files_tile_lc = find_file_in_S3_bucket(bucket_files_lc,
                                               args_processing.tile)
        if not files_tile_lc:
            raise ValueError(f"Land cover file {LC_file} not found in S3 bucket!")
        elif len(files_tile_lc) > 1:
            raise ValueError(f"Multiple land cover files found for {LC_file}!")
        else:
            LC_file = files_tile_lc[0]
    else:
        LC_file = ""
    return lst_files_VI, lst_files_QFLAG, timevector, LC_file


def select_best_pixel(qa_cube):
    """
    Reduce a QA cube [y, x, time] to [y, x] using custom priority.

    Parameters
    ----------
    qa_cube : np.ndarray of shape (H, W, T)
        QA values with classes 0,1,2,3,4,5,255

    Returns
    -------
    qa_best : np.ndarray of shape (H, W)
        Best QA class per pixel based on the priority rule
    """
    # priority order: lower = better
    priority = {
        0: 0,  # Surface
        4: 1,  # Snow
        5: 2,  # Snow-ambiguous
        2: 3,  # Cloud-ambiguous
        3: 4,  # Shadows
        1: 5,  # Clouds
        255: 6,  # No data
    }
    rank = np.vectorize(priority.get)(qa_cube)
    best_idx = np.argmin(rank, axis=2)  # pick the lowest-rank value along time
    return np.take_along_axis(qa_cube, best_idx[..., None], axis=2).squeeze(axis=2)


def open_image_data(
    x_map: int,
    y_map: int,
    x: int,
    y: int,
    yflist: List[str],
    wflist: List[str] | str,
    lcfile: str,
    data_type: str,
    p_a,
    layer: int,
    s3: dict | None = None,
):
    """Read VI, QA, and LC blocks as arrays."""
    z = len(yflist)
    dates = [os.path.basename(f).split("_")[1] for f in yflist]
    dates = sorted(dates)

    vi = np.ndarray((y, x, z), order="F", dtype=data_type)
    qa = np.ndarray((y, x, z), order="F", dtype=data_type)
    # Loop over the sorted dates to store chronologically in raster
    for i, date in enumerate(dates):
        # Find the index in yflist that matches this date
        yfiles = [f for f in yflist if os.path.basename(f).split("_")[1] == date]
        if len(yfiles) > 1 or not yfiles:
            raise ValueError(f"Multiple or no VI files found for date {date}")
        with rasterio.Env(**s3):
            with rasterio.open(yfiles[0], "r") as temp:
                vi[:, :, i] = temp.read(layer, window=Window(x_map, y_map, x, y))

    # QA stack
    if wflist == "" or wflist == []:
        qa = np.ones((y, x, z))
    else:
        if "LOS" in os.path.basename(wflist[0]):
            date_pos = 3
        else:
            date_pos = 1
        for i, date in enumerate(dates):
            # Find the index in wflist that matches this date
            wfiles = [
                f for f in wflist if os.path.basename(f).split("_")[date_pos] == date
            ]
            if not wfiles:
                raise ValueError(f"No QA file found for date {date}")
            if len(wfiles) > 1:
                nd_arr = np.zeros((y, x, len(wfiles)))
                for p, f in enumerate(wfiles):
                    with rasterio.Env(**s3):
                        with rasterio.open(f, "r") as temp2:
                            nd_arr[:, :, p] = temp2.read(
                                1, window=Window(x_map, y_map, x, y)
                            )
                qa[:, :, i] = select_best_pixel(nd_arr)
            else:
                with rasterio.Env(**s3):
                    with rasterio.open(wfiles[0], "r") as temp2:
                        qa[:, :, i] = temp2.read(1, window=Window(x_map, y_map, x, y))
        qa = assign_qa_weight(p_a, qa)

    # LC
    if lcfile == "":
        lc = np.ones((y, x))
    else:
        if s3 is not None:
            with rasterio.Env(**s3):
                with rasterio.open(lcfile, "r") as temp3:
                    lc = temp3.read(1, window=Window(x_map, y_map, x, y))
        else:
            with rasterio.open(lcfile, "r") as temp3:
                lc = temp3.read(1, window=Window(x_map, y_map, x, y))
    # set lc to uint8
    lc = lc.astype("uint8")

    return vi, qa, lc