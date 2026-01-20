"""
Test script for running TIMESAT at tile level.

Run a test with python run_timesat_test.py --testing --tile 31UFS
"""

import datetime
import getpass
import calendar
import importlib.metadata
import os
import sys
import shlex
import numpy as np
import rasterio

from pathlib import Path
from loguru import logger
from mepsy import SparkApp
from timesat_cli.config import load_config, build_param_array
from timesat_cli.dateutils import build_monthly_sample_indices
from timesat_cli.fsutils import create_output_folders
from timesat_cli.writers import (
    prepare_profiles,
)
from tqdm import tqdm
from hrvpp2.extractions.retrieve_s3 import (
    _get_params_S3_loading,
    get_img_profile_from_s3,
    read_s3_raster_window,
)
from hrvpp2.utils.timeseries import _file_date
from hrvpp2.timesat.timesat_args_parser import (
    parse_and_validate_args,
    _create_mep_config,
)
from hrvpp2.utils.constants import VPP_products, drive_name
from hrvpp2.utils.geom import create_windows_df
from hrvpp2.timesat.utils import (
    get_timesat_input_files,
    process_window_timesat,
    stitch_chunks_fast,
    _apply_timesat_metadata_and_cog,
    list_completed_window_ids,
    get_month_dekad_offsets,
    plot_vi_yfit,
    _build_output_filenames,
    load_version_from_pyproject,
)
USERNAME = getpass.getuser()


if __name__ == "__main__":
    timesatversion = importlib.metadata.version("timesat")
    PROJECT_VERSION = load_version_from_pyproject(Path(__file__).parent)
    # Below some suggested settings for running on cluster
    # exec memory 8
    # max executors 300
    # driver memory 20
    
    # If you want to run the script in debug mode without passing CLI args,
    # set the environment variable TIMESAT_DEBUG to a string of arguments, e.g.:
    #  TIMESAT_DEBUG="--start-year 2019 --tile 31UFS --testing"
    # or just run the script with no args (useful in an IDE).
    debug_env = os.getenv("TIMESAT_DEBUG")

    # Only override argv if *no* CLI args were given
    if len(sys.argv) == 1:
        if debug_env:
            # shlex.split handles quoted values properly
            debug_args = shlex.split(debug_env.strip())
        else:
            # Default debug args when none provided on the command line
            debug_args = [
                "--start-year", "2022",
                "--end-year", "2023",
                "--tile", "31UFS",
                "--testing",
                "--qflag", "LOS",
                "--version-raster", "V100",
                "--vi", "FAPAR",
                "--settings-file", os.path.join(Path(__file__).parent,
                                                "settings_hrvpp2_FAPAR.json"),
            ]

        sys.argv = [sys.argv[0]] + debug_args
        logger.info(f"Using DEBUG CLI args: {debug_args}")
    else:
        # Normal run: use the CLI args as provided
        logger.info(f"Using CLI args from command line: {sys.argv[1:]}")

    # Open HRVPP2 S3 bucket credentials
    args_processing = parse_and_validate_args()
    # load input parameters for processing
    # need to take also one year before and after for stable output
    yrstart = int(args_processing.start_year) - 1
    yrend = int(args_processing.end_year) + 1
    yr = yrend - yrstart + 1
    Q_FLAG = args_processing.qflag
    VI = args_processing.vi
        
    s3_credentials = args_processing.s3_credentials
    if args_processing.testing:
        download_raw_VI = False
        logger.info("Running in TESTING mode with reduced data volume.")
    else:
        download_raw_VI = False
        logger.info("Running in NORMAL mode with full data volume.")

    settings_file = args_processing.settings_file

    dict_band_settings = {
        "ST_FAPAR": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "p_ylu": [0, 10000],
            "phy_range": [0, 1],
            "dtype": "int16",
            "unit": "-",
        },
        "ST_PPI": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "p_ylu": [0, 10000],
            "phy_range": [0, 3],
            "unit": "-",
            "dtype": "int16",
        },
        "ST_LAI": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "p_ylu": [0, 10000],
            "phy_range": [0, 8],
            "unit": "-",
            "dtype": "int16",
        },
        "SPROD": {
            "scale": 0.1,
            "offset": 0,
            "p_nodata": 65535,
            "phy_range": [0, 1095],
            "unit": "-",
            "dtype": "uint16",
        },
        "TPROD": {
            "scale": 0.1,
            "offset": 0,
            "p_nodata": 65535,
            "phy_range": [0, 1095],
            "unit": "-",
            "dtype": "uint16",
        },
        "SOSD": {
            "scale": 1,
            "offset": 0,
            "p_nodata": 0,
            "phy_range": "-",
            "unit": "YYDOY",
            "dtype": "uint16",
        },
        "EOSD": {
            "scale": 1,
            "offset": 0,
            "p_nodata": 0,
            "phy_range": "-",
            "unit": "YYDOY",
            "dtype": "uint16",
        },
        "LENGTH": {
            "scale": 1,
            "offset": 0,
            "p_nodata": 0,
            "phy_range": "0 to 1096",
            "unit": "days",
            "dtype": "uint16",
        },
        "MAXD": {
            "scale": 1,
            "offset": 0,
            "p_nodata": 0,
            "phy_range": "0 to 1096",
            "unit": "YYDOY",
            "dtype": "uint16",
        },
        "SOSV": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "phy_range": "0 to 1",
            "unit": "days",
            "dtype": "int16",
        },
        "EOSV": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "phy_range": "0 to 1",
            "unit": "days",
            "dtype": "int16",
        },
        "LSLOPE": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "phy_range": "0.01 to 0.5",
            "unit": "-",
            "dtype": "int16",
        },
        "RSLOPE": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "phy_range": "0.01 to 0.5",
            "unit": "-",
            "dtype": "int16",
        },
        "MAXV": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "phy_range": "0 to 1",
            "unit": "-",
            "dtype": "int16",
        },
        "MINV": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "phy_range": "0 to 1",
            "unit": "-",
            "dtype": "int16",
        },
        "AMPL": {
            "scale": 0.0001,
            "offset": 0,
            "p_nodata": -32768,
            "phy_range": "0 to 1",
            "unit": "-",
            "dtype": "int16",
        },
    }

    dict_filename_settings = {
        "ST_FAPAR": "ST_{date}_S2_T{tile}-010m_{version}_FAPAR.tif",
        "ST_FAPAR_QFLAG": "ST_{date}_S2_T{tile}-010m_{version}_QFLAG.tif",
        "VPP": "VPP_{year}_S2_T{tile}-010m_{version}_s{season}_{product}.tif",
        "VPP_QFLAG": "VPP_{year}_S2_T{tile}-010m_{version}_s{season}_QFLAG.tif",
    }

    s_vi = dict_band_settings[f"ST_{VI}"]
    cfg = load_config(settings_file)
    s = cfg.settings
    s.p_nodata = s_vi["p_nodata"]
    s.p_ylu = s_vi["p_ylu"]
    s.scale = s_vi["scale"]
    s.offset = s_vi["offset"]
    if "FAPAR" in os.path.basename(settings_file) or "LAI" in os.path.basename(settings_file):
        # fix output options for FAPAR/LAI runs
        ST_only = True
        LC_specific = False
    else:
        ST_only = False
        LC_specific = True
    
    s.lc_file = "CLMS/Pan-European/Ancillary/Landcover/v03/2021"

    # only use one CPU core
    s.para_check = 1

    # Output base directories
    if args_processing.testing:
        s.outputfolder = os.path.join(
            drive_name, "user", USERNAME, "timesat_test", args_processing.tile
        )
        os.makedirs(s.outputfolder, exist_ok=True)
        plotting = True
    else:
        s.outputfolder = os.path.join(
            drive_name, "user", USERNAME, "timesat", args_processing.tile
        )
        plotting = False
    
    # TIMESAT input file lists + time indexing
    flist, qlist, timevector, LC_file = get_timesat_input_files(
        s3_credentials, args_processing, VI, Q_FLAG, yrstart, yrend,
        LC_file=s.lc_file
    )
   
    if ST_only:
        # only keep the files from the last 2 months of buffer
        # year before and first two months of buffer year after
        # Keep all files from yrstart..yrend plus buffer months
        # (Nov/Dec of the year before and Jan/Feb of the year after)
        keep_year_months = set()
        for y in range(yrstart+1, yrend):
            for m in range(1, 13):
                keep_year_months.add((y, m))
        keep_year_months.update(
            {(yrstart, 11), (yrstart, 12), (yrend, 1),
             (yrend, 2)}
        )
    else:
        # only keep the files from 12 months of buffer
        # year before and first two months of buffer year after
        # Keep all files from yrstart..yrend plus buffer months
        keep_year_months = set()
        for y in range(yrstart, yrend):
            for m in range(1, 13):
                keep_year_months.add((y, m))
        keep_year_months.update(
            {(yrend, 1),
             (yrend, 2)}
        )

    filtered_flist = []
    filtered_dates = set()
    for f in flist:
        dt = _file_date(f, pos=1)
        if not dt:
            continue
        if (dt.year, dt.month) in keep_year_months:
            filtered_flist.append(f)
            filtered_dates.add(dt.strftime("%Y%m%d"))

    if not filtered_flist:
        raise RuntimeError(
            "No VI files found after applying ST-only temporal filter. "
            "Check input files / years."
        )

    # if qlist present, keep only those that match the filtered VI dates
    if qlist:
        qa_date_pos = 3 if Q_FLAG == "LOS" else 1
        filtered_qlist = []
        for q in qlist:
            dtq = _file_date(q, pos=qa_date_pos)
            if (dtq.year, dtq.month) in keep_year_months:
                filtered_qlist.append(q)
        qlist = filtered_qlist
        # # find dates with multiple files in qlist
        # # if so only retain first file
        # # sort qlist chronologically
        # qa_date_pos = 3 if Q_FLAG == "LOS" else 1
        # seen_dates = set()
        # filtered_qlist = []
        # dup_count = 0
        # for q in qlist:
        #     dt = _file_date(q, pos=qa_date_pos)
        #     if not dt:
        #         # skip files we cannot parse
        #         continue
        #     date_str = dt.strftime("%Y%m%d")
        #     if date_str in seen_dates:
        #         dup_count += 1
        #         continue
        #     seen_dates.add(date_str)
        #     filtered_qlist.append(q)
        # if dup_count > 0:
        #     logger.warning(f"Found {dup_count} QA files with duplicate dates; keeping first occurrence for each date.")
        # qlist = filtered_qlist

    # replace flist with filtered list and rebuild timevector accordingly
    flist = filtered_flist
    timevector = np.ndarray(len(flist), order="F", dtype="uint32")
    for i, f in enumerate(flist):
        vaitem = os.path.basename(f).split("_")[1][0:8]
        dt_item = datetime.datetime.strptime(vaitem, "%Y%m%d")
        timevector[i] = int(f"{dt_item.year}{dt_item.timetuple().tm_yday:03d}")

    s.lc_file = os.path.join("s3://", s3_credentials["bucket_name"], LC_file)

    # === DEBUG: optional write of raw VI windows to disk
    if args_processing.testing and download_raw_VI:
        for f in tqdm(flist, desc="Reading S3 rasters"):
            data = read_s3_raster_window(
                f,
                s3_credentials,
                row_off=s.imwindow[1],
                col_off=s.imwindow[0],
                window_size_x=s.imwindow[2],
                window_size_y=s.imwindow[3],
            )
            if s.scale != 0 or s.offset != 0:
                data = data.astype(np.float32)
                data[data != s.p_nodata] = data[data != s.p_nodata] * s.scale + s.offset
                data[data == s.p_nodata] = np.nan
            profile = get_img_profile_from_s3(s3_credentials, [f])
            debug_file = os.path.join(s.outputfolder, "raw", os.path.basename(f))
            if os.path.exists(debug_file):
                continue
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            profile.update(
                {
                    "height": s.imwindow[3],
                    "width": s.imwindow[2],
                    "transform": rasterio.windows.transform(
                        rasterio.windows.Window(*s.imwindow), profile["transform"]
                    ),
                    "dtype": "float32"
                    if (s.scale != 0 or s.offset != 0)
                    else profile["dtype"],
                    "nodata": np.nan
                    if (s.scale != 0 or s.offset != 0)
                    else profile["nodata"],
                }
            )
            with rasterio.open(debug_file, "w", **profile) as dst:
                dst.write(data, 1)
        
    # Period and output indices
    start_date = datetime.datetime(yrstart, 1, 1)
    end_date = datetime.datetime(yrstart + yr - 1, 12, 31)

    # p_outindex = np.arange(1, (yr*365)+1)
    # now get the index position for every dekad in p_outindex
    # for every month the index of day 1, 11, 21 should be included
    # month_dekad_offsets = get_month_dekad_offsets(yrstart, yrend, start_date)
    month_dekad_offsets = build_monthly_sample_indices(
        yrstart, yr
    )
    # TODO: add fix to get only indices of years excluding years in buffer monhts
    # Condition 1 should start at index position 36
    # Condition 2 should be exclude last 36 indices of month_dekad_offsets
    p_outindex = [month_dekad_offsets[i] for i in range(len(month_dekad_offsets))
                  if (i >= 36 and
                      i < (len(month_dekad_offsets) - 36))]
    p_outindex_num = len(p_outindex)

    # Get a representative profile for output files
    img_profile = get_img_profile_from_s3(s3_credentials, flist)
    st_folder, vpp_folder = create_output_folders(s.outputfolder)
    outyfitfn, outvppfn, outnsfn = _build_output_filenames(
        start_date, st_folder, vpp_folder, p_outindex, yrstart, yrend
    )

    # add _tmp to filenames
    outyfitfn = [f.replace(".tif", "_tmp.tif") for f in outyfitfn]
    # start os.path.basename with ST_
    outyfitfn = [os.path.join(st_folder, f"ST_{Path(f).name}") for f in outyfitfn]
    # create also QFLAG names
    outyfitfn_qflag = [
        os.path.join(st_folder, f"{Path(f).stem}_QFLAG.tif") for f in outyfitfn
    ]
    outvppfn = [f.replace(".tif", "_tmp.tif") for f in outvppfn]
    outnsfn = [f.replace(".tif", "_tmp.tif") for f in outnsfn]
    # set scale and offset to zero
    # otherwise st wil be a float output
    img_profile_st, img_profile_vpp, _, img_profile_ns = prepare_profiles(
        img_profile, s.p_nodata, 0, 0
    )
    # set image profiles VPP to int32
    img_profile_vpp.update({"dtype": "int32"})

    # S3 params + absolute S3 URLs for open_image_data
    s3_params, bucket = _get_params_S3_loading(s3_credentials)
    flist = [os.path.join("s3://", bucket, f) for f in flist]
    if qlist:
        qlist = [os.path.join("s3://", bucket, f) for f in qlist]

    # Window size for tiling (tune if needed)
    WINDOW_SIZE = args_processing.window_size
    df_window = create_windows_df(args_processing.tile,
                                  window_size=WINDOW_SIZE)
    nr_windows_all = df_window.shape[0]
    # remove windows for which output is already created
    if ST_only:
        ids_done = list_completed_window_ids(
            df_window,
            st_folder,
            args_processing.tile,
            product_CAT="ST",
            suffix="",
        )
    else:
        ids_done_ST = list_completed_window_ids(
            df_window,
            st_folder,
            args_processing.tile,
            product_CAT="ST",
            suffix="",
        )
        ids_done_VPP = list_completed_window_ids(
            df_window,
            vpp_folder,
            args_processing.tile,
            product_CAT="VPP",
            suffix="",
        )
        ids_done = ids_done_ST.intersection(ids_done_VPP)
    # remove completed windows from df_window
    df_window = df_window[
        ~df_window.apply(
            lambda row: f"{int(row['x_end'])}_{int(row['y_end'])}" in ids_done,
            axis=1,
        )
    ].reset_index(drop=True)
    # Prepare broadcast context for workers
    # Add all items from s to s_dict
    s_dict = {k: v for k, v in s.__dict__.items()}
    st_chunk_dir = os.path.join(s.outputfolder, "st_chunks")
    os.makedirs(st_chunk_dir, exist_ok=True)
    if not ST_only:
        vpp_chunk_dir = os.path.join(s.outputfolder, "vpp_chunks")
        os.makedirs(vpp_chunk_dir, exist_ok=True)
    else:
        vpp_chunk_dir = None

    ctx = {
        "s_dict": s_dict,
        "flist": flist,
        "qlist": qlist,
        "lc_file": s.lc_file,
        "img_dtype": img_profile["dtype"],
        "p_a": s.p_a if Q_FLAG is not None else [],
        "para_check": s.para_check,
        "p_band_id": s.p_band_id,
        "s3_params": s3_params,
        "yr": yr,
        "timevector": timevector.tolist(),
        "landuse_arr": build_param_array(s, "landuse", "uint8"),
        "p_outindex": p_outindex,
        "p_outindex_num": int(p_outindex_num),
        "scale": s.scale,
        "offset": s.offset,
        "p_nodata": s.p_nodata,
        "st_chunk_dir": st_chunk_dir,
        "vpp_chunk_dir": vpp_chunk_dir,
        "tile": args_processing.tile,
        "img_profile_st": img_profile_st,
        "img_profile_vpp": img_profile_vpp,
        "ST_only": ST_only,
        "VPP_products": VPP_products,
        "LC_specific": LC_specific,
    }
    # ============== DEBUG MODE: single-window, plotting, direct write ==============
    if args_processing.testing:
        # process two windows for testing
        df_window = df_window.head(2)
        nr_windows_all = df_window.shape[0]
        for _, window_info in df_window.iterrows():
            if len(ids_done) == 2:
                break
            if plotting:
                vi_w, yfit_w, vpp_w, qa_w, yfitqa_w, vppqa_w = process_window_timesat(
                    window_info, ctx, return_vars=plotting
                )
                x_map = window_info["x_start"]
                y_map = window_info["y_start"]
                x = WINDOW_SIZE
                y = WINDOW_SIZE

                np.random.seed(42)
                y_indices = np.random.randint(0, vi_w.shape[0], size=5)
                x_indices = np.random.randint(0, vi_w.shape[1], size=5)
                # define the timevector for yfit based on month dekads
                # do not account for leap days between start and end date
                total_days = (yrend - yrstart + 1) * 365 
                p_outindex_daily = np.arange(0, total_days)
                # now get the index position for every dekad in p_outindex
                # for every month the index of day 1, 11, 21 should be included
                month_dekad_offsets = get_month_dekad_offsets(
                    yrstart, yrend, start_date
                )
                # get the actual days
                timevector_yfit = np.array(p_outindex_daily)[month_dekad_offsets]
                # convert to YYYYDDD format
                timevector_yfit_yyyydoy = []
                for day in timevector_yfit:
                    # compute tentative date then subtract
                    # dec 31 that lie between
                    # start_date (exclusive) and tentative (inclusive)
                    leap_days = 0
                    tentative = start_date + datetime.timedelta(days=int(day))
                    start_d = start_date.date()
                    end_d = tentative.date()
                    for y in range(start_d.year, end_d.year + 1):
                        if calendar.isleap(y):
                            dec31 = datetime.date(y, 12, 31)
                            if start_d < dec31 <= end_d:
                                leap_days += 1
                    date = tentative + datetime.timedelta(days=leap_days)
                    timevector_yfit_yyyydoy.append(
                        int(f"{date.year}{date.timetuple().tm_yday:03d}")
                    )
                timevector_yfit_yyyydoy = np.array(timevector_yfit_yyyydoy)
                # plot for random pixels
                for y_idx, x_idx in zip(y_indices, x_indices):
                    plot_vi_yfit(
                        vi_w,
                        yfit_w,
                        timevector,
                        timevector_yfit_yyyydoy,
                        nodata_value=-32768,
                        pixel_coords=(y_idx, x_idx),
                        outputfolder=s.outputfolder,
                    )
            else:
                process_window_timesat(window_info, ctx)

    else:
        # ================= NORMAL MODE: parallel per-window with mep.foreach =================
        # change app name to TIMESAT instead of GPP
        app_name = f"TIMESAT_processing_{args_processing.tile}_{VI}_{yrstart}_{yrend}"
        # remove docker image entry if present

        aws_env = f"/data/users/Private/{USERNAME}/configs/aws.env"
        kinit_env = f"/data/users/Private/{USERNAME}/configs/kinit.env"
        # set local_spark to False to use cluster spark
        args_processing.local_spark = False
        # Create MEP configuration for SparkApp
        mep_config = _create_mep_config(
            args_processing,
            app_name,
            settings_file,
            PROJECT_VERSION,
        )

        mep = SparkApp(**mep_config)
       
        def _wrapper(win_info):
            return process_window_timesat(win_info, ctx)

        window_infos = df_window.to_dict(orient="records")
        mep.foreach(_wrapper, window_infos)

    # st stitching
    # add predictor and zlevel to img_profile
    img_profile_st.update(
        compress="LZW",
        predictor=2,
        zlevel=6,
    )
    stitched_st_tmp = stitch_chunks_fast(
        st_chunk_dir,
        outyfitfn,
        p_outindex_num,
        "ST",
        VI,
        img_profile=img_profile_st,
        qa=False,
        band_settings_map=dict_band_settings,
        nr_windows=nr_windows_all,
    )

    # stqa stitching
    stitched_stqa_tmp = stitch_chunks_fast(
        st_chunk_dir,
        outyfitfn_qflag,
        len(outyfitfn_qflag),
        "ST",
        VI,
        img_profile=img_profile_st,
        qa=True,
        band_settings_map=dict_band_settings,
        nr_windows=nr_windows_all,
    )
    # cleanup chunk dir now
    if os.path.exists(st_chunk_dir):
        # first empty the dir
        # Remove all files in st_chunk_dir and its subfolders
        for root, dirs, files in os.walk(st_chunk_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(st_chunk_dir)
    # vpp stitching
    if not ST_only:
        # update img_profile_vpp
        img_profile_vpp.update(
            compress="deflate",
            predictor=2,
            zlevel=6,
        )
        stitched_vpp_tmp = stitch_chunks_fast(
            vpp_chunk_dir,
            outvppfn,
            len(outvppfn),
            "VPP",
            VI,
            img_profile=img_profile_vpp,
            qa=False,
            band_settings_map=dict_band_settings,
            nr_windows=df_window.shape[0],
        )
        # VPP QA stitching
        n_seasons = 2
        outvppfn_qa = []
        for yr_ in range(yrstart+1, yrend):
            for season in range(1, n_seasons + 1):
                outvppfn_qa.append(
                    os.path.join(
                        vpp_folder,
                        (
                            f"VPP_{yr_}_S2_T{args_processing.tile}-010m_"
                            f"{args_processing.version_raster}_s{season}_QFLAG_tmp.tif"
                        ),
                    )
                )
        total_vpp_qa_bands = len(outvppfn_qa)
        stitched_vppqa_tmp = stitch_chunks_fast(
            vpp_chunk_dir,
            outvppfn_qa,
            total_vpp_qa_bands,
            "VPP",
            VI,
            img_profile=img_profile_vpp,
            qa=True,
            band_settings_map=dict_band_settings,
            nr_windows=df_window.shape[0],
        )
        # cleanup vpp chunk dir now
        for root, dirs, files in os.walk(vpp_chunk_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(vpp_chunk_dir)

    # daterange (used in tags)
    start_date = f"{yrstart+1}-01-01"
    end_date = f"{yrend-1}-12-31"
    daterange = f"{start_date} to {end_date}"

    # APPLY META + COG (same function as window writer)
    _apply_timesat_metadata_and_cog(
        stitched_st_tmp,
        band_desc_key=VI,
        format_key=f"ST_{VI}",
        date_idx=-1,
        extra_tags=None,
        band_settings=dict_band_settings[f"ST_{VI}"],
        daterange=daterange,
        tile=args_processing.tile,
        version=args_processing.version_raster,
        dict_formats=dict_filename_settings,
        TimesatVersion=timesatversion,
        product_CAT="ST",
        s3_credentials=s3_credentials,
    )

    _apply_timesat_metadata_and_cog(
        stitched_stqa_tmp,
        band_desc_key=f"{VI}_QFLAG",
        format_key=f"ST_{VI}_QFLAG",
        date_idx=-3,
        extra_tags={
            "Flag_meaning": (
                "no data (time series was not processed)",
                "Filled (extrapolation), Filled (intrapolation), "
                "Low, Medium, High"
            ),
            "Flag_value": "(0, 1, 2, 3, 4, 5)",
        },
        band_settings={},  # QA: no scale/offset/PhysRange
        daterange=daterange,
        tile=args_processing.tile,
        version=args_processing.version_raster,
        dict_formats=dict_filename_settings,
        TimesatVersion=timesatversion,
        product_CAT="ST",
        s3_credentials=s3_credentials,
    )

    if not ST_only:
        # Loop over VPP product types to apply metadata and COG for each
        for product in VPP_products:
            # Find stitched VPP files for this product
            stitched_vpp_tmp_product = [
                f for f in stitched_vpp_tmp if f"_{product}_" in f
            ]
            _apply_timesat_metadata_and_cog(
                stitched_vpp_tmp_product,
                band_desc_key=product,
                format_key="VPP",
                date_idx=-3,
                extra_tags=None,
                band_settings=dict_band_settings.get(product, {}),
                daterange=daterange,
                tile=args_processing.tile,
                version=args_processing.version_raster,
                dict_formats=dict_filename_settings,
                TimesatVersion=timesatversion,
                product_CAT="VPP",
                s3_credentials=s3_credentials,
            )

        _apply_timesat_metadata_and_cog(
            stitched_vppqa_tmp,
            band_desc_key="VPP_QFLAG",
            format_key="VPP_QFLAG",
            date_idx=-6,
            extra_tags={
                "Flag_meaning": (
                    "(no data (time series was not processed), No season found,"
                    "Filled, -, "
                    "Low green down, Low green peak,"
                    "Low green up, Medium green up, "
                    "Medium green peak, Medium green down, High)",
                ),
                "Flag_value": "(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)",
            },
            band_settings={},  # QA: no scale/offset/PhysRange
            daterange=daterange,
            tile=args_processing.tile,
            version=args_processing.version_raster,
            dict_formats=dict_filename_settings,
            TimesatVersion=timesatversion,
            product_CAT="VPP",
            s3_credentials=s3_credentials,
        )