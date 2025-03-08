# """
# Performance tests.
# Also runnable from command-line.
# """

# # stdlib
# import argparse
# import sys
# import time

# # third-party
# import pytest

# # pkg
# from idaes.core.io.api import StateFile
# from idaes.core.io.common import DataFormat


# def create_model(days):
#     from idaes_models.models.stepload_case.flowsheets.heating_flowsheet_stepload_additional_constraints import (
#         build_model,
#         setup_model,
#     )

#     ts = 24 * days  # hours * days
#     print(f"create model with {ts} timesteps")
#     pbd = build_model(timesteps=ts, load_kw=1000)
#     setup_model(pbd, timesteps=ts, load_kw=1000)
#     return pbd


# def test_json_write(pbd, fname, gz=False, **kwargs):
#     StateFile(fname, fmt=DataFormat.JSON, gz=gz).save(pbd)


# def test_json_read(fname, gz=False, encoding="utf-8"):
#     filename = fname + ".json.gz" if gz else fname + ".json"
#     return load(None, filename, gz=gz, data_format=DataFormat.JSON, set_values=False)


# def test_protobuf_write(pbd, fname, gz=False, **kwargs):
#     model = ModelState()
#     model.core.build(pbd, **kwargs)
#     save(model, fname + ".pbuf", data_format=DataFormat.PROTOBUF, gz=gz)


# def test_protobuf_read(fname, gz=False):
#     filename = fname + ".pbuf.gz" if gz else fname + ".pbuf"
#     return load(
#         None, filename, data_format=DataFormat.PROTOBUF, gz=gz, set_values=False
#     )


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--days", type=int, default=2)
#     p.add_argument("--verbose", "-v", action="count")
#     args = p.parse_args()

#     pbd = create_model(args.days)

#     for kwargs in (
#         {},
#         {"include_suffixes": True, "include_configs": True, "include_conn": True},
#     ):
#         print("================================")
#         if not kwargs:
#             print("flags = default")
#         else:
#             flags_str = ", ".join(kwargs.keys())
#             print(f"flags = {flags_str}")
#         print("--------------------------------")
#         times = {
#             "protobuf": {"read": {"gz": [], "raw": []}, "write": {"gz": [], "raw": []}},
#             "json": {"read": {"gz": [], "raw": []}, "write": {"gz": [], "raw": []}},
#         }
#         for i in range(3):
#             for gz in (True, False):
#                 gz_key = "gz" if gz else "raw"
#                 print(f"run {i + 1} gzip={gz}")
#                 flags = "_".join((k[8:] for k in kwargs.keys()))
#                 if not flags:
#                     filename = f"test-base-{i}"
#                 else:
#                     filename = f"test-{flags}-{i}"
#                 print("  protobuf")
#                 t0 = time.time()
#                 test_protobuf_write(pbd, filename, gz=gz, **kwargs)
#                 times["protobuf"]["write"][gz_key].append(time.time() - t0)
#                 t0 = time.time()
#                 test_protobuf_read(filename, gz=gz)
#                 times["protobuf"]["read"][gz_key].append(time.time() - t0)
#                 print("  json")
#                 t0 = time.time()
#                 test_json_write(pbd, filename, gz=gz, **kwargs)
#                 times["json"]["write"][gz_key].append(time.time() - t0)
#                 t0 = time.time()
#                 test_json_read(filename, gz=gz)
#                 times["json"]["read"][gz_key].append(time.time() - t0)
#         for fmt in times:
#             for direction in ("read", "write"):
#                 for gz_key in ("raw", "gz"):
#                     t = times[fmt][direction][gz_key]
#                     median_time = median(t)
#                     time_info = " ".join((f"{v:.3g}" for v in t))
#                     print(
#                         f"{fmt} + {direction} + {gz_key}: {median_time:.3g} [ {time_info} ]"
#                     )
