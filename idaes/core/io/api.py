"""
Main API.

Usage:

        

    To save only selected parts,
    save_state("my-model.pbuf", my_model, select=StatePart.VALUE | StatePart.SUFFIX)

    To load/save the *entire* model, use the thin wrapper around `pickle`:
    save_model("my-model.pickle", model)
    my_model = load_model("my-model.pickle")
"""
from pyomo.environ import Block
from .common import (
    DataFile, 
    DataFormat,
    FileTools,
    ModelState,
    StateOption
)
    
# TODO: Implement
def save_state(
    output_file: DataFile,
    model: Block,
    data_format: DataFormat = None,
    options: int = StateOption.VALUE,
    gz: bool = False
) -> ModelState:
    """Save model state to a file or stream

    Args:
        fp (DataFile): _description_
        model (Block): _description_
        data_format (DataFormat, optional): _description_. Defaults to None.
        options (int, optional): _description_. Defaults to StateOption.VALUE.

    Returns:
        ModelState: _description_
    """
    fp = FileTools.open(output_file, is_output=True, gz=gz)
    # format model contents to buffer
    if data_format == DataFormat.PROTOBUF:
        t0 = time.time()
        buf = _protobuf_bytes(mstate)
        t1 = time.time()
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"protobuf dump time={(t1 - t0):.3g}s")
        mode, encoding = "wb", None
    else:
        t0 = time.time()
        buf = json.dumps(mstate.as_dict(), check_circular=False, **dump_kw)
        t1 = time.time()
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"json dump time={(t1 - t0):.3g}s")
        mode = "wt"
    # open output
    if isinstance(fp, str):
        fp = Path(fp)
    if isinstance(fp, Path):
        if gz:
            fp = _add_suffix(fp, ".gz")
            f = gzip.open(fp, mode)
        else:
            f = fp.open(mode, encoding=encoding)
    else:
        f = fp
    # write buffer to output
    f.write(buf)
    f.close()


# TODO: Implement
def load_state(fp: DataFile, data_format: DataFormat = None) -> ModelState:
    pass



# TODO: Implement
def load_model(fp: Datafile) -> Block


# TODO: Implement
def save_model(fp: DataFile, model: Block):
    pass

