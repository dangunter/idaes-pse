#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
List of JSON documents, with accompanying index
"""

# stdlib
import csv
import json
import logging
from pathlib import Path
import re
from typing import Optional, Iterable

_log = logging.getLogger(__name__)

__author__ = "Dan Gunter (LBNL)"


class BadIndexFile(Exception):
    """Raised if index file is not expected format, or is not accessible."""


class JsonList:
    """Files to save/load an append-only list of JSON documents."""

    #: Suffix added to the stem of the main data filename to
    #: create an accompanying index file
    INDEX_FILE_SUFFIX = "-index.csv"

    _INDEX_FILE_HEADER = ("offset", "hash", "desc", "tags")

    def __init__(self, data_file: Path | str, index_file: Path | str | None = None):
        """Constructor

        Raises:
            BadIndexFile: If index file cannot be accessed or has a bad format
        """
        # set file paths
        self.data_file = Path(data_file)
        if index_file is None:
            self.index_file = self.data_file.parent / (
                self.data_file.stem + self.INDEX_FILE_SUFFIX
            )
        else:
            self.index_file = Path(index_file)
        # read index file (create if necessary)
        self._load_index()

    def __len__(self) -> int:
        """Return number of entries."""
        return self._index_len

    def __getitem__(self, index: int) -> dict:
        """Get document at the given index.

        Args:
            index: 0-based index to get document.
                   Special value -1 gets last document.
        Returns:
            Document, parsed into a dictionary

        Raises:
            IndexError: if index is not valid
        """
        if index == -1:
            index = self._index_len - 1
            is_last = True
        elif index < 0 or index >= self._index_len:
            raise IndexError(
                f"document index {index} out of range 0..{self._index_len - 1}"
            )
        else:
            is_last = index == self._index_len - 1
        offs = self._index["offset"][index]
        with self.data_file.open("r") as f:
            f.seek(offs)
            buf = f.read() if is_last else f.readline()
        return json.loads(buf)

    @property
    def metadata(self) -> Iterable[dict[str, int | str | list[str]]]:
        """Iterable over the document metadata.

        Each item is a dict with keys: "hash", "desc", and "tags"

        Returns:
           iterable of dicts
        """
        return (
            {
                col: self._index[col][i]
                for col in self._INDEX_FILE_HEADER
                if col != "offset"
            }
            for i in range(self._index_len)
        )

    def append(
        self,
        doc: dict | str,
        hash: str = "",
        desc: str = "",
        tags: Optional[list[str]] = None,
    ):
        """Append one JSON document to the list.

        Args:
            doc: JSON document to append
            hash: File hash
            desc: Description (newlines will be converted to \n)
            tags: List of string tags (any non-alphanumeric chars
                  will be converted to underscores)
        """
        if isinstance(doc, dict):
            clean_doc = json.dumps(doc)
        else:
            # remove any embedded newlines
            clean_doc = doc.replace("\n", "\\n")
        if tags is None:
            tags = []
        # add document to data file
        with self.data_file.open("a") as f:
            offs = f.tell()
            f.write(clean_doc)
            f.write("\n")
        # add to index
        self._index["offset"].append(offs)
        self._index["hash"].append(hash)
        self._index["desc"].append(desc)
        self._index["tags"].append(tags)
        self._index_len += 1
        # write out new index file
        with open(self.index_file, "w") as f:
            wrt = csv.writer(f)
            wrt.writerow(self._INDEX_FILE_HEADER)
            for i in range(self._index_len):
                row = []
                for colname in self._INDEX_FILE_HEADER:
                    value = self._index[colname][i]
                    # format certain columns as strings
                    if colname == "offset":
                        value = str(value)
                    elif colname == "tags":
                        value = ";".join(value)
                    row.append(value)
                # write out the index row
                wrt.writerow(row)

    @staticmethod
    def _clean_tag(cls, tag):
        return re.sub(r"[^0-9a-zA-Z_]", "_", tag)

    def _load_index(self):
        if not self.index_file.exists():
            _log.debug(
                f"new file '{self.index_file}': create index file and add header"
            )
            try:
                with self.index_file.open("w") as f:
                    wrt = csv.writer(f)
                    wrt.writerow(self._INDEX_FILE_HEADER)
            except IOError as err:
                raise BadIndexFile(f"{self.index_file} cannot be created: {err}")

        _log.debug(f"load index file '{self.index_file}'")
        with self.index_file.open("r") as f:
            rdr = csv.reader(f)
            try:
                header = next(rdr)
            except StopIteration as err:
                raise BadIndexFile(f"{self.index_file}: {err}")
            _log.debug(f"index file header={header}")
            self._index = {k: [] for k in header}
            self._index_len = 0
            for row in rdr:
                self._index_len += 1
                for j, value in enumerate(row):
                    colname = header[j]
                    # parse certain column string values
                    if colname == "offset":
                        value = int(value)
                    elif colname == "tags":
                        value = value.split(";")
                    # add row values to index
                    self._index[header[j]].append(value)
            _log.debug(f"read {self._index_len} index file rows")
