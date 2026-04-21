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
Simple database to contain reports for any number of flowsheets.
"""

import json
import logging
import sqlite3
import time

__author__ = "Dan Gunter (LBNL)"

_log = logging.Logger(__name__)


class ReportDB:
    """Store and retrieve flowsheet reports in a SQLite database.

    Each row stores report content together with target-identifying metadata
    such as flowsheet name, module, source filename, and user-defined tags.

    Args:
        filename: Path to the SQLite database file.
        **target_kw: Initial target metadata values keyed by names in
            :attr:`TARGET_COLUMNS`.

    Raises:
        KeyError: If any target keyword does not match a supported target
            column.
    """

    TABLE = "reports"
    TARGET_COLUMNS = (
        ("name", "TEXT"),
        ("module", "TEXT"),
        ("filename", "TEXT"),
        ("main_func", "TEXT"),
        ("fs_object", "TEXT"),
    )
    COLUMNS = tuple(
        [
            ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
            ("created", "REAL"),
            ("target_hash", "TEXT"),
            ("tags", "TEXT"),
            ("report", "BLOB"),
        ]
        + list(TARGET_COLUMNS)
    )

    def __init__(self, filename, **target_kw):
        self._filename = filename
        self._tgtcol = [name for name, type_ in self.TARGET_COLUMNS]
        self._tgtval = {k: "" for k in self._tgtcol}
        if target_kw:
            self.set_target(**target_kw)

    def set_target(self, **kw):
        """Set default target metadata for future queries and inserts.

        Args:
            **kw: Target metadata values keyed by names in
                :attr:`TARGET_COLUMNS`.

        Raises:
            ValueError: If no keyword arguments are provided.
            KeyError: If a keyword does not match a supported target column.
        """
        if not kw:
            raise ValueError("At least one keyword argument required")
        for k, v in kw.items():
            k = k.lower()
            if k not in self._tgtcol:
                raise KeyError(f"Unknown target column '{k}'")
            self._tgtval[k] = v

    def _connect(self):
        return sqlite3.connect(self._filename)

    def create(self, drop=False):
        """Create the reports table in the database.

        Args:
            drop: If ``True``, drop the existing reports table before creating
                it again.

        Raises:
            sqlite3.Error: If SQLite cannot drop or create the table.
        """
        _log.info("Create reports table")
        with self._connect() as conn:
            if drop:
                conn.execute(f"DROP TABLE {self.TABLE};")
            create_cols = self._all_columns(typed=True)
            conn.execute(f"CREATE TABLE {self.TABLE} ( {', '.join(create_cols)} );")
            conn.commit()

    def _all_columns(self, typed=False, exclude=None):
        result = []
        for nm, ty in self.COLUMNS:
            if exclude and nm in exclude:
                continue
            result.append(f"{nm} {ty}" if typed else nm)
        return result

    def add_report(self, data: str | dict, tags: str = "", hash_=None, **target_kw):
        """Insert a report and its metadata into the database.

        Args:
            data: Report payload as a JSON string or dictionary.
            tags: Space-separated tags to store with the report. Tags are
                normalized to lowercase and sorted before storage.
            hash_: Optional hash for the report target. If omitted, an empty
                string is stored.
            **target_kw: Target metadata values for this report keyed by names
                in :attr:`TARGET_COLUMNS`. Values not provided here fall back
                to the current target set by :meth:`set_target`.

        Raises:
            KeyError: If required target metadata is missing from both
                ``target_kw`` and the current target defaults.
            TypeError: If ``data`` cannot be JSON-encoded when provided as a
                dictionary-like object.
            sqlite3.Error: If SQLite cannot insert the report.
        """
        _log.debug(f"Add a report, target={target_kw}")
        with self._connect() as conn:
            # set non-user column values
            created = time.time()
            if hash_ is None:
                hash_ = ""
            insert_cols = self._all_columns(exclude=("id",))
            # sort tags so simple LIKE search can work
            tag_items = [t.lower() for t in tags.split()]
            tag_items.sort()
            tags = " ".join(tag_items)
            # get user-defined column values from 'kw'
            tgtvalues = [
                target_kw[u] if u in target_kw else self._tgtval[u]
                for u in self._tgtcol
            ]
            # get report as bytes
            if isinstance(data, str):
                rpt_bytes = data.encode("utf-8")
            else:
                rpt_bytes = json.dumps(data).encode("utf-8")
            # construct inserted values and placeholder
            colvalues = [created, hash_, tags, rpt_bytes] + tgtvalues
            ph = ",".join("?" * len(insert_cols))
            # execute the insert
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO {self.TABLE} ({", ".join(insert_cols)}) VALUES ({ph})",
                colvalues,
            )
            # cleanup
            cur.close()
            conn.commit()

    def get_metadata(self, tags: str = "", **target_kw):
        """Yield metadata rows for reports matching the provided filters.

        Args:
            tags: Space-separated tags that must all be present in a matching
                report.
            **target_kw: Target metadata filters keyed by names in
                :attr:`TARGET_COLUMNS`. Values not provided here fall back to
                the current target defaults.

        Yields:
            tuple: Report metadata rows excluding the report payload itself.

        Raises:
            sqlite3.Error: If SQLite cannot execute the query.
        """
        columns = ", ".join(self._all_columns(exclude=("report",)))
        stmt = f"SELECT {columns} from {self.TABLE}"
        stmt += self._where(target_kw, tags=tags)
        with self._connect() as conn:
            for row in conn.execute(stmt):
                yield row

    def get_report(self, index) -> str:
        """Read a report payload by database row identifier.

        Args:
            index: SQLite row identifier for the report to read.

        Returns:
            dict | list | str | int | float | bool | None: Parsed JSON content
            stored for the report.

        Raises:
            sqlite3.Error: If SQLite cannot read the report blob.
            json.JSONDecodeError: If the stored payload is not valid JSON.
        """
        with self._connect() as conn:
            with conn.blobopen(self.TABLE, "report", index) as blob:
                data = blob.read()
        return json.loads(data.decode("utf-8"))

    def get_last_report(self, **kwargs) -> dict | None:
        """Return the newest report matching the provided filters.

        Examples:
            ``db.get_last_report(name="test_1")``
            ``db.get_last_report(name="hda", tags="test Monday")``
            ``db.get_last_report(module="my.cool.flowsheet")``

        Args:
            **kwargs: Target metadata filters keyed by names in
                :attr:`TARGET_COLUMNS`, plus the optional ``tags`` keyword. If
                ``tags`` is provided, its value must be a string of
                space-separated tags that must all be present in the report.

        Returns:
            dict | None: The newest matching report, or ``None`` if no report
            matches the filters.

        Raises:
            sqlite3.Error: If SQLite cannot execute the query or read the
                report blob.
            json.JSONDecodeError: If the stored payload is not valid JSON.
        """
        # Extract 'tags' from kwargs (may be None)
        # (Don't put tags=None in function signature, otherwise it
        # will confusingly be a positional argument as well)
        tags = kwargs.pop("tags", None)

        # connect to db
        with self._connect() as conn:
            # build query
            stmt = f"SELECT MAX(id) FROM {self.TABLE}"
            stmt += self._where(kwargs, tags=tags)
            # run query
            index = conn.execute(stmt).fetchone()[0]
            # if none, done; else read the report
            if index is None:
                return None  # RETURN!
            with conn.blobopen(self.TABLE, "report", int(index)) as blob:
                data = blob.read()

        # parse report into dict before returning it
        return json.loads(data.decode("utf-8"))

    def _where(self, fltr, tags=None):
        expr = []
        for col in self._tgtcol:
            if col in fltr:
                expr.append(f"{col} = '{fltr[col]}'")
            elif self._tgtval[col]:
                expr.append(f"{col} = '{self._tgtval[col]}'")
        if tags:
            tag_items = [t.lower() for t in tags.split()]
            tag_items.sort()
            pattern = "%" + "%".join(tag_items) + "%"
            expr.append(f"tags LIKE '{pattern}'")
        if expr:
            conj = " AND ".join(expr)
            clause = f" WHERE {conj}"
        else:
            clause = ""  # nothing at all
        return clause
