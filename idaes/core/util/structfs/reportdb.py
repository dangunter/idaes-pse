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

import sqlite3
import time
import json


class ReportDB:
    TABLE = "reports"
    TARGET_COLUMNS = (
        ("module", "TEXT"),
        ("filename", "TEXT"),
        ("main_func", "TEXT"),
        ("fs_object", "TEXT"),
    )
    COLUMNS = tuple(
        [
            ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
            ("created", "INTEGER"),
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
        """Set current target (file, module, etc.)"""
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
        with self._connect() as conn:
            # set non-user column values
            created = int(time.time())
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
        columns = ", ".join(self._all_columns(exclude=("report",)))
        stmt = f"SELECT {columns} from {self.TABLE}"
        stmt += self._where(target_kw, tags=tags)
        with self._connect() as conn:
            for row in conn.execute(stmt):
                yield row

    def get_report(self, index) -> str:
        with self._connect() as conn:
            with conn.blobopen(self.TABLE, "report", index) as blob:
                data = blob.read()
        return json.loads(data.decode("utf-8"))

    def get_last_report(self, tags: str = "", **target_kw) -> dict | None:
        with self._connect() as conn:
            stmt = f"SELECT MAX(id) FROM {self.TABLE}"
            stmt += self._where(target_kw, tags=tags)
            index = conn.execute(stmt).fetchone()[0]
            if index is None:
                return None
            with conn.blobopen(self.TABLE, "report", int(index)) as blob:
                data = blob.read()
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
