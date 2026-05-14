"""Results-database access split by read/write boundary.

`DbReader` opens the SQLite file in read-only mode and exposes the
query methods used by Search threads, report handlers, and any
read-only CLI. `DbWriter` opens read-write and owns the write
transactions (`add_run`, `save_model`, the predicted-estimate
upserts, etc.). The two classes share helpers / SQL constants /
return-value dataclasses out of `texmo.db.common`.

The boundary is currently enforced by class membership only — both
classes can still be created in any thread. Step 6 of the threading
refactor moves writes onto a dedicated DBWriter thread; this split
pins down the read/write boundary so that step doesn't have to
guess which methods cross it.
"""

from .reader import ConfScore, ConfWithRuns, DbReader
from .writer import DbWriter

__all__ = ['ConfScore', 'ConfWithRuns', 'DbReader', 'DbWriter']
