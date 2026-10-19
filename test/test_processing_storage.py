"""Failed checkpoint writes retain original state and remove only their temp."""
import errno
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from app.repository.processing_jobs import save


class ProcessingStorageTests(TestCase):
    def test_enospc_after_partial_write_preserves_original_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "job.json"
            original = b'{"status":"running","completed":15}'
            path.write_bytes(original)
            unrelated = Path(root) / "another.tmp"
            unrelated.write_text("retain")
            write_text = Path.write_text

            def failing_write(temporary, *args, **kwargs):
                write_text(temporary, "partial", encoding="utf-8")
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch.object(Path, "write_text", failing_write):
                with self.assertRaises(OSError) as failure:
                    save(path, {"status": "queued"})
            self.assertEqual(failure.exception.errno, errno.ENOSPC)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(root).glob("job.json.*.tmp")), [])
            self.assertEqual(unrelated.read_text(), "retain")

    def test_replace_failure_preserves_original_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "job.json"
            path.write_text('{"status":"running"}')
            with patch.object(Path, "replace", side_effect=OSError(errno.EACCES, "Permission denied")):
                with self.assertRaises(OSError):
                    save(path, {"status": "queued"})
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertEqual(list(Path(root).glob("*.tmp")), [])

    def test_success_atomically_replaces_original_without_temp(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "job.json"
            path.write_text('{"status":"queued"}')
            save(path, {"status": "done", "label": "中文"})
            self.assertEqual(json.loads(path.read_text()), {"status": "done", "label": "中文"})
            self.assertEqual(list(Path(root).glob("*.tmp")), [])

    def test_low_capacity_stops_before_creating_a_temporary_file(self):
        from collections import namedtuple
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "job.json"
            path.write_text('{"status":"running"}')
            usage = namedtuple("usage", "total used free")(500, 499, 1)
            with patch("app.repository.processing_jobs.shutil.disk_usage", return_value=usage):
                with self.assertRaises(OSError) as failure:
                    save(path, {"status": "queued"})
            self.assertEqual(failure.exception.errno, errno.ENOSPC)
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertEqual(list(Path(root).glob("*.tmp")), [])
