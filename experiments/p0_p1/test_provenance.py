"""Check published Git descendants without changing the scientific settings."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.p0_p1 import protocol


class ProvenanceTests(unittest.TestCase):
    def test_base_descendant_unrelated_history_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
                ).strip()

            git("init", "-q")
            git("config", "user.name", "Provenance Test")
            git("config", "user.email", "test@example.invalid")
            git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "base")
            base = git("rev-parse", "HEAD")
            with patch.object(protocol, "BACKBONE_COMMIT", base):
                self.assertEqual(protocol.backbone_commit(root), base)
                git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "experiment")
                self.assertEqual(protocol.backbone_commit(root), base)
                git("checkout", "--orphan", "unrelated")
                git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "unrelated")
                with self.assertRaises(ValueError):
                    protocol.backbone_commit(root)
            archive = root / "archive"
            archive.mkdir()
            self.assertEqual(protocol.backbone_commit(archive), protocol.BACKBONE_COMMIT)


if __name__ == "__main__":
    unittest.main()
