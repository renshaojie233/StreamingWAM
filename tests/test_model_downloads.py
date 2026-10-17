import hashlib
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


class ModelDownloadTests(unittest.TestCase):
    def test_model_downloader_verifies_and_installs_compatible_name(self) -> None:
        from real_robot import download_assets

        def digest(data: bytes) -> str:
            return hashlib.sha256(data).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "streamingwam_realworld.pt").write_bytes(b"abcdef")
            runtime = source / "runtime.tar.gz"
            stats = root / "stats.json"
            stats.write_text("{}", encoding="utf-8")
            with tarfile.open(runtime, "w:gz") as archive:
                archive.add(stats, arcname="train_split_zscore_stats.json")

            original = (
                download_assets.RELEASE_BASE,
                download_assets.CHECKPOINT_SHA256,
                download_assets.RUNTIME_ARCHIVE,
                download_assets.RUNTIME_ARCHIVE_SHA256,
                download_assets.CHECKPOINT_FILE,
                sys.argv,
            )
            try:
                download_assets.RELEASE_BASE = source.as_uri()
                download_assets.CHECKPOINT_SHA256 = digest(b"abcdef")
                download_assets.RUNTIME_ARCHIVE = runtime.name
                download_assets.RUNTIME_ARCHIVE_SHA256 = download_assets.sha256(runtime)
                download_assets.CHECKPOINT_FILE = "streamingwam_realworld.pt"
                sys.argv = ["download_assets.py", "--output", str(output)]
                download_assets.main()
            finally:
                (
                    download_assets.RELEASE_BASE,
                    download_assets.CHECKPOINT_SHA256,
                    download_assets.RUNTIME_ARCHIVE,
                    download_assets.RUNTIME_ARCHIVE_SHA256,
                    download_assets.CHECKPOINT_FILE,
                    sys.argv,
                ) = original
            self.assertEqual((output / "step_030000.pt").read_bytes(), b"abcdef")
            self.assertEqual((output / "train_split_zscore_stats.json").read_text(), "{}")

    def test_rejects_corrupt_download(self):
        from real_robot import download_assets
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / "source"; source.mkdir()
            (source / "weight.pt").write_bytes(b"bad")
            with patch.object(download_assets, "RELEASE_BASE", source.as_uri()):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    download_assets.download_checked("weight.pt", "not-the-digest", root)
            self.assertFalse((root / "weight.pt").exists())

    def test_valid_cached_checkpoint_does_not_need_network(self):
        from real_robot import download_assets
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "step_030000.pt").write_bytes(b"cached")
            with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected download")):
                result = download_assets.download_checked("streamingwam_realworld.pt",
                    hashlib.sha256(b"cached").hexdigest(), root, local_name="step_030000.pt")
            self.assertEqual(result, root / "step_030000.pt")


if __name__ == "__main__":
    unittest.main()
