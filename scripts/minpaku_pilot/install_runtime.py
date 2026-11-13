#!/usr/bin/env python3
"""Install only the approved, pinned Ollama archive in an isolated user directory.

No sudo, packages, accounts, shell profiles, global runtime or model downloads.
Requires existing Python 3, zstd and Linux x86_64. Temporary downloads are removed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen


def same_runtime_artifact(recorded, requested):
    """Model selection may change without re-downloading identical verified bytes."""
    keys = ("version", "archive_url", "archive_size", "archive_sha256")
    return isinstance(recorded, dict) and all(recorded.get(key) == requested[key] for key in keys)


def safe_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe archive path")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes installation directory")
    return destination


def disk_bytes(root):
    seen, total = set(), 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            entry = (Path(directory) / name).lstat()
            identity = (entry.st_dev, entry.st_ino)
            if identity not in seen:
                total += entry.st_blocks * 512
                seen.add(identity)
    return total


def link_target(root, member):
    link = PurePosixPath(member.linkname)
    if link.is_absolute():
        raise ValueError("Absolute archive link is not allowed")
    name = PurePosixPath(member.name).parent / link if member.issym() else link
    return safe_path(root, os.path.normpath(str(name)))


def extract_checked(archive, destination, max_bytes=18_000_000_000):
    """Validate every member and link before extracting any file."""
    with tarfile.open(archive, "r:") as source:
        members = source.getmembers()
        if sum(item.size for item in members) > max_bytes:
            raise ValueError("Unpacked runtime exceeds installation budget")
        for item in members:
            target = safe_path(destination, item.name)
            parts = PurePosixPath(item.name).parts
            if parts and parts[0] not in ("bin", "lib"):
                raise ValueError("Unexpected archive root")
            if item.issym() or item.islnk():
                link_target(destination, item)
            elif not (item.isfile() or item.isdir()):
                raise ValueError("Unsupported archive member")
        # Files precede links: a link cannot redirect a later file write.
        for item in members:
            target = safe_path(destination, item.name)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(item) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                target.chmod(0o700 if item.mode & 0o111 else 0o600)
        for item in members:
            target = safe_path(destination, item.name)
            if item.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(item.linkname)
            elif item.islnk():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(link_target(destination, item), target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(Path(__file__).with_name("release.json").read_text())
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        parser.error("This approved artifact is for Linux x86_64 only")
    decoder_path = shutil.which("zstd") or shutil.which("zstd", path=str(Path(sys.executable).parent))
    if not decoder_path:
        parser.error("Existing zstd executable required; do not install without approval")
    if args.project.is_symlink():
        parser.error("Project directory must not be a symlink")
    project = args.project.expanduser().resolve()
    project.mkdir(mode=0o700, parents=True, exist_ok=True)
    if project.stat().st_uid != os.getuid() or stat.S_IMODE(project.stat().st_mode) & 0o077:
        parser.error("Use a user-owned project directory with permissions 0700")
    final = project / "runtime" / pin["version"]
    if final.exists():
        marker = final / "verified-release.json"
        if marker.exists() and same_runtime_artifact(json.loads(marker.read_text()), pin):
            print("Pinned runtime already installed; no download performed")
            return
        parser.error("Runtime path exists without matching verification; inspect it first")
    if shutil.disk_usage(project).free < pin["budget_bytes"]:
        parser.error("At least 25 GB free is required before this pilot installation")
    if disk_bytes(project) + pin["archive_size"] + 8_000_000_000 > pin["budget_bytes"]:
        parser.error("Existing project data leaves insufficient space in the approved total budget")
    final.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=project) as temporary:
        workspace = Path(temporary)
        compressed, archive = workspace / "runtime.tar.zst", workspace / "runtime.tar"
        digest, received = hashlib.sha256(), 0
        with urlopen(pin["archive_url"], timeout=60) as response, compressed.open("xb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > pin["archive_size"]:
                    raise ValueError("Download exceeds pinned release size")
                digest.update(chunk)
                output.write(chunk)
        if received != pin["archive_size"] or digest.hexdigest() != pin["archive_sha256"]:
            raise ValueError("Official release size/checksum mismatch; installation stopped")
        # Bound decompression too: temporary archive + extracted files must fit
        # inside the approved 25 GB total, before any model pulls are allowed.
        decoder = subprocess.Popen([decoder_path, "-d", "-q", "-c", str(compressed)], stdout=subprocess.PIPE)
        try:
            unpacked_bytes = 0
            with archive.open("xb") as output:
                while True:
                    chunk = decoder.stdout.read(1024 * 1024)
                    if not chunk:
                        break
                    unpacked_bytes += len(chunk)
                    if unpacked_bytes > 8_000_000_000:
                        raise ValueError("Decompressed runtime exceeds pilot budget")
                    output.write(chunk)
            if decoder.wait() != 0:
                raise ValueError("Runtime decompression failed")
        finally:
            decoder.stdout.close()
            if decoder.poll() is None:
                decoder.terminate()
                decoder.wait()
        compressed.unlink()
        if disk_bytes(project) + archive.stat().st_size > pin["budget_bytes"]:
            raise ValueError("Runtime extraction would exceed the total project budget")
        unpacked = workspace / "unpacked"
        unpacked.mkdir(mode=0o700)
        extract_checked(archive, unpacked, max_bytes=8_000_000_000)
        if not (unpacked / "bin" / "ollama").is_file():
            raise ValueError("Verified archive does not contain the expected executable")
        (unpacked / "verified-release.json").write_text(json.dumps(pin, indent=2) + "\n")
        unpacked.rename(final)
    print("Installed verified Ollama " + pin["version"] + "; models were not downloaded")


if __name__ == "__main__":
    main()
