"""Read public BCMID metadata and ZIP directory using bounded range requests.

Does not upload anything, extract patient images, or construct clinical labels.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
LIMIT = 50 * 1024 * 1024


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, n=-1):
        n = min(self.size-self.position, n if n >= 0 else self.size-self.position)
        if n <= 0:
            return b""
        if self.transferred+n > LIMIT:
            raise RuntimeError("Audit download ceiling reached")
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{self.position+n-1}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise RuntimeError("Server does not honor byte ranges; refusing full archive")
            data = response.read(n+1)
            if len(data) != n:
                raise RuntimeError("Unexpected range length")
        self.position += n
        self.transferred += n
        return data


def main():
    folder = ROOT / "results/bcmid-audit"
    folder.mkdir(parents=True, exist_ok=True)
    api = "https://zenodo.org/api/records/19402690"
    with urllib.request.urlopen(api, timeout=30) as response:
        metadata = json.load(response)
    files = {x["key"]: x for x in metadata["files"]}
    tsv_url = files["BCMID_annotations.tsv"]["links"]["self"]
    with urllib.request.urlopen(tsv_url, timeout=30) as response:
        tsv = response.read(1024*1024)
    if len(tsv) != files["BCMID_annotations.tsv"]["size"]:
        raise RuntimeError("Annotation length mismatch")
    (folder / "BCMID_annotations.tsv").write_bytes(tsv)
    text = tsv.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    info = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": 19402690, "record_api": api,
        "version": metadata["metadata"].get("version"),
        "license": metadata["metadata"].get("license"),
        "access_right": metadata["metadata"].get("access_right"),
        "description": metadata["metadata"].get("description"),
        "files": [{"name": x["key"], "size": x["size"], "checksum": x.get("checksum")} for x in files.values()],
        "annotation_header": rows[0], "annotation_data_rows": len(rows)-1,
        "annotation_sha256": hashlib.sha256(tsv).hexdigest(),
        "annotation_columns": len(rows[0]),
        "zip_directory_status": "NOT_READ",
        "download_cap_bytes": LIMIT,
        "clinical_training_gate": "NOT_MET_MICROCALCIFICATION_REFERENCE_UNCONFIRMED",
    }
    try:
        remote = RemoteZip(files["BCMID.zip"]["links"]["self"], files["BCMID.zip"]["size"])
        with zipfile.ZipFile(remote) as archive:
            entries = [{"name": x.filename, "size": x.file_size, "compressed_size": x.compress_size} for x in archive.infolist()]
        (folder / "zip-directory.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
        suffixes = {}
        for x in entries:
            suffix = Path(x["name"]).suffix.lower() or "directory_or_no_suffix"
            suffixes[suffix] = suffixes.get(suffix, 0)+1
        annotation_names = [x["name"] for x in entries if Path(x["name"]).suffix.lower() in (".csv", ".tsv", ".json", ".xml", ".xlsx", ".xls")]
        info.update(zip_directory_status="READ", zip_entries=len(entries), zip_suffix_counts=suffixes,
                    annotation_like_member_count=len(annotation_names), transferred_zip_bytes=remote.transferred,
                    annotation_like_member_names=annotation_names)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        info.update(zip_directory_status="UNAVAILABLE", zip_directory_error=str(error))
    (folder / "audit.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k not in ("description", "annotation_like_member_names")}, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
