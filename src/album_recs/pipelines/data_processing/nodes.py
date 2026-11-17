"""
Pipeline 'data_processing' nodes for raw CSV and SQL dump ingestion.
Generated using Kedro 1.8.0
"""

import ast
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# --- Tracks CSV helpers & node ---


def _parse_first_item(x: pd.Series) -> pd.Series:
    """
    Extract the lead artist/ID (index 0) from stringified Python lists.
    """
    return x.apply(
        lambda val: ast.literal_eval(val)[0] if pd.notna(val) and val != "[]" else None
    )


def _parse_release_date(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, format="mixed", dayfirst=True, errors="coerce")


def _parse_year(x: pd.Series) -> pd.Series:
    year = pd.to_numeric(x, errors="coerce")
    year = year.where(
        year > 0, pd.NA
    )  # raw CSV uses 0 as placeholder when release year unknown
    return year.astype("Int64")


def preprocess_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Kaggle tracks CSV: rename ID columns, extract primary artist, and fix dates.
    """
    df = df.rename(
        columns={
            "id": "track_id",
            "name": "track_name",
            "album": "album_name",
        }
    )

    df = df.dropna(
        subset=["track_id", "album_id", "track_name", "album_name", "artists"]
    ).drop_duplicates(subset=["track_id"])

    df["primary_artist"] = _parse_first_item(df["artists"])
    df["primary_artist_id"] = _parse_first_item(df["artist_ids"])
    df = df.drop(columns=["artists", "artist_ids"])

    df["release_date"] = _parse_release_date(df["release_date"])
    df["year"] = _parse_year(df["year"])

    return df


# --- Playlist2vec SQL dump helpers & streaming node ---

TOKEN_RE = re.compile(r"\(|\)|'((?:\\.|''|[^'\\])*)'|([^,()\s';]+)")
FAST_PAIR_RE = re.compile(r"\((?:'([^']*)'|NULL),(?:'([^']*)'|NULL)\)")

DEFAULT_SQL_SCHEMAS = {
    "playlist": ["id", "name", "followers", "uri", "total_tracks"],
    "track": [
        "id",
        "name",
        "duration",
        "popularity",
        "explicit",
        "preview_url",
        "uri",
        "album_id",
    ],
    "album": ["id", "name", "uri"],
    "track_playlist1": ["track_id", "playlist_id"],
}

# Only retain the 4 tables needed for album-level recommendations (skip artist tables)
OUTPUT_FILENAMES = {
    "playlist": "preprocessed_playlists.pq",
    "track": "preprocessed_sql_tracks.pq",
    "album": "preprocessed_sql_albums.pq",
    "track_playlist1": "preprocessed_playlist_tracks.pq",
}

ARROW_SCHEMAS = {
    "playlist": pa.schema(
        [
            ("playlist_id", pa.string()),
            ("playlist_name", pa.string()),
            ("followers", pa.int64()),
            ("total_tracks", pa.int64()),
        ]
    ),
    "track": pa.schema(
        [
            ("track_id", pa.string()),
            ("track_name", pa.string()),
            ("duration_ms", pa.int64()),
            ("popularity", pa.int64()),
            ("explicit", pa.int64()),
            ("album_id", pa.string()),
        ]
    ),
    "album": pa.schema(
        [
            ("album_id", pa.string()),
            ("album_name", pa.string()),
        ]
    ),
    "track_playlist1": pa.schema(
        [
            ("track_id", pa.string()),
            ("playlist_id", pa.string()),
        ]
    ),
}


def _parse_nullable_int(x: pd.Series) -> pd.Series:
    """
    Turn text numbers into proper integers while keeping missing values as NA.
    """
    return pd.to_numeric(x, errors="coerce").astype("Int64")


def preprocess_playlists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the playlists table: rename ID/name columns, drop broken rows, and turn counts into integers.
    """
    df = df.rename(columns={"id": "playlist_id", "name": "playlist_name"})
    df = df.replace("", pd.NA)
    df = df.dropna(subset=["playlist_id"]).drop_duplicates(subset=["playlist_id"])
    df = df.drop(columns=["uri"], errors="ignore")

    df["followers"] = _parse_nullable_int(df["followers"])
    df["total_tracks"] = _parse_nullable_int(df["total_tracks"])
    return df[["playlist_id", "playlist_name", "followers", "total_tracks"]]


def preprocess_sql_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the SQL tracks table: rename columns, remove local MP3 rows with empty IDs, and convert numbers to integers.
    """
    df = df.rename(
        columns={"id": "track_id", "name": "track_name", "duration": "duration_ms"}
    )
    df = df.replace("", pd.NA)
    df = df.dropna(subset=["track_id", "album_id", "track_name"]).drop_duplicates(
        subset=["track_id"]
    )
    df = df.drop(columns=["uri", "preview_url"], errors="ignore")

    df["duration_ms"] = _parse_nullable_int(df["duration_ms"])
    df["popularity"] = _parse_nullable_int(df["popularity"])
    df["explicit"] = _parse_nullable_int(df["explicit"])
    return df[
        ["track_id", "track_name", "duration_ms", "popularity", "explicit", "album_id"]
    ]


def preprocess_sql_albums(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the SQL albums table: rename ID/name columns and drop rows with missing or empty album info.
    """
    df = df.rename(columns={"id": "album_id", "name": "album_name"})
    df = df.replace("", pd.NA)
    df = df.dropna(subset=["album_id", "album_name"]).drop_duplicates(
        subset=["album_id"]
    )
    df = df.drop(columns=["uri"], errors="ignore")
    return df[["album_id", "album_name"]]


def preprocess_playlist_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the playlist-track link table by removing empty IDs and duplicate songs within the same playlist.
    """
    df = df.dropna(subset=["track_id", "playlist_id"])
    df = df[(df["track_id"] != "") & (df["playlist_id"] != "")]
    return df.drop_duplicates(subset=["track_id", "playlist_id"])[
        ["track_id", "playlist_id"]
    ]


TABLE_CLEANERS = {
    "playlist": (preprocess_playlists, "playlist_id"),
    "track": (preprocess_sql_tracks, "track_id"),
    "album": (preprocess_sql_albums, "album_id"),
    "track_playlist1": (preprocess_playlist_tracks, None),
}


def _iter_mysql_rows(values_str: str):
    """
    Read a MySQL VALUES line and yield one clean list of values per row, handling quotes and NULLs.
    """
    row = None
    for m in TOKEN_RE.finditer(values_str):
        tok = m.group(0)
        if tok == "(":
            row = []
        elif tok == ")":
            if row is not None:
                yield row
                row = None
        elif row is not None:
            if m.group(1) is not None:
                val = (
                    m.group(1)
                    .replace("''", "'")
                    .replace("\\'", "'")
                    .replace('\\"', '"')
                    .replace("\\n", "\n")
                    .replace("\\r", "\r")
                    .replace("\\t", "\t")
                    .replace("\\\\", "\\")
                )
                row.append(val)
            else:
                raw = m.group(2)
                row.append(None if raw == "NULL" else raw)


def preprocess_playlists_dump(
    sql_path: Path, intermediate_dir: str, batch_size: int = 250_000
) -> tuple[Path, Path, Path, Path]:
    """
    Read the 10.7 GB SQL dump line by line in 250k-row batches, clean each table, and save as 4 Parquet files.
    """
    out_dir = Path(intermediate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {t: out_dir / fname for t, fname in OUTPUT_FILENAMES.items()}
    schemas = {t: list(cols) for t, cols in DEFAULT_SQL_SCHEMAS.items()}
    buffers = {t: [] for t in OUTPUT_FILENAMES}
    seen_ids = {"playlist": set(), "track": set(), "album": set()}
    written_counts = {t: 0 for t in OUTPUT_FILENAMES}
    seen_tables = set()
    writers = {}

    def _flush_table(table: str) -> None:
        """
        Clean the current batch of rows for one table, drop IDs seen in earlier batches, and append to its Parquet file.
        """
        if not buffers[table]:
            return
        raw_df = pd.DataFrame(buffers[table], columns=schemas[table])
        buffers[table].clear()

        cleaner_fn, pk_col = TABLE_CLEANERS[table]
        clean_df = cleaner_fn(raw_df)

        if pk_col is not None:
            clean_df = clean_df[~clean_df[pk_col].isin(seen_ids[table])]
            seen_ids[table].update(clean_df[pk_col].tolist())

        if clean_df.empty:
            return

        pa_table = pa.Table.from_pandas(
            clean_df, schema=ARROW_SCHEMAS[table], preserve_index=False
        )
        if table not in writers:
            writers[table] = pq.ParquetWriter(output_paths[table], ARROW_SCHEMAS[table])
        writers[table].write_table(pa_table)

        prev_millions = written_counts[table] // 1_000_000
        written_counts[table] += len(clean_df)
        if written_counts[table] // 1_000_000 > prev_millions:
            logger.info(
                "Progress — %s: %s rows written...",
                table,
                f"{written_counts[table]:,}",
            )

    current_table = None
    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.upper().startswith("CREATE TABLE"):
                match = re.search(
                    r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+`?(\w+)`?", stripped, re.I
                )
                if match and match.group(1) in OUTPUT_FILENAMES:
                    current_table = match.group(1)
                    schemas[current_table] = []
                else:
                    current_table = None
            elif current_table is not None:
                if stripped.startswith(")"):
                    current_table = None
                elif stripped.startswith("`"):
                    schemas[current_table].append(stripped.split("`")[1])
            elif stripped.startswith("INSERT INTO"):
                table = stripped.split("`")[1]
                if table not in seen_tables:
                    seen_tables.add(table)
                    logger.info(
                        "Scanning SQL table `%s` (%s)...",
                        table,
                        "keeping" if table in OUTPUT_FILENAMES else "skipping",
                    )
                if table not in OUTPUT_FILENAMES:
                    continue

                values_part = re.split(r"\bVALUES\s*", stripped, maxsplit=1)[1]
                if table == "track_playlist1":
                    buffers[table].extend(FAST_PAIR_RE.findall(values_part))
                else:
                    buffers[table].extend(_iter_mysql_rows(values_part))

                if len(buffers[table]) >= batch_size:
                    _flush_table(table)

    for table in OUTPUT_FILENAMES:
        _flush_table(table)
        if table in writers:
            writers[table].close()

    logger.info("Finished SQL dump preprocessing. Rows written: %s", written_counts)

    return (
        output_paths["playlist"],
        output_paths["track"],
        output_paths["album"],
        output_paths["track_playlist1"],
    )
