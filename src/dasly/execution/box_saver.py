import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.types import Float, Integer, DateTime, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)


def denormalize_boxesn(
    boxesn: np.ndarray,
    t_start: float,
    t_end: float,
    n_start: float,
    n_end: float,
) -> np.ndarray:
    """Convert detected boxes from normalized coordinates to absolute
    coordinates in the spatial and temporal domains.

    Args:
        boxesn (np.ndarray): Array of shape (n, 4) with rows [n1_norm, t1_norm,
            n2_norm, t2_norm], where n denotes channel and t temporal values.
        t_start (float): Start timestamp.
        t_end (float): End timestamp.
        n_start (float): Start value for the spatial channel.
        n_end (float): End value for the spatial channel.

    Returns:
        np.ndarray: Array with absolute coordinates.
    """
    boxes = boxesn.astype(np.float64, copy=True)
    boxes[:, 0] = (boxes[:, 0] * (n_end - n_start)) + n_start  # n1
    boxes[:, 2] = (boxes[:, 2] * (n_end - n_start)) + n_start  # n2
    boxes[:, 1] = (boxes[:, 1] * (t_end - t_start)) + t_start  # t1
    boxes[:, 3] = (boxes[:, 3] * (t_end - t_start)) + t_start  # t2
    return boxes


def cast_box_times_to_datetime64(boxes: np.ndarray) -> np.ndarray:
    """Convert temporal columns (t1, t2) in a (n, 4) array to numpy.datetime64
    objects, while leaving channel columns (n1, n2) unchanged.

    Args:
        boxes (np.ndarray): Array of shape (n, 4) with rows [n1, t1, n2, t2]
            where t1 and t2 are timestamps (in seconds).

    Returns:
        np.ndarray: Object array with t1 and t2 as datetime64 objects.
    """
    boxes_dt = boxes.copy().astype(object)
    boxes_dt[:, 1] = pd.to_datetime(boxes[:, 1], unit='s', utc=True)
    boxes_dt[:, 3] = pd.to_datetime(boxes[:, 3], unit='s', utc=True)
    boxes_dt[:, 0] = boxes[:, 0].astype(np.float64)
    boxes_dt[:, 2] = boxes[:, 2].astype(np.float64)
    return boxes_dt


def build_create_table_sql(
    table_name: str,
    dtype_mapping: dict,
    dialect=postgresql.dialect()
) -> str:
    """Build a CREATE TABLE SQL statement based on dtype_mapping."""
    local_mapping = dtype_mapping.copy()
    local_mapping.setdefault("id", Integer)
    local_mapping.setdefault("parent_id", Integer)
    local_mapping.setdefault("created_at", lambda: DateTime(timezone=True))
    local_mapping.setdefault("updated_at", lambda: DateTime(timezone=True))

    columns_sql_parts = []

    col = "id"
    compiled_type = "SERIAL"
    col_def = f"{col} {compiled_type} PRIMARY KEY"
    columns_sql_parts.append(col_def)

    col = "parent_id"
    type_obj = local_mapping[col]
    compiled_type = type_obj().compile(dialect=dialect)
    col_def = f"{col} {compiled_type}"
    columns_sql_parts.append(col_def)

    for col in local_mapping:
        if col in ["id", "parent_id", "created_at", "updated_at"]:
            continue
        type_obj = local_mapping[col]
        compiled_type = type_obj().compile(dialect=dialect)
        col_def = f"{col} {compiled_type}"
        columns_sql_parts.append(col_def)

    for col in ["created_at", "updated_at"]:
        type_obj = local_mapping[col]
        compiled_type = type_obj().compile(dialect=dialect)
        col_def = f"{col} {compiled_type} DEFAULT CURRENT_TIMESTAMP"
        columns_sql_parts.append(col_def)

    create_table_sql = (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n    " +
        ",\n    ".join(columns_sql_parts) +
        "\n);"
    )
    return create_table_sql


def create_table_with_triggers(
    table_name: str,
    engine: Engine,
    dtype_mapping: dict
) -> None:
    """Create a table based on dtype_mapping and add triggers for updating
    'updated_at' and 'parent_id' columns."""
    with engine.connect() as conn:
        create_table_sql = build_create_table_sql(
            table_name, dtype_mapping, conn.dialect)
        conn.execute(text(create_table_sql))
        conn.commit()

        trigger_function_sql = """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
        conn.execute(text(trigger_function_sql))
        conn.commit()

        trigger_sql = f"""
        DROP TRIGGER IF EXISTS update_updated_at_trigger ON {table_name};
        CREATE TRIGGER update_updated_at_trigger
        BEFORE UPDATE ON {table_name}
        FOR EACH ROW
        EXECUTE PROCEDURE update_updated_at_column();
        """
        conn.execute(text(trigger_sql))
        conn.commit()

        trigger_function_sql = """
        CREATE OR REPLACE FUNCTION set_parent_id_from_id()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.parent_id IS NULL THEN
                NEW.parent_id := NEW.id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
        conn.execute(text(trigger_function_sql))
        conn.commit()

        trigger_sql = f"""
        DROP TRIGGER IF EXISTS set_parent_id_trigger ON {table_name};
        CREATE TRIGGER set_parent_id_trigger
        BEFORE INSERT ON {table_name}
        FOR EACH ROW
        EXECUTE PROCEDURE set_parent_id_from_id();
        """
        conn.execute(text(trigger_sql))
        conn.commit()


def _quote_table_name(table_name: str, engine: Engine) -> str:
    preparer = engine.dialect.identifier_preparer
    if "." in table_name:
        schema, table = table_name.split(".", 1)
        return f"{preparer.quote(schema)}.{preparer.quote(table)}"
    return preparer.quote(table_name)


def _normalize_columns_for_table(
    df: pd.DataFrame,
    table_columns: list[str] | None
) -> pd.DataFrame:
    if table_columns is None:
        rename = {col: col.lower() for col in df.columns}
    else:
        existing_lower = {col.lower(): col for col in table_columns}
        existing_all_lower = all(col == col.lower() for col in table_columns)
        rename = {}
        for col in df.columns:
            lower = col.lower()
            if lower in existing_lower:
                rename[col] = existing_lower[lower]
            elif existing_all_lower:
                rename[col] = lower
            else:
                rename[col] = col

    new_cols = list(rename.values())
    if len(set(new_cols)) != len(new_cols):
        collisions = sorted(
            {col for col in new_cols if new_cols.count(col) > 1}
        )
        raise ValueError(
            "Column name collision after normalization: "
            f"{', '.join(collisions)}"
        )

    return df.rename(columns=rename)


def _ensure_table_has_columns(
    table_name: str,
    engine: Engine,
    dtype_mapping: dict,
    existing_columns: list[str]
) -> None:
    missing = [col for col in dtype_mapping if col not in existing_columns]
    if not missing:
        return

    table_sql = _quote_table_name(table_name, engine)
    preparer = engine.dialect.identifier_preparer

    with engine.begin() as conn:
        for col in missing:
            type_def = dtype_mapping[col]
            try:
                type_obj = type_def()
            except TypeError:
                type_obj = type_def
            compiled_type = type_obj.compile(dialect=engine.dialect)
            col_sql = preparer.quote(col)
            alter_sql = (
                f"ALTER TABLE {table_sql} "
                f"ADD COLUMN {col_sql} {compiled_type}"
            )
            conn.execute(text(alter_sql))


def map_numpy_dtype_to_sqla(np_dtype: np.dtype) -> type:
    """Map a NumPy dtype to a corresponding SQLAlchemy type."""
    if is_datetime64tz_dtype(np_dtype):
        return lambda: DateTime(timezone=True)
    if is_float_dtype(np_dtype):
        return Float
    if is_integer_dtype(np_dtype):
        return Integer
    if is_datetime64_any_dtype(np_dtype):
        return DateTime
    if is_string_dtype(np_dtype):
        return String

    try:
        if np.issubdtype(np_dtype, np.floating):
            return Float
        if np.issubdtype(np_dtype, np.integer):
            return Integer
        if np.issubdtype(np_dtype, np.datetime64):
            return DateTime
    except TypeError:
        pass

    return String


def auto_dtype_mapping(df: pd.DataFrame) -> dict:
    """Automatically create a dtype mapping for a DataFrame."""
    mapping = {col: map_numpy_dtype_to_sqla(dtype)
               for col, dtype in df.dtypes.to_dict().items()}
    return mapping


def build_box_df(
    boxesp: np.ndarray,
    boxesn: np.ndarray,
    chunk: str,
    chunk_size: int,
    additional: dict = None
) -> pd.DataFrame:
    """Create a DataFrame from detected boxes and additional information.

    Args:
        boxesp (np.ndarray): Array of shape (n, 4) with rows [n1, t1, n2, t2].
        boxesn (np.ndarray): Array of shape (n, 4) with normalized coordinates.
        chunk (str): Chunk identifier.
        chunk_size (int): Size of the chunk (number of files).
        additional (dict, optional): Additional columns to add.
    """
    n = boxesp.shape[0]

    df = pd.DataFrame({
        'n1': boxesp[:, 0],
        't1': boxesp[:, 1],
        'n2': boxesp[:, 2],
        't2': boxesp[:, 3],
        'x1n': boxesn[:, 0],
        'y1n': boxesn[:, 1],
        'x2n': boxesn[:, 2],
        'y2n': boxesn[:, 3],
        'chunk': [chunk] * n,
        'chunk_size': [chunk_size] * n
    })
    df['n1'] = df['n1'].astype(np.float64)
    df['n2'] = df['n2'].astype(np.float64)

    if additional is not None:
        additional_df = pd.DataFrame(additional)
        df = pd.concat([df, additional_df], axis=1)

    return df


def save_to_db(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str
) -> None:
    """Save DataFrame to a PostgreSQL database. If the table does not exist,
    it will be created with the appropriate columns and types."""
    if df.empty:
        print("DataFrame is empty. No data to save.")
        return

    engine = create_engine(connection_string)

    inspector = inspect(engine)
    table_exists = inspector.has_table(table_name)
    if table_exists:
        existing_columns = [
            col["name"] for col in inspector.get_columns(table_name)
        ]
        df = _normalize_columns_for_table(df, existing_columns)
    else:
        df = _normalize_columns_for_table(df, None)

    dtype_mapping = auto_dtype_mapping(df)

    if not table_exists:
        print(f"Table '{table_name}' does not exist. Creating it now.")
        create_table_with_triggers(
            table_name=table_name, engine=engine, dtype_mapping=dtype_mapping)
    else:
        _ensure_table_has_columns(
            table_name=table_name,
            engine=engine,
            dtype_mapping=dtype_mapping,
            existing_columns=existing_columns,
        )

    sqlalchemy_dtype_mapping = {}
    for col, type_def in dtype_mapping.items():
        try:
            sqlalchemy_dtype_mapping[col] = type_def()
        except TypeError:
            sqlalchemy_dtype_mapping[col] = type_def
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        dtype=sqlalchemy_dtype_mapping
    )

    engine.dispose()
