from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def connect_args_for_database_url(database_url: str) -> dict[str, bool]:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


settings = get_settings()
engine = create_engine(settings.database_url, connect_args=connect_args_for_database_url(settings.database_url))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    run_sqlite_schema_migrations(engine)


def run_sqlite_schema_migrations(bind_engine=None) -> None:
    target_engine = bind_engine or engine
    if target_engine.dialect.name != "sqlite":
        return
    additions = {
        "signals": {
            "spread_bps": "FLOAT",
            "quote_imbalance": "FLOAT",
            "model_version": "VARCHAR(128)",
        },
        "collected_market_data": {
            "source": "VARCHAR(64)",
            "source_used": "VARCHAR(64)",
            "backfilled": "BOOLEAN DEFAULT 0",
            "provider_metadata": "TEXT",
        },
        "trades": {
            "entry_price": "FLOAT",
            "exit_price": "FLOAT",
            "notional": "FLOAT",
            "gross_pnl": "FLOAT",
            "net_pnl": "FLOAT",
            "fee_amount": "FLOAT",
            "slippage_amount": "FLOAT",
            "hold_seconds": "FLOAT",
            "reason": "TEXT",
        },
    }
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())
    with target_engine.begin() as connection:
        for table_name, columns in additions.items():
            if table_name not in existing_tables:
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, sql_type in columns.items():
                if column_name not in existing:
                    connection.exec_driver_sql(
                        f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {sql_type}'
                    )
        if "collected_market_data" in existing_tables:
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ux_collected_market_data_symbol_timeframe_timestamp "
                "ON collected_market_data(symbol, timeframe, timestamp)"
            )


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
