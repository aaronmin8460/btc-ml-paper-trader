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
    _add_sqlite_dashboard_observability_columns()


def _add_sqlite_dashboard_observability_columns() -> None:
    if engine.dialect.name != "sqlite":
        return
    additions = {
        "signals": {
            "spread_bps": "FLOAT",
            "quote_imbalance": "FLOAT",
            "model_version": "VARCHAR(128)",
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
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table_name, columns in additions.items():
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, sql_type in columns.items():
                if column_name not in existing:
                    connection.exec_driver_sql(
                        f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {sql_type}'
                    )


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
