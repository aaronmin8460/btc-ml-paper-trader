from sqlalchemy import create_engine, text

from app.db.database import connect_args_for_database_url


def test_sqlite_engine_creation_uses_sqlite_connect_args(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'trading.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))

    try:
        with engine.connect() as connection:
            assert connection.execute(text("select 1")).scalar_one() == 1
    finally:
        engine.dispose()


def test_non_sqlite_database_url_uses_empty_connect_args():
    database_url = "postgresql+psycopg://user:password@localhost:5432/trading"

    assert connect_args_for_database_url(database_url) == {}
