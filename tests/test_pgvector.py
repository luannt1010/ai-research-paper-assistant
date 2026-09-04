from unittest.mock import MagicMock

import psycopg
import pytest

from src.database.pgvector_storage import PGVectorStore


# ============================================================
# HELPER
# ============================================================

def normalize_sql(query: str) -> str:
    """Normalize whitespace in SQL for easier assertions."""
    return " ".join(query.split())


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def mock_cursor():
    return MagicMock()


@pytest.fixture
def mock_conn(mock_cursor):
    conn = MagicMock()

    # Support:
    #
    # with self.conn.cursor() as cursor:
    #
    conn.cursor.return_value.__enter__.return_value = mock_cursor
    conn.cursor.return_value.__exit__.return_value = False

    return conn


@pytest.fixture
def store(mock_conn, mocker):
    """
    Prevent real PostgreSQL connection.
    """

    mocker.patch.object(
        PGVectorStore,
        "_get_connection",
        return_value=mock_conn
    )

    return PGVectorStore()


# ============================================================
# _BUILD_FILTER
# ============================================================

def test_build_filter_none(store):
    """
    None = no filtering.
    """

    where_clause, params = store._parse_filters(None)

    assert where_clause == ""
    assert params == []


def test_build_filter_empty_dict_raises(store):
    """
    {} means filters was supplied but contains no condition.
    """

    with pytest.raises(
        ValueError,
        match="Filters cannot be empty"
    ):
        store._parse_filters({})


def test_build_filter_document_id(store):
    filters = {
        "document_id": "paper_001"
    }

    where_clause, params = store._parse_filters(filters)

    assert normalize_sql(where_clause) == (
        "WHERE document_id = %s"
    )

    assert params == [
        "paper_001"
    ]


def test_build_filter_metadata(store):
    filters = {
        "page": 5
    }

    where_clause, params = store._parse_filters(filters)

    assert normalize_sql(where_clause) == (
        "WHERE metadata->>%s = %s"
    )

    assert params == [
        "page",
        "5"
    ]


def test_build_filter_multiple_filters(store):
    filters = {
        "document_id": "paper_001",
        "page": 5,
        "section": "Method"
    }

    where_clause, params = store._parse_filters(filters)

    assert normalize_sql(where_clause) == (
        "WHERE document_id = %s "
        "AND metadata->>%s = %s "
        "AND metadata->>%s = %s"
    )

    assert params == [
        "paper_001",
        "page",
        "5",
        "section",
        "Method"
    ]


# ============================================================
# SEARCH
# ============================================================

def test_search_without_filter(
    store,
    mock_cursor
):
    """
    filters=None -> search entire vector store.
    """

    query_embedding = [
        0.1,
        0.2,
        0.3
    ]

    expected_result = [
        {
            "id": 1,
            "document_id": "paper_001",
            "content": "Example document",
            "metadata": {
                "page": 1
            },
            "similarity": 0.95
        }
    ]

    mock_cursor.fetchall.return_value = expected_result

    result = store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters=None
    )

    assert result == expected_result

    mock_cursor.execute.assert_called_once()

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "FROM chunks_db" in query
    assert "WHERE" not in query
    assert "ORDER BY embedding <=>" in query
    assert "LIMIT %s" in query

    assert list(params) == [
        query_embedding,
        query_embedding,
        5
    ]


def test_search_empty_filter_raises(
    store
):
    with pytest.raises(
        ValueError,
        match="Filters cannot be empty"
    ):
        store.search(
            query_embedding=[
                0.1,
                0.2,
                0.3
            ],
            filters={}
        )


def test_search_with_document_id_filter(
    store,
    mock_cursor
):
    query_embedding = [
        0.1,
        0.2,
        0.3
    ]

    mock_cursor.fetchall.return_value = []

    result = store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters={
            "document_id": "paper_001"
        }
    )

    assert result == []

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "WHERE document_id = %s" in query

    assert list(params) == [
        query_embedding,
        "paper_001",
        query_embedding,
        5
    ]


def test_search_with_metadata_filter(
    store,
    mock_cursor
):
    query_embedding = [
        0.1,
        0.2,
        0.3
    ]

    mock_cursor.fetchall.return_value = []

    store.search(
        query_embedding=query_embedding,
        top_k=10,
        filters={
            "page": 5
        }
    )

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "metadata->>%s = %s" in query

    assert list(params) == [
        query_embedding,
        "page",
        "5",
        query_embedding,
        10
    ]


def test_search_with_multiple_filters(
    store,
    mock_cursor
):
    query_embedding = [
        0.1,
        0.2,
        0.3
    ]

    mock_cursor.fetchall.return_value = []

    store.search(
        query_embedding=query_embedding,
        top_k=5,
        filters={
            "document_id": "paper_001",
            "page": 5,
            "section": "Method"
        }
    )

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "WHERE document_id = %s" in query

    assert query.count(
        "metadata->>%s = %s"
    ) == 2

    assert list(params) == [
        query_embedding,
        "paper_001",
        "page",
        "5",
        "section",
        "Method",
        query_embedding,
        5
    ]


# ============================================================
# COUNT
# ============================================================

def test_count_without_filter(
    store,
    mock_cursor
):
    """
    filters=None -> count all rows.
    """

    mock_cursor.fetchone.return_value = {
        "total": 100
    }

    result = store.count(
        filters=None
    )

    assert result == 100

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "SELECT COUNT(*) AS total" in query
    assert "FROM chunks_db" in query
    assert "WHERE" not in query

    assert list(params) == []


def test_count_empty_filter_raises(
    store
):
    with pytest.raises(
        ValueError,
        match="Filters cannot be empty"
    ):
        store.count(
            filters={}
        )


def test_count_with_document_id_filter(
    store,
    mock_cursor
):
    mock_cursor.fetchone.return_value = {
        "total": 5
    }

    result = store.count(
        filters={
            "document_id": "paper_001"
        }
    )

    assert result == 5

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "SELECT COUNT(*) AS total" in query
    assert "WHERE document_id = %s" in query

    assert list(params) == [
        "paper_001"
    ]


def test_count_with_multiple_filters(
    store,
    mock_cursor
):
    mock_cursor.fetchone.return_value = {
        "total": 2
    }

    result = store.count(
        filters={
            "document_id": "paper_001",
            "page": 5,
            "section": "Method"
        }
    )

    assert result == 2

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "WHERE document_id = %s" in query

    assert query.count(
        "metadata->>%s = %s"
    ) == 2

    assert list(params) == [
        "paper_001",
        "page",
        "5",
        "section",
        "Method"
    ]


# ============================================================
# DELETE
# ============================================================

def test_delete_empty_filter_raises(
    store
):
    """
    {} is invalid.
    """

    with pytest.raises(
        ValueError,
        match="Filters cannot be empty"
    ):
        store.delete(
            filters={}
        )


def test_delete_with_document_id_filter(
    store,
    mock_cursor,
    mock_conn
):
    store.delete(
        filters={
            "document_id": "paper_001"
        }
    )

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "DELETE FROM chunks_db" in query
    assert "WHERE document_id = %s" in query

    assert list(params) == [
        "paper_001"
    ]

    mock_conn.commit.assert_called_once()


def test_delete_with_multiple_filters(
    store,
    mock_cursor,
    mock_conn
):
    store.delete(
        filters={
            "document_id": "paper_001",
            "page": 5
        }
    )

    query, params = (
        mock_cursor.execute.call_args.args
    )

    query = normalize_sql(query)

    assert "DELETE FROM chunks_db" in query
    assert "WHERE document_id = %s" in query
    assert "metadata->>%s = %s" in query

    assert list(params) == [
        "paper_001",
        "page",
        "5"
    ]

    mock_conn.commit.assert_called_once()


# ============================================================
# CLOSE
# ============================================================

def test_close(
    store,
    mock_conn
):
    store.close()

    mock_conn.close.assert_called_once()


# ============================================================
# ERROR HANDLING
# ============================================================

def test_search_database_error(
    store,
    mock_cursor,
    mock_conn
):
    mock_cursor.execute.side_effect = (
        psycopg.DatabaseError(
            "Test database error"
        )
    )

    with pytest.raises(
        psycopg.DatabaseError
    ):
        store.search(
            query_embedding=[
                0.1,
                0.2,
                0.3
            ]
        )

    mock_conn.rollback.assert_called_once()


def test_count_database_error(
    store,
    mock_cursor,
    mock_conn
):
    mock_cursor.execute.side_effect = (
        psycopg.DatabaseError(
            "Test database error"
        )
    )

    with pytest.raises(
        psycopg.DatabaseError
    ):
        store.count()

    mock_conn.rollback.assert_called_once()


def test_delete_database_error(
    store,
    mock_cursor,
    mock_conn
):
    mock_cursor.execute.side_effect = (
        psycopg.DatabaseError(
            "Test database error"
        )
    )

    with pytest.raises(
        psycopg.DatabaseError
    ):
        store.delete(
            filters={
                "document_id": "paper_001"
            }
        )

    mock_conn.rollback.assert_called_once()


# ============================================================
# GET CONNECTION
# ============================================================

def test_get_connection(mocker):
    """
    Test PG connection creation without actually
    connecting to PostgreSQL.
    """

    fake_database_url = (
        "postgresql://user:password@localhost:5432/test"
    )

    mock_connection = MagicMock()

    # _get_connection checks this global variable first
    mocker.patch(
        "src.database.pgvector_storage.DATABASE_URL",
        fake_database_url
    )

    mock_connect = mocker.patch(
        "src.database.pgvector_storage.psycopg.connect",
        return_value=mock_connection
    )

    mock_register = mocker.patch(
        "src.database.pgvector_storage.register_vector"
    )

    store = PGVectorStore.__new__(
        PGVectorStore
    )

    result = store._get_connection()

    assert result == mock_connection

    mock_connect.assert_called_once()

    mock_register.assert_called_once_with(
        mock_connection
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        pytest.main([
            __file__,
            "-v"
        ])
    )