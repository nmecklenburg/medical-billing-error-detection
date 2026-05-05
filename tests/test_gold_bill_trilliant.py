from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import duckdb

from claim.cli_logging import CliLogger
from claim.gold_bill_trilliant import (
    TRILLIANT_ALIAS,
    TrilliantSchema,
    build_component_imputed_rate_quote,
    build_hospital_local_estimate_quote,
    build_hospital_local_professional_support,
    build_imputed_rate_quote,
    connect_trilliant_cache,
    discover_public_hospital_rate_schema,
    discover_trilliant_schema,
    ensure_cache_tables,
    filter_public_directory_candidates,
    build_hospital_geography_key,
    get_hospital_candidates,
    open_public_directory_hospital_database,
    professional_billing_code_family,
    query_selected_hospital_exact_professional_rows,
    query_public_directory_component_rate_quotes,
    query_rate_level,
    query_trilliant_prior_quote,
    quote_supports_scenario,
    resolve_trilliant_prior_quote,
    resolve_trilliant_prior_quotes,
    sanitize_rate_quote,
    select_financial_scenario,
    select_public_directory_hospital_with_rate_coverage,
)


class GoldBillTrilliantTests(unittest.TestCase):
    def test_connect_trilliant_cache_logs_attach_fallback_warning(self) -> None:
        class FakeConnection:
            def __init__(self) -> None:
                self.executed: list[str] = []

            def execute(self, sql: str, _params: object = None) -> "FakeConnection":
                self.executed.append(sql)
                if sql.startswith("ATTACH "):
                    raise RuntimeError("attach failed")
                return self

        fake_connection = FakeConnection()

        with TemporaryDirectory() as tmpdir, mock.patch(
            "claim.gold_bill_trilliant.duckdb",
            SimpleNamespace(connect=lambda _path: fake_connection),
        ):
            with self.assertLogs("claim.gold_bill_trilliant", level="WARNING") as logs:
                returned = connect_trilliant_cache(
                    Path(tmpdir) / "cache.duckdb",
                    "https://example.com/trilliant.duckdb",
                )

        self.assertIs(returned, fake_connection)
        self.assertTrue(
            any(
                "falling back to public-directory schema discovery" in message
                for message in logs.output
            )
        )

    def test_discover_trilliant_schema_logs_public_directory_fallback(self) -> None:
        class FailingConnection:
            def execute(self, _sql: str, _params: object = None) -> "FailingConnection":
                raise RuntimeError("show tables failed")

        with self.assertLogs("claim.gold_bill_trilliant", level="WARNING") as logs:
            schema = discover_trilliant_schema(
                FailingConnection(),
                hospital_table_override=None,
                rate_table_override=None,
            )

        self.assertEqual(schema.source_mode, "public_directory")
        self.assertTrue(
            any(
                "falling back to public-directory schema discovery" in message
                for message in logs.output
            )
        )

    def test_ensure_cache_tables_initializes_expected_tables(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            tables = {
                row[0]
                for row in connection.execute("SHOW TABLES").fetchall()
            }
        finally:
            connection.close()

        self.assertTrue(
            {
                "schema_cache",
                "hospital_candidate_cache",
                "rate_quote_cache",
                "artifact_cache",
            }.issubset(tables)
        )

    def test_query_selected_hospital_exact_professional_rows_returns_charge_side_row_without_allowed_amount(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute(f"ATTACH ':memory:' AS {TRILLIANT_ALIAS}")
            connection.execute(
                f"""
                CREATE TABLE {TRILLIANT_ALIAS}.rates (
                    billing_code VARCHAR,
                    negotiated_rate DOUBLE,
                    gross_charge DOUBLE,
                    standard_charge_dollar DOUBLE,
                    minimum DOUBLE,
                    maximum DOUBLE,
                    hospital_id VARCHAR
                )
                """
            )
            connection.execute(
                f"""
                INSERT INTO {TRILLIANT_ALIAS}.rates
                VALUES ('99283', NULL, 800.0, 400.0, NULL, NULL, 'hospital-1')
                """
            )

            quotes = query_selected_hospital_exact_professional_rows(
                connection,
                TrilliantSchema(
                    hospital_table="hospitals",
                    rate_table="rates",
                    hospital_name_column=None,
                    hospital_npi_column=None,
                    hospital_zip_column=None,
                    hospital_state_column=None,
                    hospital_county_name_column=None,
                    hospital_county_fips_column=None,
                    hospital_address_columns=[],
                    hospital_identity_columns=[],
                    rate_billing_code_column="billing_code",
                    rate_payer_column=None,
                    rate_negotiated_rate_column="negotiated_rate",
                    rate_gross_charge_column="gross_charge",
                    rate_state_column=None,
                    rate_identity_columns=["hospital_id"],
                    source_mode="aggregate",
                    rate_standard_dollar_column="standard_charge_dollar",
                    rate_minimum_column="minimum",
                    rate_maximum_column="maximum",
                ),
                billing_codes=["99283"],
                encounter_regime="outpatient",
                identity_filter=("hospital_id", "hospital-1"),
                max_quote_amount=1_000_000.0,
            )
        finally:
            connection.close()

        self.assertEqual(quotes["99283"]["negotiated_rate"], None)
        self.assertEqual(quotes["99283"]["standard_charge_dollar"], 400.0)
        self.assertEqual(quotes["99283"]["gross_charge"], 800.0)

    def test_build_hospital_local_estimate_quote_uses_charge_anchor_precedence(self) -> None:
        support_data = {"ratios_by_family": {}, "hospital_ratios": [0.5]}
        cases = [
            ("standard_charge_dollar", {"standard_charge_dollar": 400.0}, 400.0),
            ("percentage_charge_amount", {"percentage_charge_amount": 300.0}, 300.0),
            ("minimum", {"minimum": 200.0}, 200.0),
            ("maximum", {"maximum": 180.0}, 180.0),
            ("gross_charge", {"gross_charge": 160.0}, 160.0),
        ]

        for expected_field, fields, expected_anchor in cases:
            with self.subTest(expected_field=expected_field):
                quote = build_hospital_local_estimate_quote(
                    {
                        "billing_code": "99283",
                        **fields,
                    },
                    support_data,
                    scenario_name="in_network",
                    max_quote_amount=1_000_000.0,
                )
                assert quote is not None
                self.assertEqual(quote["estimate_charge_field"], expected_field)
                self.assertEqual(quote["gross_charge"], expected_anchor)
                self.assertEqual(quote["negotiated_rate"], expected_anchor * 0.5)

    def test_build_hospital_local_estimate_quote_applies_guardrails(self) -> None:
        capped_quote = build_hospital_local_estimate_quote(
            {
                "billing_code": "99283",
                "gross_charge": 100.0,
            },
            {"ratios_by_family": {}, "hospital_ratios": [1.4]},
            scenario_name="in_network",
            max_quote_amount=1_000_000.0,
        )
        self.assertIsNotNone(capped_quote)
        self.assertEqual(capped_quote["negotiated_rate"], 100.0)

        self.assertIsNone(
            build_hospital_local_estimate_quote(
                {"billing_code": "99283"},
                {"ratios_by_family": {}, "hospital_ratios": [0.5]},
                scenario_name="in_network",
                max_quote_amount=1_000_000.0,
            )
        )
        self.assertIsNone(
            build_hospital_local_estimate_quote(
                {"billing_code": "99283", "gross_charge": 100.0},
                {"ratios_by_family": {}, "hospital_ratios": []},
                scenario_name="in_network",
                max_quote_amount=1_000_000.0,
            )
        )

    def test_build_hospital_local_professional_support_groups_by_family(self) -> None:
        support = build_hospital_local_professional_support(
            [
                {
                    "billing_code": "99213",
                    "negotiated_rate": 100.0,
                    "standard_charge_dollar": 200.0,
                },
                {
                    "billing_code": "80053",
                    "negotiated_rate": 40.0,
                    "gross_charge": 100.0,
                },
            ]
        )

        self.assertEqual(professional_billing_code_family("99283"), "992")
        self.assertEqual(professional_billing_code_family("J1100"), "J11")
        self.assertEqual(support["ratios_by_family"]["992"], [0.5])
        self.assertEqual(support["hospital_ratios"], [0.5, 0.4])

    def test_filter_public_directory_candidates_prioritizes_zip_matches_but_keeps_in_state_candidates(
        self,
    ) -> None:
        rows = [
            {
                "id": "hospital-b",
                "hospitalName": "Bravo Regional",
                "address": "2 Main St, Somewhere, AL 36004",
                "city": "Somewhere",
                "state": "AL",
            },
            {
                "id": "hospital-a",
                "hospitalName": "Alpha Medical Center",
                "address": "1 Main St, Somewhere, AL 36003",
                "city": "Somewhere",
                "state": "AL",
            },
            {
                "id": "hospital-c",
                "hospitalName": "Charlie General",
                "address": "3 Main St, Elsewhere, GA 30000",
                "city": "Elsewhere",
                "state": "GA",
            },
        ]

        candidates, query_mode = filter_public_directory_candidates(
            rows,
            zip_codes=["36003"],
            state_value="AL",
            max_candidates=10,
        )

        self.assertEqual(query_mode, "directory_zip")
        self.assertEqual(
            [candidate["hospital_directory_id"] for candidate in candidates],
            ["hospital-a", "hospital-b"],
        )

    def test_build_hospital_geography_key_includes_cache_version(self) -> None:
        key = build_hospital_geography_key(
            ["36003"],
            "01001",
            "Autauga",
            "AL",
            source_mode="public_directory",
        )

        self.assertIn('"cache_version": 2', key)

    def test_open_public_directory_hospital_database_refreshes_stale_cached_url(self) -> None:
        hospital = {
            "hospital_directory_id": "hospital-a",
            "hospital_page_url": "https://example.com/hospital-a",
        }

        class DummyRemote:
            def close(self) -> None:
                return None

        fetch_calls: list[bool] = []
        open_calls: list[str] = []

        def fetch_url(*_args: object, force_refresh: bool = False, **_kwargs: object) -> str:
            fetch_calls.append(force_refresh)
            return "fresh-url" if force_refresh else "stale-url"

        def open_remote(db_url: str) -> DummyRemote:
            open_calls.append(db_url)
            if db_url == "stale-url":
                raise RuntimeError("stale cached url")
            return DummyRemote()

        with (
            mock.patch(
                "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                side_effect=fetch_url,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.open_remote_trilliant_database",
                side_effect=open_remote,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                return_value=SimpleNamespace(rate_table="standard_charge_details"),
            ),
        ):
            remote_connection, remote_schema, duckdb_url = open_public_directory_hospital_database(
                object(),
                object(),
                hospital,
            )

        self.assertIsInstance(remote_connection, DummyRemote)
        self.assertEqual(remote_schema.rate_table, "standard_charge_details")
        self.assertEqual(duckdb_url, "fresh-url")
        self.assertEqual(open_calls, ["stale-url", "fresh-url"])
        self.assertEqual(fetch_calls, [False, True])
        self.assertEqual(hospital["duckdb_url"], "fresh-url")

    def test_get_hospital_candidates_passes_zip_codes_to_public_directory_filter(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            rows = [{"id": "hospital-a", "hospitalName": "Alpha Medical Center"}]
            with (
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.filter_public_directory_candidates",
                    return_value=([{"hospital_directory_id": "hospital-a", "hospital_name": "Alpha Medical Center"}], "directory_zip"),
                ) as mock_filter,
            ):
                candidates, query_mode, from_cache = get_hospital_candidates(
                    connection,
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    zip_codes=["36003"],
                    county_fips="01001",
                    county_name="Autauga",
                    state_value="AL",
                    max_candidates=25,
                )
        finally:
            connection.close()

        self.assertEqual(query_mode, "directory_zip")
        self.assertFalse(from_cache)
        self.assertEqual(candidates[0]["hospital_directory_id"], "hospital-a")
        mock_filter.assert_called_once_with(
            rows,
            zip_codes=["36003"],
            state_value="AL",
            max_candidates=25,
        )

    def test_query_rate_level_prefers_public_directory_median_amount_over_charge_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        estimated_amount DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Emergency department visit', 650.0, '99283', NULL, 'Aetna', 999.0, NULL, 425.0)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_rate_level(
                    cache,
                    schema,
                    billing_codes=["99283"],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    identity_filter=None,
                    state_value=None,
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertEqual(quotes["99283"]["negotiated_rate"], 425.0)
        self.assertEqual(quotes["99283"]["gross_charge"], 650.0)

    def test_discover_trilliant_schema_finds_hospital_and_rate_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "remote.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals_table (
                        hospital_name TEXT,
                        npi TEXT,
                        zip_code TEXT,
                        state TEXT,
                        county_fips TEXT,
                        address TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE rates_table (
                        billing_code TEXT,
                        payer_name TEXT,
                        negotiated_rate DOUBLE,
                        gross_charge DOUBLE,
                        npi TEXT,
                        state TEXT
                    )
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_trilliant_schema(
                    cache,
                    hospital_table_override=None,
                    rate_table_override=None,
                )
            finally:
                cache.close()

        self.assertEqual(schema.hospital_table, "hospitals_table")
        self.assertEqual(schema.rate_table, "rates_table")
        self.assertEqual(schema.hospital_name_column, "hospital_name")
        self.assertEqual(schema.rate_billing_code_column, "billing_code")

    def test_select_financial_scenario_avoids_medicare_advantage_for_under_65_default_cases(self) -> None:
        payer_names = {
            select_financial_scenario(
                "case-under-65",
                seed,
                0.0,
                ["Medicare Advantage", "Aetna", "UnitedHealthcare"],
                age_at_encounter=61,
            )["payer_name"]
            for seed in range(12)
        }

        self.assertNotIn("Medicare Advantage", payer_names)
        self.assertTrue(payer_names.issubset({"Aetna", "UnitedHealthcare"}))

    def test_query_rate_level_supports_public_directory_standard_charge_details(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        standard_charge_percentage DOUBLE,
                        estimated_amount DOUBLE,
                        minimum DOUBLE,
                        maximum DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Emergency department visit', 650.0, '99283', NULL, 'Aetna', NULL, NULL, 400.0, NULL, NULL, NULL),
                    (2, 1, 'Emergency department visit', 650.0, '99283', NULL, 'Other Payer', NULL, NULL, 420.0, NULL, NULL, NULL)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_rate_level(
                    cache,
                    schema,
                    billing_codes=["99283"],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    identity_filter=None,
                    state_value=None,
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertEqual(
            quotes["99283"],
            {
                "billing_code": "99283",
                "negotiated_rate": 400.0,
                "gross_charge": 650.0,
                "standard_charge_dollar": None,
                "percentage_charge_amount": None,
                "minimum": None,
                "maximum": None,
                "description": "Emergency department visit",
                "row_count": 1,
                "lookup_level": "hospital_payer",
                "payer_name": "Aetna",
                "identity_filter": None,
                "state_value": None,
            },
        )

    def test_query_rate_level_filters_implausible_public_directory_amounts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        standard_charge_percentage DOUBLE,
                        estimated_amount DOUBLE,
                        minimum DOUBLE,
                        maximum DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Emergency department visit', 650.0, '99283', NULL, 'Aetna', NULL, NULL, 400.0, NULL, NULL, NULL),
                    (2, 1, 'Emergency department visit', 999999999.0, '99283', NULL, 'Aetna', NULL, NULL, 999999999.0, NULL, NULL, NULL)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_rate_level(
                    cache,
                    schema,
                    billing_codes=["99283"],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    identity_filter=None,
                    state_value=None,
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertEqual(quotes["99283"]["negotiated_rate"], 400.0)
        self.assertEqual(quotes["99283"]["gross_charge"], 650.0)

    def test_query_rate_level_supports_public_directory_charge_only_quotes_out_of_network(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        standard_charge_percentage DOUBLE,
                        estimated_amount DOUBLE,
                        minimum DOUBLE,
                        maximum DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Emergency department visit', NULL, '99283', NULL, NULL, 500.0, NULL, NULL, NULL, NULL, NULL)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_rate_level(
                    cache,
                    schema,
                    billing_codes=["99283"],
                    lookup_level="hospital_average",
                    payer_name=None,
                    identity_filter=None,
                    state_value=None,
                    scenario_name="out_of_network",
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertIsNone(quotes["99283"]["negotiated_rate"])
        self.assertEqual(quotes["99283"]["gross_charge"], 500.0)

    def test_query_rate_level_prefers_real_gross_charge_for_out_of_network_public_directory_quotes(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        standard_charge_percentage DOUBLE,
                        estimated_amount DOUBLE,
                        minimum DOUBLE,
                        maximum DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Emergency department visit', 1200.0, '99283', NULL, NULL, 500.0, NULL, NULL, NULL, NULL, NULL)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_rate_level(
                    cache,
                    schema,
                    billing_codes=["99283"],
                    lookup_level="hospital_average",
                    payer_name=None,
                    identity_filter=None,
                    state_value=None,
                    scenario_name="out_of_network",
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertIsNone(quotes["99283"]["negotiated_rate"])
        self.assertEqual(quotes["99283"]["gross_charge"], 1200.0)

    def test_sanitize_rate_quote_supports_scenario_specific_fallback(self) -> None:
        quote = sanitize_rate_quote(
            {
                "billing_code": "99283",
                "negotiated_rate": 999999999.0,
                "gross_charge": 650.0,
            },
            max_quote_amount=10_000_000.0,
        )

        self.assertIsNone(quote["negotiated_rate"])
        self.assertEqual(quote["gross_charge"], 650.0)
        self.assertFalse(quote_supports_scenario(quote, scenario_name="in_network"))
        self.assertTrue(quote_supports_scenario(quote, scenario_name="out_of_network"))

    def test_query_public_directory_component_rate_quotes_canonicalizes_alias_keys(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        ms_drg TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        standard_charge_percentage DOUBLE,
                        estimated_amount DOUBLE,
                        minimum DOUBLE,
                        maximum DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (
                        1,
                        1,
                        'Major bowel procedure without CC',
                        12000.0,
                        NULL,
                        NULL,
                        '331',
                        'Aetna',
                        NULL,
                        NULL,
                        8000.0,
                        NULL,
                        NULL,
                        NULL
                    )
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_public_directory_component_rate_quotes(
                    cache,
                    schema,
                    occurrences=[{"code": "331", "code_system": "MSDRG"}],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertIn("MS-DRG:331", quotes)
        self.assertNotIn("MSDRG:331", quotes)
        self.assertEqual(quotes["MS-DRG:331"]["negotiated_rate"], 8000.0)

    def test_query_public_directory_component_rate_quotes_resolves_prefixed_ms_drg_code(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute("CREATE TABLE hospitals (hospital_id BIGINT, hospital_name TEXT)")
                remote.execute("INSERT INTO hospitals VALUES (1, 'Alpha Medical Center')")
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        ms_drg TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        estimated_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Major bowel procedure without CC', 12000.0, NULL, NULL, '291', 'Aetna', NULL, 9000.0)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_public_directory_component_rate_quotes(
                    cache,
                    schema,
                    occurrences=[{"code": "MSDRG291", "code_system": "MS-DRG"}],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertEqual(quotes["MS-DRG:291"]["billing_code"], "291")
        self.assertEqual(quotes["MS-DRG:291"]["negotiated_rate"], 9000.0)

    def test_query_public_directory_component_rate_quotes_resolves_prefixed_revenue_code(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute("CREATE TABLE hospitals (hospital_id BIGINT, hospital_name TEXT)")
                remote.execute("INSERT INTO hospitals VALUES (1, 'Alpha Medical Center')")
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        rc TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        estimated_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Pharmacy', 300.0, NULL, NULL, '0250', 'Aetna', NULL, 150.0)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_public_directory_component_rate_quotes(
                    cache,
                    schema,
                    occurrences=[{"code": "REVENUECODE0250", "code_system": "REVENUE_CODE"}],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertEqual(quotes["REVENUE_CODE:0250"]["billing_code"], "0250")
        self.assertEqual(quotes["REVENUE_CODE:0250"]["negotiated_rate"], 150.0)

    def test_query_public_directory_component_rate_quotes_resolves_revenue_code_padding_variants(
        self,
    ) -> None:
        for stored_code, queried_code in (("0100", "100"), ("100", "0100")):
            with self.subTest(stored_code=stored_code, queried_code=queried_code):
                with TemporaryDirectory() as tmpdir:
                    remote_path = Path(tmpdir) / "hospital.duckdb"
                    remote = duckdb.connect(str(remote_path))
                    try:
                        remote.execute(
                            "CREATE TABLE hospitals (hospital_id BIGINT, hospital_name TEXT)"
                        )
                        remote.execute(
                            "INSERT INTO hospitals VALUES (1, 'Alpha Medical Center')"
                        )
                        remote.execute(
                            """
                            CREATE TABLE standard_charge_details (
                                charge_id BIGINT,
                                hospital_id BIGINT,
                                description TEXT,
                                gross_charge DOUBLE,
                                cpt TEXT,
                                hcpcs TEXT,
                                rc TEXT,
                                payer_name TEXT,
                                standard_charge_dollar DOUBLE,
                                estimated_amount DOUBLE
                            )
                            """
                        )
                        remote.execute(
                            """
                            INSERT INTO standard_charge_details VALUES
                            (1, 1, 'Room and board', 800.0, NULL, NULL, ?, 'Aetna', NULL, 400.0)
                            """,
                            [stored_code],
                        )
                    finally:
                        remote.close()

                    cache = duckdb.connect(":memory:")
                    try:
                        ensure_cache_tables(cache)
                        escaped_remote_path = str(remote_path).replace("'", "''")
                        cache.execute(
                            f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                        )
                        schema = discover_public_hospital_rate_schema(cache)
                        quotes = query_public_directory_component_rate_quotes(
                            cache,
                            schema,
                            occurrences=[{"code": queried_code, "code_system": "REVENUE_CODE"}],
                            lookup_level="hospital_payer",
                            payer_name="Aetna",
                            max_quote_amount=10_000_000.0,
                        )
                    finally:
                        cache.close()

                self.assertEqual(quotes["REVENUE_CODE:0100"]["billing_code"], "0100")
                self.assertEqual(quotes["REVENUE_CODE:0100"]["negotiated_rate"], 400.0)

    def test_query_public_directory_component_rate_quotes_supports_charge_only_rows_out_of_network(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute("CREATE TABLE hospitals (hospital_id BIGINT, hospital_name TEXT)")
                remote.execute("INSERT INTO hospitals VALUES (1, 'Alpha Medical Center')")
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        rc TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        estimated_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Room and board', NULL, NULL, NULL, '0100', NULL, 700.0, NULL)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_public_directory_component_rate_quotes(
                    cache,
                    schema,
                    occurrences=[{"code": "0100", "code_system": "REVENUE_CODE"}],
                    lookup_level="hospital_average",
                    payer_name=None,
                    scenario_name="out_of_network",
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertIn("REVENUE_CODE:0100", quotes)
        self.assertIsNone(quotes["REVENUE_CODE:0100"]["negotiated_rate"])
        self.assertEqual(quotes["REVENUE_CODE:0100"]["gross_charge"], 700.0)

    def test_build_component_imputed_rate_quote_uses_anchor_claim_payment_without_multiplier(
        self,
    ) -> None:
        quote = build_component_imputed_rate_quote(
            {
                "code": "331",
                "code_system": "MS-DRG",
                "source_claim_id": "claim-1",
            },
            {
                "claim-1": {
                    "claim_id": "claim-1",
                    "amounts": {"claim_payment_amount": 2500.0},
                }
            },
            encounter={
                "encounter": {
                    "anchor_claim_type": "inpatient",
                    "from_date": "2024-01-01",
                    "thru_date": "2024-01-02",
                }
            },
            source_medicare_multiplier=2.5,
        )

        self.assertEqual(quote["lookup_level"], "inpatient_anchor_imputed")
        self.assertEqual(quote["negotiated_rate"], 2500.0)
        self.assertEqual(quote["source_allowed_amount"], 2500.0)
        self.assertFalse(quote["floor_applied"])

    def test_build_component_imputed_rate_quote_uses_short_stay_floor(self) -> None:
        quote = build_component_imputed_rate_quote(
            {
                "code": "331",
                "code_system": "MS-DRG",
                "source_claim_id": "claim-1",
            },
            {
                "claim-1": {
                    "claim_id": "claim-1",
                    "amounts": {"claim_payment_amount": 500.0},
                }
            },
            encounter={
                "encounter": {
                    "anchor_claim_type": "inpatient",
                    "from_date": "2024-01-01",
                    "thru_date": "2024-01-02",
                }
            },
            source_medicare_multiplier=2.5,
        )

        self.assertEqual(quote["negotiated_rate"], 2000.0)
        self.assertEqual(quote["floor_amount"], 2000.0)
        self.assertTrue(quote["floor_applied"])

    def test_build_component_imputed_rate_quote_uses_long_stay_floor(self) -> None:
        quote = build_component_imputed_rate_quote(
            {
                "code": "331",
                "code_system": "MS-DRG",
                "source_claim_id": "claim-1",
            },
            {
                "claim-1": {
                    "claim_id": "claim-1",
                    "amounts": {"claim_payment_amount": 500.0},
                }
            },
            encounter={
                "encounter": {
                    "anchor_claim_type": "inpatient",
                    "from_date": "2024-01-01",
                    "thru_date": "2024-01-05",
                }
            },
            source_medicare_multiplier=2.5,
        )

        self.assertEqual(quote["negotiated_rate"], 3000.0)
        self.assertEqual(quote["floor_amount"], 3000.0)
        self.assertTrue(quote["floor_applied"])

    def test_build_component_imputed_rate_quote_preserves_generic_non_drg_behavior(self) -> None:
        occurrence = {
            "code": "0360",
            "code_system": "REVENUE_CODE",
            "submitted_charge_amount": 100.0,
        }

        self.assertEqual(
            build_component_imputed_rate_quote(
                occurrence,
                {},
                encounter={"encounter": {"anchor_claim_type": "inpatient"}},
                source_medicare_multiplier=2.5,
            ),
            build_imputed_rate_quote(
                occurrence,
                {},
                source_medicare_multiplier=2.5,
            ),
        )

    def test_build_component_imputed_rate_quote_preserves_generic_drg_behavior_outside_inpatient(
        self,
    ) -> None:
        occurrence = {
            "code": "331",
            "code_system": "MS-DRG",
            "source_claim_id": "claim-1",
        }
        claim_index = {
            "claim-1": {
                "claim_id": "claim-1",
                "amounts": {"claim_payment_amount": 500.0},
            }
        }

        self.assertEqual(
            build_component_imputed_rate_quote(
                occurrence,
                claim_index,
                encounter={"encounter": {"anchor_claim_type": "outpatient"}},
                source_medicare_multiplier=2.5,
            ),
            build_imputed_rate_quote(
                occurrence,
                claim_index,
                source_medicare_multiplier=2.5,
            ),
        )

    def test_select_public_directory_hospital_with_rate_coverage_prefers_anchor_quotes(self) -> None:
        candidates = [
            {"hospital_directory_id": "svc-only", "hospital_name": "Service Only"},
            {"hospital_directory_id": "drg-covered", "hospital_name": "DRG Covered"},
        ]

        def service_quotes(*_args: object, **kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
            hospital = kwargs["hospital"]
            if hospital["hospital_directory_id"] == "svc-only":
                return ({"99213": {"billing_code": "99213", "negotiated_rate": 100.0}}, {"cache_hits": 0, "cache_misses": 1})
            return ({"99213": {"billing_code": "99213", "negotiated_rate": 100.0}}, {"cache_hits": 0, "cache_misses": 1})

        def component_quotes(*_args: object, **kwargs: object) -> dict[str, dict[str, object]]:
            hospital = kwargs["hospital"]
            if hospital["hospital_directory_id"] == "svc-only":
                return {}
            return {"MS-DRG:331": {"billing_code": "331", "negotiated_rate": 8000.0}}

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                side_effect=service_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                side_effect=component_quotes,
            ),
        ):
            hospital, service_quote_by_code, component_quote_by_code, telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-1",
                    seed=0,
                    candidates=candidates,
                    billing_codes=["99213"],
                    component_occurrences=[{"code": "331", "code_system": "MS-DRG"}],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    source_state_value="AL",
                    max_hospital_rate_probes=2,
                    max_quote_amount=10000000.0,
                )
            )

        self.assertEqual(hospital["hospital_directory_id"], "drg-covered")
        self.assertEqual(hospital["drg_quote_count"], 1)
        self.assertEqual(hospital["other_anchor_quote_count"], 0)
        self.assertEqual(hospital["service_quote_count"], 1)
        self.assertEqual(hospital["total_quote_count"], 2)
        self.assertEqual(hospital["selection_score"], [1, 0, 1, 2])
        self.assertEqual(service_quote_by_code["99213"]["negotiated_rate"], 100.0)
        self.assertEqual(component_quote_by_code["MS-DRG:331"]["negotiated_rate"], 8000.0)
        self.assertEqual(telemetry, {"cache_hits": 0, "cache_misses": 1})

    def test_public_directory_selection_expands_from_zip_to_county_when_zip_best_has_zero_coverage(
        self,
    ) -> None:
        candidates = [
            {
                "hospital_directory_id": "zip-local",
                "hospital_name": "Zip Local Hospital",
                "hospital_zip": "36003",
                "hospital_state": "AL",
            },
            {
                "hospital_directory_id": "county-fallback",
                "hospital_name": "County General",
                "hospital_zip": "36004",
                "hospital_state": "AL",
            },
        ]
        probed: list[str] = []

        def service_quotes(*_args: object, **kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
            hospital = kwargs["hospital"]
            probed.append(hospital["hospital_directory_id"])
            if hospital["hospital_directory_id"] == "county-fallback":
                return (
                    {"99213": {"billing_code": "99213", "negotiated_rate": 100.0}},
                    {"cache_hits": 0, "cache_misses": 1},
                )
            return ({}, {"cache_hits": 0, "cache_misses": 1})

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                side_effect=service_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                return_value={},
            ),
        ):
            hospital, _service_quote_by_code, _component_quote_by_code, _telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-county-fallback",
                    seed=0,
                    candidates=candidates,
                    billing_codes=["99213"],
                    component_occurrences=[],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    encounter_regime="outpatient",
                    zip_codes=["36003"],
                    county_fips="01001",
                    local_hud_mapping={"01001": ["36003", "36004"]},
                    primary_clinical_direction="Routine outpatient follow-up",
                    source_state_value="AL",
                    max_hospital_rate_probes=2,
                    max_quote_amount=10_000_000.0,
                )
            )

        self.assertEqual(probed, ["zip-local", "county-fallback"])
        self.assertEqual(hospital["hospital_directory_id"], "county-fallback")
        self.assertEqual(hospital["geographic_stage"], "county")

    def test_public_directory_selection_continues_after_partial_zip_coverage(self) -> None:
        candidates = [
            {
                "hospital_directory_id": "zip-local",
                "hospital_name": "Zip Local Hospital",
                "hospital_zip": "36003",
                "hospital_state": "AL",
            },
            {
                "hospital_directory_id": "county-complete",
                "hospital_name": "County General",
                "hospital_zip": "36004",
                "hospital_state": "AL",
            },
        ]
        probed: list[str] = []

        def service_quotes(*_args: object, **kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
            hospital = kwargs["hospital"]
            probed.append(hospital["hospital_directory_id"])
            if hospital["hospital_directory_id"] == "zip-local":
                return (
                    {"99213": {"billing_code": "99213", "negotiated_rate": 100.0}},
                    {"cache_hits": 0, "cache_misses": 1},
                )
            return (
                {
                    "99213": {"billing_code": "99213", "negotiated_rate": 100.0},
                    "80053": {"billing_code": "80053", "negotiated_rate": 55.0},
                },
                {"cache_hits": 0, "cache_misses": 1},
            )

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                side_effect=service_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                return_value={},
            ),
        ):
            hospital, service_quote_by_code, _component_quote_by_code, _telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-partial-zip-fallback",
                    seed=0,
                    candidates=candidates,
                    billing_codes=["99213", "80053"],
                    component_occurrences=[],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    encounter_regime="outpatient",
                    zip_codes=["36003"],
                    county_fips="01001",
                    local_hud_mapping={"01001": ["36003", "36004"]},
                    primary_clinical_direction="Routine outpatient follow-up",
                    source_state_value="AL",
                    max_hospital_rate_probes=1,
                    max_quote_amount=10_000_000.0,
                )
            )

        self.assertEqual(probed, ["zip-local", "county-complete"])
        self.assertEqual(hospital["hospital_directory_id"], "county-complete")
        self.assertEqual(hospital["geographic_stage"], "county")
        self.assertEqual(sorted(service_quote_by_code), ["80053", "99213"])

    def test_public_directory_selection_retries_specialty_mismatched_zip_hospital_when_local_pool_would_empty(
        self,
    ) -> None:
        candidates = [
            {
                "hospital_directory_id": "psych-zip",
                "hospital_name": "Psychiatric Specialty Hospital",
                "hospital_zip": "36003",
                "hospital_state": "AL",
            },
            {
                "hospital_directory_id": "general-state",
                "hospital_name": "General Hospital",
                "hospital_zip": "36004",
                "hospital_state": "AL",
            },
        ]
        probed: list[str] = []

        def service_quotes(*_args: object, **kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
            hospital = kwargs["hospital"]
            probed.append(hospital["hospital_directory_id"])
            return (
                {"99213": {"billing_code": "99213", "negotiated_rate": 100.0}},
                {"cache_hits": 0, "cache_misses": 1},
            )

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                side_effect=service_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                return_value={},
            ),
        ):
            hospital, _service_quote_by_code, _component_quote_by_code, _telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-specialty-skip",
                    seed=0,
                    candidates=candidates,
                    billing_codes=["99213"],
                    component_occurrences=[],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    encounter_regime="outpatient",
                    zip_codes=["36003"],
                    county_fips=None,
                    local_hud_mapping=None,
                    primary_clinical_direction="Routine outpatient follow-up",
                    source_state_value="AL",
                    max_hospital_rate_probes=2,
                    max_quote_amount=10_000_000.0,
                )
            )

        self.assertEqual(probed, ["psych-zip"])
        self.assertEqual(hospital["hospital_directory_id"], "psych-zip")
        self.assertEqual(hospital["geographic_stage"], "zip")
        self.assertFalse(hospital["specialty_filter_active"])

    def test_public_directory_selection_uses_same_state_unfiltered_fallback_when_needed(
        self,
    ) -> None:
        candidates = [
            {
                "hospital_directory_id": "psych-only",
                "hospital_name": "Psychiatric Specialty Hospital",
                "hospital_zip": "36003",
                "hospital_state": "AL",
            }
        ]

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                return_value=(
                    {"99213": {"billing_code": "99213", "negotiated_rate": 100.0}},
                    {"cache_hits": 0, "cache_misses": 1},
                ),
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                return_value={},
            ),
        ):
            hospital, _service_quote_by_code, _component_quote_by_code, _telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-unfiltered-fallback",
                    seed=0,
                    candidates=candidates,
                    billing_codes=["99213"],
                    component_occurrences=[],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    encounter_regime="outpatient",
                    zip_codes=[],
                    county_fips=None,
                    local_hud_mapping=None,
                    primary_clinical_direction="Routine outpatient follow-up",
                    source_state_value="AL",
                    max_hospital_rate_probes=1,
                    max_quote_amount=10_000_000.0,
                )
            )

        self.assertEqual(hospital["hospital_directory_id"], "psych-only")
        self.assertFalse(hospital["specialty_filter_active"])
        self.assertTrue(hospital["same_state_unfiltered_fallback_invoked"])

    def test_public_directory_selection_retries_zip_pool_without_specialty_filter_when_needed(
        self,
    ) -> None:
        candidates = [
            {
                "hospital_directory_id": "psych-zip",
                "hospital_name": "Psychiatric Specialty Hospital",
                "hospital_zip": "36003",
                "hospital_state": "AL",
            }
        ]

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                return_value=(
                    {"99213": {"billing_code": "99213", "negotiated_rate": 100.0}},
                    {"cache_hits": 0, "cache_misses": 1},
                ),
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                return_value={},
            ),
        ):
            hospital, _service_quote_by_code, _component_quote_by_code, _telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-zip-unfiltered-fallback",
                    seed=0,
                    candidates=candidates,
                    billing_codes=["99213"],
                    component_occurrences=[],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    encounter_regime="outpatient",
                    zip_codes=["36003"],
                    county_fips=None,
                    local_hud_mapping=None,
                    primary_clinical_direction="Routine outpatient follow-up",
                    source_state_value="AL",
                    max_hospital_rate_probes=1,
                    max_quote_amount=10_000_000.0,
                )
            )

        self.assertEqual(hospital["hospital_directory_id"], "psych-zip")
        self.assertEqual(hospital["geographic_stage"], "zip")
        self.assertFalse(hospital["specialty_filter_active"])
        self.assertFalse(hospital["same_state_unfiltered_fallback_invoked"])

    def test_inpatient_public_directory_selection_expands_probes_only_after_zero_anchor_quotes(
        self,
    ) -> None:
        candidates = [
            {"hospital_directory_id": "first", "hospital_name": "First"},
            {"hospital_directory_id": "second", "hospital_name": "Second"},
            {"hospital_directory_id": "third", "hospital_name": "Third"},
            {"hospital_directory_id": "fourth", "hospital_name": "Fourth"},
        ]
        probed_hospitals: list[str] = []

        def service_quotes(*_args: object, **kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
            hospital = kwargs["hospital"]
            probed_hospitals.append(hospital["hospital_directory_id"])
            return ({}, {"cache_hits": 0, "cache_misses": 1})

        def component_quotes(*_args: object, **kwargs: object) -> dict[str, dict[str, object]]:
            hospital = kwargs["hospital"]
            if hospital["hospital_directory_id"] == "third":
                return {"MS-DRG:331": {"billing_code": "331", "negotiated_rate": 8000.0}}
            return {}

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                side_effect=service_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                side_effect=component_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.ordered_hospital_candidates_for_case",
                return_value=candidates,
            ),
        ):
            hospital, _service_quote_by_code, component_quote_by_code, _telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-1",
                    seed=0,
                    candidates=candidates,
                    billing_codes=[],
                    component_occurrences=[{"code": "331", "code_system": "MS-DRG"}],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    source_state_value="AL",
                    max_hospital_rate_probes=2,
                    max_quote_amount=10_000_000.0,
                )
            )

        self.assertEqual(probed_hospitals, ["first", "second", "third"])
        self.assertEqual(hospital["hospital_directory_id"], "third")
        self.assertEqual(hospital["selection_probe_count"], 3)
        self.assertEqual(hospital["selection_probe_limit"], 4)
        self.assertEqual(component_quote_by_code["MS-DRG:331"]["negotiated_rate"], 8000.0)

    def test_inpatient_public_directory_selection_does_not_expand_after_phase_one_anchor_hit(
        self,
    ) -> None:
        candidates = [
            {"hospital_directory_id": "first", "hospital_name": "First"},
            {"hospital_directory_id": "second", "hospital_name": "Second"},
            {"hospital_directory_id": "third", "hospital_name": "Third"},
        ]
        probed_hospitals: list[str] = []

        def service_quotes(*_args: object, **kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
            hospital = kwargs["hospital"]
            probed_hospitals.append(hospital["hospital_directory_id"])
            return ({}, {"cache_hits": 0, "cache_misses": 1})

        def component_quotes(*_args: object, **kwargs: object) -> dict[str, dict[str, object]]:
            hospital = kwargs["hospital"]
            if hospital["hospital_directory_id"] == "second":
                return {"MS-DRG:331": {"billing_code": "331", "negotiated_rate": 8000.0}}
            return {}

        with (
            mock.patch(
                "claim.gold_bill_trilliant.resolve_rate_quotes",
                side_effect=service_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.resolve_component_rate_quotes",
                side_effect=component_quotes,
            ),
            mock.patch(
                "claim.gold_bill_trilliant.ordered_hospital_candidates_for_case",
                return_value=candidates,
            ),
        ):
            hospital, _service_quote_by_code, component_quote_by_code, _telemetry = (
                select_public_directory_hospital_with_rate_coverage(
                    object(),
                    SimpleNamespace(source_mode="public_directory"),
                    session=object(),
                    case_id="case-1",
                    seed=0,
                    candidates=candidates,
                    billing_codes=[],
                    component_occurrences=[{"code": "331", "code_system": "MS-DRG"}],
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    source_state_value="AL",
                    max_hospital_rate_probes=2,
                    max_quote_amount=10_000_000.0,
                )
            )

        self.assertEqual(probed_hospitals, ["first", "second"])
        self.assertEqual(hospital["hospital_directory_id"], "second")
        self.assertEqual(hospital["selection_probe_count"], 2)
        self.assertEqual(hospital["selection_probe_limit"], 2)
        self.assertEqual(component_quote_by_code["MS-DRG:331"]["negotiated_rate"], 8000.0)

    def test_query_trilliant_prior_quote_uses_public_directory_hospital_duckdbs_when_rate_table_missing(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_table="standard_charge_details",
            )
            search_rows = [
                {
                    "id": "hospital-a",
                    "hospitalName": "Alpha",
                    "address": "1 Main St, Somewhere, AL 36003",
                    "state": "AL",
                    "city": "Somewhere",
                },
                {
                    "id": "hospital-b",
                    "hospitalName": "Bravo",
                    "address": "2 Main St, Somewhere, AL 36004",
                    "state": "AL",
                    "city": "Somewhere",
                },
                {
                    "id": "hospital-c",
                    "hospitalName": "Charlie",
                    "address": "3 Main St, Elsewhere, GA 30000",
                    "state": "GA",
                    "city": "Elsewhere",
                },
            ]
            queried_hospitals: list[str] = []

            class DummyRemote:
                def __init__(self, hospital_id: str) -> None:
                    self.hospital_id = hospital_id

                def close(self) -> None:
                    return None

            def hospital_duckdb_url(*args: object, **_kwargs: object) -> str:
                hospital = args[2]
                queried_hospitals.append(hospital["hospital_directory_id"])
                return hospital["hospital_directory_id"]

            def hospital_quotes(remote_connection: DummyRemote, *_args: object, **_kwargs: object) -> dict[str, dict[str, object]]:
                if remote_connection.hospital_id == "hospital-a":
                    return {
                        "99213": {
                            "billing_code": "99213",
                            "negotiated_rate": 80.0,
                            "row_count": 10,
                        }
                    }
                if remote_connection.hospital_id == "hospital-b":
                    return {
                        "99213": {
                            "billing_code": "99213",
                            "negotiated_rate": 100.0,
                            "row_count": 20,
                        }
                    }
                return {}

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    side_effect=hospital_duckdb_url,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    side_effect=lambda hospital_id: DummyRemote(hospital_id),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    side_effect=hospital_quotes,
                ),
            ):
                quote = query_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
        finally:
            connection.close()

        self.assertIsNotNone(quote)
        self.assertEqual(quote["negotiated_rate"], 90.0)
        self.assertEqual(quote["hospital_count"], 2)
        self.assertEqual(quote["supporting_row_count"], 30)
        self.assertEqual(queried_hospitals, ["hospital-a", "hospital-b"])

    def test_resolve_trilliant_prior_quotes_uses_state_scope_for_public_directory_candidates_without_rate_state_column(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_table="standard_charge_details",
                rate_state_column=None,
            )
            candidates = [
                {"hospital_directory_id": "ga-hospital", "hospital_state": "GA"},
                {"hospital_directory_id": "fl-hospital", "hospital_state": "FL"},
            ]
            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                    return_value=(
                        {
                            "99223": {
                                "billing_code": "99223",
                                "lookup_level": "trilliant_prior",
                                "negotiated_rate": 240.0,
                                "hospital_count": 3,
                                "supporting_row_count": 9,
                                "prior_lookup_scope": "state",
                            }
                        },
                        {"99223": True},
                        2,
                    ),
                ) as mock_query,
            ):
                quotes = resolve_trilliant_prior_quotes(
                    connection,
                    schema,
                    billing_codes=["99223"],
                    encounter_regime="inpatient",
                    claim_class="carrier",
                    state_value="GA",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                    public_directory_candidates=candidates,
                    allow_national_fallback=False,
                )
        finally:
            connection.close()

        self.assertEqual(quotes["99223"]["prior_lookup_scope"], "state")
        self.assertEqual(
            mock_query.call_args.kwargs["public_directory_candidates"],
            [{"hospital_directory_id": "ga-hospital", "hospital_state": "GA"}],
        )

    def test_query_trilliant_prior_quote_caps_public_directory_state_scan_at_50_hospitals(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_table="standard_charge_details",
            )
            search_rows = [
                {
                    "id": f"hospital-{index:03d}",
                    "hospitalName": f"Hospital {index:03d}",
                    "address": f"{index} Main St, Somewhere, AL 36003",
                    "state": "AL",
                    "city": "Somewhere",
                }
                for index in range(60)
            ]
            queried_hospitals: list[str] = []

            class DummyRemote:
                def __init__(self, hospital_id: str) -> None:
                    self.hospital_id = hospital_id

                def close(self) -> None:
                    return None

            def hospital_duckdb_url(*args: object, **_kwargs: object) -> str:
                hospital = args[2]
                queried_hospitals.append(hospital["hospital_directory_id"])
                return hospital["hospital_directory_id"]

            def hospital_quotes(
                remote_connection: DummyRemote, *_args: object, **_kwargs: object
            ) -> dict[str, dict[str, object]]:
                hospital_index = int(remote_connection.hospital_id.split("-")[-1])
                return {
                    "99213": {
                        "billing_code": "99213",
                        "negotiated_rate": float(hospital_index + 1),
                        "row_count": 1,
                    }
                }

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    side_effect=hospital_duckdb_url,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    side_effect=lambda hospital_id: DummyRemote(hospital_id),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    side_effect=hospital_quotes,
                ),
            ):
                quote = query_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
        finally:
            connection.close()

        self.assertIsNotNone(quote)
        self.assertEqual(quote["hospital_count"], 50)
        self.assertEqual(quote["supporting_row_count"], 50)
        self.assertEqual(quote["negotiated_rate"], 25.5)
        self.assertEqual(len(queried_hospitals), 50)
        self.assertEqual(queried_hospitals[0], "hospital-000")
        self.assertEqual(queried_hospitals[-1], "hospital-049")

    def test_query_trilliant_prior_quote_caps_public_directory_national_scan_at_100_hospitals(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_table="standard_charge_details",
            )
            search_rows = [
                {
                    "id": f"hospital-{index:03d}",
                    "hospitalName": f"Hospital {index:03d}",
                    "address": f"{index} Main St, Somewhere, {'AL' if index % 2 == 0 else 'GA'} 36003",
                    "state": "AL" if index % 2 == 0 else "GA",
                    "city": "Somewhere",
                }
                for index in range(120)
            ]
            queried_hospitals: list[str] = []

            class DummyRemote:
                def __init__(self, hospital_id: str) -> None:
                    self.hospital_id = hospital_id

                def close(self) -> None:
                    return None

            def hospital_duckdb_url(*args: object, **_kwargs: object) -> str:
                hospital = args[2]
                queried_hospitals.append(hospital["hospital_directory_id"])
                return hospital["hospital_directory_id"]

            def hospital_quotes(
                remote_connection: DummyRemote, *_args: object, **_kwargs: object
            ) -> dict[str, dict[str, object]]:
                hospital_index = int(remote_connection.hospital_id.split("-")[-1])
                return {
                    "99213": {
                        "billing_code": "99213",
                        "negotiated_rate": float(hospital_index + 1),
                        "row_count": 1,
                    }
                }

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    side_effect=hospital_duckdb_url,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    side_effect=lambda hospital_id: DummyRemote(hospital_id),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    side_effect=hospital_quotes,
                ),
            ):
                quote = query_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value=None,
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
        finally:
            connection.close()

        self.assertIsNotNone(quote)
        self.assertEqual(quote["hospital_count"], 100)
        self.assertEqual(quote["supporting_row_count"], 100)
        self.assertEqual(quote["negotiated_rate"], 50.5)
        self.assertEqual(len(queried_hospitals), 100)
        self.assertNotIn("hospital-119", queried_hospitals)

    def test_resolve_trilliant_prior_quote_accepts_state_prior_at_three_hospitals(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column="state",
            )
            query_calls: list[str | None] = []

            def prior_query(*_args: object, **kwargs: object) -> dict[str, object]:
                query_calls.append(kwargs["state_value"])
                return {
                    "billing_code": "99213",
                    "negotiated_rate": 87.0,
                    "gross_charge": None,
                    "row_count": 1,
                    "lookup_level": "trilliant_prior",
                    "payer_name": None,
                    "identity_filter": None,
                    "state_value": kwargs["state_value"],
                    "baseline_amount": 87.0,
                    "hospital_count": 3,
                    "supporting_row_count": 1,
                    "encounter_regime": "outpatient",
                }

            with mock.patch(
                "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                side_effect=lambda *_args, **kwargs: (
                    {"99213": prior_query(**kwargs)},
                    {"99213": True},
                    0,
                ),
            ):
                quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
        finally:
            connection.close()

        self.assertIsNotNone(quote)
        self.assertEqual(quote["state_value"], "AL")
        self.assertEqual(quote["hospital_count"], 3)
        self.assertEqual(query_calls, ["AL"])

    def test_resolve_trilliant_prior_quote_rejects_state_prior_below_three_hospitals(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column="state",
            )
            query_calls: list[str | None] = []

            def prior_query(*_args: object, **kwargs: object) -> dict[str, object] | None:
                query_calls.append(kwargs["state_value"])
                if kwargs["state_value"] == "AL":
                    return {
                        "billing_code": "99213",
                        "negotiated_rate": 87.0,
                        "gross_charge": None,
                        "row_count": 999,
                        "lookup_level": "trilliant_prior",
                        "payer_name": None,
                        "identity_filter": None,
                        "state_value": "AL",
                        "baseline_amount": 87.0,
                        "hospital_count": 2,
                        "supporting_row_count": 999,
                        "encounter_regime": "outpatient",
                    }
                return None

            def batched_prior_query(*_args: object, **kwargs: object) -> tuple[dict[str, object], dict[str, bool], int]:
                quote = prior_query(**kwargs)
                return ({"99213": quote} if quote is not None else {}, {"99213": True}, 0)

            with mock.patch(
                "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                side_effect=batched_prior_query,
            ):
                quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
        finally:
            connection.close()

        self.assertIsNone(quote)
        self.assertEqual(query_calls, ["AL", None])

    def test_resolve_trilliant_prior_quote_accepts_national_prior_at_eight_hospitals(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            query_calls: list[str | None] = []

            def prior_query(*_args: object, **kwargs: object) -> dict[str, object]:
                query_calls.append(kwargs["state_value"])
                return {
                    "billing_code": "99213",
                    "negotiated_rate": 91.0,
                    "gross_charge": None,
                    "row_count": 1,
                    "lookup_level": "trilliant_prior",
                    "payer_name": None,
                    "identity_filter": None,
                    "state_value": kwargs["state_value"],
                    "baseline_amount": 91.0,
                    "hospital_count": 8,
                    "supporting_row_count": 1,
                    "encounter_regime": "outpatient",
                }

            with mock.patch(
                "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                side_effect=lambda *_args, **kwargs: (
                    {"99213": prior_query(**kwargs)},
                    {"99213": True},
                    0,
                ),
            ):
                quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
        finally:
            connection.close()

        self.assertIsNotNone(quote)
        self.assertEqual(quote["state_value"], "AL")
        self.assertEqual(quote["hospital_count"], 8)
        self.assertEqual(query_calls, ["AL"])

    def test_resolve_trilliant_prior_quote_rejects_national_prior_below_eight_hospitals(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            query_calls: list[str | None] = []

            def prior_query(*_args: object, **kwargs: object) -> dict[str, object]:
                query_calls.append(kwargs["state_value"])
                return {
                    "billing_code": "99213",
                    "negotiated_rate": 91.0,
                    "gross_charge": None,
                    "row_count": 999,
                    "lookup_level": "trilliant_prior",
                    "payer_name": None,
                    "identity_filter": None,
                    "state_value": kwargs["state_value"],
                    "baseline_amount": 91.0,
                    "hospital_count": 7,
                    "supporting_row_count": 999,
                    "encounter_regime": "outpatient",
                }

            with mock.patch(
                "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                side_effect=lambda *_args, **kwargs: (
                    {"99213": prior_query(**kwargs)},
                    {"99213": True},
                    0,
                ),
            ):
                quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
        finally:
            connection.close()

        self.assertIsNotNone(quote)
        self.assertEqual(quote["state_value"], "AL")
        self.assertEqual(query_calls, ["AL"])

    def test_resolve_trilliant_prior_quote_caches_no_result_miss_for_public_directory(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            search_rows = [
                {
                    "id": "hospital-a",
                    "hospitalName": "Alpha",
                    "address": "1 Main St, Somewhere, AL 36003",
                    "state": "AL",
                    "city": "Somewhere",
                },
                {
                    "id": "hospital-b",
                    "hospitalName": "Bravo",
                    "address": "2 Main St, Somewhere, AL 36004",
                    "state": "AL",
                    "city": "Somewhere",
                },
            ]
            open_calls: list[str] = []

            class DummyRemote:
                def __init__(self, hospital_id: str) -> None:
                    self.hospital_id = hospital_id

                def close(self) -> None:
                    return None

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    side_effect=lambda *_args, **_kwargs: _args[2]["hospital_directory_id"],
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    side_effect=lambda hospital_id: open_calls.append(hospital_id) or DummyRemote(hospital_id),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    return_value={},
                ),
            ):
                first_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
                second_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
        finally:
            connection.close()

        self.assertIsNone(first_quote)
        self.assertIsNone(second_quote)
        self.assertEqual(open_calls, ["hospital-a", "hospital-b"])

    def test_resolve_trilliant_prior_quote_caches_insufficient_support_miss(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            query_calls: list[str | None] = []

            def prior_query(*_args: object, **kwargs: object) -> dict[str, object]:
                query_calls.append(kwargs["state_value"])
                return {
                    "billing_code": "99213",
                    "negotiated_rate": 91.0,
                    "gross_charge": None,
                    "row_count": 7,
                    "lookup_level": "trilliant_prior",
                    "payer_name": None,
                    "identity_filter": None,
                    "state_value": kwargs["state_value"],
                    "baseline_amount": 91.0,
                    "hospital_count": 7,
                    "supporting_row_count": 7,
                    "encounter_regime": "outpatient",
                }

            with mock.patch(
                "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                side_effect=lambda *_args, **kwargs: (
                    {"99213": prior_query(**kwargs)},
                    {"99213": True},
                    0,
                ),
            ):
                first_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
                second_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
        finally:
            connection.close()

        self.assertIsNotNone(first_quote)
        self.assertIsNotNone(second_quote)
        self.assertEqual(first_quote["state_value"], "AL")
        self.assertEqual(second_quote["state_value"], "AL")
        self.assertEqual(query_calls, ["AL"])

    def test_resolve_trilliant_prior_quote_skips_state_rescan_when_state_miss_is_cached(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column="state",
            )
            query_calls: list[str | None] = []

            def prior_query(*_args: object, **kwargs: object) -> dict[str, object]:
                query_calls.append(kwargs["state_value"])
                if kwargs["state_value"] == "AL":
                    return {
                        "billing_code": "99213",
                        "negotiated_rate": 87.0,
                        "gross_charge": None,
                        "row_count": 2,
                        "lookup_level": "trilliant_prior",
                        "payer_name": None,
                        "identity_filter": None,
                        "state_value": "AL",
                        "baseline_amount": 87.0,
                        "hospital_count": 2,
                        "supporting_row_count": 2,
                        "encounter_regime": "outpatient",
                    }
                return {
                    "billing_code": "99213",
                    "negotiated_rate": 91.0,
                    "gross_charge": None,
                    "row_count": 8,
                    "lookup_level": "trilliant_prior",
                    "payer_name": None,
                    "identity_filter": None,
                    "state_value": None,
                    "baseline_amount": 91.0,
                    "hospital_count": 8,
                    "supporting_row_count": 8,
                    "encounter_regime": "outpatient",
                }

            with mock.patch(
                "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                side_effect=lambda *_args, **kwargs: (
                    {"99213": prior_query(**kwargs)},
                    {"99213": True},
                    0,
                ),
            ):
                first_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
                second_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
        finally:
            connection.close()

        self.assertIsNotNone(first_quote)
        self.assertIsNotNone(second_quote)
        self.assertEqual(first_quote["state_value"], None)
        self.assertEqual(second_quote["state_value"], None)
        self.assertEqual(query_calls, ["AL", None])

    def test_resolve_trilliant_prior_quote_does_not_cache_incomplete_public_directory_miss(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            search_rows = [
                {
                    "id": f"hospital-{index:03d}",
                    "hospitalName": f"Hospital {index:03d}",
                    "address": f"{index} Main St, Somewhere, AL 36003",
                    "state": "AL",
                    "city": "Somewhere",
                }
                for index in range(8)
            ]
            open_attempts: list[str] = []

            class DummyRemote:
                def __init__(self, hospital_id: str) -> None:
                    self.hospital_id = hospital_id

                def close(self) -> None:
                    return None

            def open_remote(hospital_id: str) -> DummyRemote:
                open_attempts.append(hospital_id)
                if len(open_attempts) <= 8:
                    raise RuntimeError("transient remote failure")
                return DummyRemote(hospital_id)

            def hospital_quotes(
                remote_connection: DummyRemote, *_args: object, **_kwargs: object
            ) -> dict[str, dict[str, object]]:
                return {
                    "99213": {
                        "billing_code": "99213",
                        "negotiated_rate": 100.0,
                        "row_count": 1,
                    }
                }

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    side_effect=lambda *_args, **_kwargs: _args[2]["hospital_directory_id"],
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    side_effect=open_remote,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    side_effect=hospital_quotes,
                ),
            ):
                first_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
                second_quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
        finally:
            connection.close()

        self.assertIsNone(first_quote)
        self.assertIsNotNone(second_quote)
        self.assertEqual(second_quote["hospital_count"], 8)
        self.assertEqual(len(open_attempts), 16)

    def test_resolve_trilliant_prior_quote_uses_state_lookup_when_public_schema_has_no_state_column(
        self,
    ) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            query_calls: list[str | None] = []

            def prior_query(*_args: object, **kwargs: object) -> dict[str, object]:
                query_calls.append(kwargs["state_value"])
                return {
                    "billing_code": "99213",
                    "negotiated_rate": 87.0,
                    "gross_charge": None,
                    "row_count": 40,
                    "lookup_level": "trilliant_prior",
                    "payer_name": None,
                    "identity_filter": None,
                    "state_value": kwargs["state_value"],
                    "baseline_amount": 87.0,
                    "hospital_count": 10,
                    "supporting_row_count": 40,
                    "encounter_regime": "outpatient",
                }

            with mock.patch(
                "claim.gold_bill_trilliant.query_trilliant_prior_quotes_with_miss_cacheability",
                side_effect=lambda *_args, **kwargs: (
                    {"99213": prior_query(**kwargs)},
                    {"99213": True},
                    0,
                ),
            ):
                quote = resolve_trilliant_prior_quote(
                    connection,
                    schema,
                    billing_code="99213",
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                )
        finally:
            connection.close()

        self.assertIsNotNone(quote)
        self.assertEqual(quote["state_value"], "AL")
        self.assertEqual(query_calls, ["AL"])

    def test_resolve_trilliant_prior_quotes_reuses_public_directory_scans_across_codes(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column="state",
                rate_table="standard_charge_details",
            )
            search_rows = [
                {
                    "id": "hospital-a",
                    "hospitalName": "Alpha",
                    "address": "1 Main St, Somewhere, AL 36003",
                    "state": "AL",
                    "city": "Somewhere",
                },
                {
                    "id": "hospital-b",
                    "hospitalName": "Bravo",
                    "address": "2 Main St, Somewhere, AL 36004",
                    "state": "AL",
                    "city": "Somewhere",
                },
                {
                    "id": "hospital-c",
                    "hospitalName": "Charlie",
                    "address": "3 Main St, Somewhere, AL 36005",
                    "state": "AL",
                    "city": "Somewhere",
                },
                {
                    "id": "hospital-d",
                    "hospitalName": "Delta",
                    "address": "4 Main St, Elsewhere, GA 30001",
                    "state": "GA",
                    "city": "Elsewhere",
                },
                {
                    "id": "hospital-e",
                    "hospitalName": "Echo",
                    "address": "5 Main St, Elsewhere, GA 30002",
                    "state": "GA",
                    "city": "Elsewhere",
                },
                {
                    "id": "hospital-f",
                    "hospitalName": "Foxtrot",
                    "address": "6 Main St, Elsewhere, GA 30003",
                    "state": "GA",
                    "city": "Elsewhere",
                },
                {
                    "id": "hospital-g",
                    "hospitalName": "Golf",
                    "address": "7 Main St, Elsewhere, GA 30004",
                    "state": "GA",
                    "city": "Elsewhere",
                },
                {
                    "id": "hospital-h",
                    "hospitalName": "Hotel",
                    "address": "8 Main St, Elsewhere, GA 30005",
                    "state": "GA",
                    "city": "Elsewhere",
                },
            ]
            scanned_hospitals: list[str] = []
            queried_codes: list[tuple[str, tuple[str, ...]]] = []

            class DummyRemote:
                def __init__(self, hospital_id: str) -> None:
                    self.hospital_id = hospital_id

                def close(self) -> None:
                    return None

            def hospital_quotes(
                remote_connection: DummyRemote, *_args: object, **kwargs: object
            ) -> dict[str, dict[str, object]]:
                billing_codes = tuple(kwargs["billing_codes"])
                queried_codes.append((remote_connection.hospital_id, billing_codes))
                if remote_connection.hospital_id in {"hospital-a", "hospital-b", "hospital-c"}:
                    if billing_codes == ("99214",):
                        return {
                            "99214": {
                                "billing_code": "99214",
                                "negotiated_rate": 110.0,
                                "row_count": 2,
                            }
                        }
                    return {
                        "99213": {
                            "billing_code": "99213",
                            "negotiated_rate": 90.0,
                            "row_count": 2,
                        }
                    }
                return {
                    "99214": {
                        "billing_code": "99214",
                        "negotiated_rate": 110.0,
                        "row_count": 2,
                    }
                }

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    side_effect=lambda *_args, **_kwargs: _args[2]["hospital_directory_id"],
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    side_effect=lambda hospital_id: scanned_hospitals.append(hospital_id) or DummyRemote(hospital_id),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    side_effect=hospital_quotes,
                ),
            ):
                quotes = resolve_trilliant_prior_quotes(
                    connection,
                    schema,
                    billing_codes=["99213", "99214"],
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="AL",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                )
        finally:
            connection.close()

        self.assertEqual(set(quotes), {"99213", "99214"})
        self.assertEqual(quotes["99213"]["prior_lookup_scope"], "state")
        self.assertEqual(quotes["99213"]["hospital_count"], 3)
        self.assertEqual(quotes["99214"]["prior_lookup_scope"], "national")
        self.assertEqual(quotes["99214"]["hospital_count"], 8)
        self.assertEqual(len(scanned_hospitals), 11)
        self.assertTrue(
            all(billing_codes == ("99214",) for _hospital_id, billing_codes in queried_codes[3:])
        )

    def test_resolve_trilliant_prior_quotes_logs_progress_only_for_slow_paths(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            logger_lines: list[str] = []
            logger = CliLogger(
                "test",
                use_color=False,
                line_writer=lambda line, _stream: logger_lines.append(line),
            )
            search_rows = [
                {
                    "id": "hospital-a",
                    "hospitalName": "Alpha",
                    "address": "1 Main St, Somewhere, GA 30001",
                    "state": "GA",
                    "city": "Somewhere",
                }
            ]

            class DummyRemote:
                def close(self) -> None:
                    return None

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    return_value="hospital-a",
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    return_value=DummyRemote(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    return_value={},
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.time.monotonic",
                    side_effect=[0.0, 6.0, 7.0],
                ),
            ):
                quotes = resolve_trilliant_prior_quotes(
                    connection,
                    schema,
                    billing_codes=["99213"],
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="GA",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                    logger=logger,
                    case_id="case-1",
                    scenario_name="in_network",
                )
        finally:
            connection.close()

        self.assertEqual(quotes, {})
        self.assertTrue(any("Starting trilliant prior resolution" in line for line in logger_lines))
        self.assertTrue(any("Trilliant prior scope start" in line for line in logger_lines))
        self.assertTrue(any("Trilliant prior progress" in line for line in logger_lines))
        self.assertTrue(any("Completed trilliant prior resolution" in line for line in logger_lines))
        self.assertTrue(any("misses=1" in line and "hospitals_scanned_state=1" in line for line in logger_lines))

    def test_resolve_trilliant_prior_quotes_skips_progress_log_for_fast_paths(self) -> None:
        connection = duckdb.connect(":memory:")
        try:
            ensure_cache_tables(connection)
            schema = SimpleNamespace(
                source_mode="public_directory",
                rate_state_column=None,
                rate_table="standard_charge_details",
            )
            logger_lines: list[str] = []
            logger = CliLogger(
                "test",
                use_color=False,
                line_writer=lambda line, _stream: logger_lines.append(line),
            )
            search_rows = [
                {
                    "id": "hospital-a",
                    "hospitalName": "Alpha",
                    "address": "1 Main St, Somewhere, GA 30001",
                    "state": "GA",
                    "city": "Somewhere",
                }
            ]

            class DummyRemote:
                def close(self) -> None:
                    return None

            with (
                mock.patch("claim.gold_bill_trilliant.relation_exists", return_value=False),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_trilliant_search_index",
                    return_value=search_rows,
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.fetch_public_hospital_duckdb_url",
                    return_value="hospital-a",
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.open_remote_trilliant_database",
                    return_value=DummyRemote(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.discover_public_hospital_rate_schema",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.query_rate_level_from_public_directory_hospital",
                    return_value={},
                ),
                mock.patch(
                    "claim.gold_bill_trilliant.time.monotonic",
                    side_effect=[0.0, 1.0, 2.0],
                ),
            ):
                resolve_trilliant_prior_quotes(
                    connection,
                    schema,
                    billing_codes=["99213"],
                    encounter_regime="outpatient",
                    claim_class="carrier",
                    state_value="GA",
                    max_quote_amount=10_000_000.0,
                    session=object(),
                    logger=logger,
                    case_id="case-1",
                    scenario_name="in_network",
                )
        finally:
            connection.close()

        self.assertTrue(any("Starting trilliant prior resolution" in line for line in logger_lines))
        self.assertTrue(any("Completed trilliant prior resolution" in line for line in logger_lines))
        self.assertFalse(any("Trilliant prior progress" in line for line in logger_lines))


if __name__ == "__main__":
    unittest.main()
