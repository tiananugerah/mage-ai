import warnings
from typing import IO, Any, Dict, List, Union

import numpy as np
import oracledb
import simplejson
from pandas import DataFrame, Series, read_sql

from mage_ai.io.base import QUERY_ROW_LIMIT, ExportWritePolicy
from mage_ai.io.config import BaseConfigLoader, ConfigKey
from mage_ai.io.constants import UNIQUE_CONFLICT_METHOD_UPDATE
from mage_ai.io.export_utils import PandasTypes
from mage_ai.io.sql import BaseSQL
from mage_ai.server.logger import Logger
from mage_ai.shared.parsers import encode_complex

logger = Logger().new_server_logger(__name__)


class OracleDB(BaseSQL):
    def __init__(self,
                 user,
                 password,
                 host,
                 port,
                 service_name,
                 verbose: bool = False,
                 mode: str = 'thin',
                 **kwargs) -> None:
        super().__init__(user=user,
                         password=password,
                         host=host,
                         port=port,
                         service_name=service_name,
                         verbose=verbose,
                         mode=mode,
                         **kwargs)

    @classmethod
    def with_config(cls, config: BaseConfigLoader) -> 'OracleDB':
        return cls(
            user=config[ConfigKey.ORACLEDB_USER],
            password=config[ConfigKey.ORACLEDB_PASSWORD],
            host=config[ConfigKey.ORACLEDB_HOST],
            port=config[ConfigKey.ORACLEDB_PORT],
            service_name=config[ConfigKey.ORACLEDB_SERVICE],
            mode=config[ConfigKey.ORACLEDB_MODE],
        )

    def open(self) -> None:
        if self.settings['mode'] and self.settings['mode'].lower() == 'thick':
            logger.info('Initializing Oracle thick mode.')
            oracledb.init_oracle_client()
        with self.printer.print_msg(f'Opening connection to OracleDB database \
                                    ({self.settings["mode"]} mode)'):
            connection_dsn = "{}:{}/{}".format(
                self.settings['host'],
                self.settings['port'],
                self.settings['service_name'])
            self._ctx = oracledb.connect(
                user=self.settings['user'], password=self.settings['password'], dsn=connection_dsn)

    def load(
        self,
        query_string: str,
        limit: int = QUERY_ROW_LIMIT,
        display_query: Union[str, None] = None,
        verbose: bool = True,
        **kwargs,
    ) -> DataFrame:
        """
        Loads data from the connected database into a Pandas data frame based on the query given.
        This will fail if the query returns no data from the database. This function will load at
        maximum 10,000,000 rows of data. To operate on more data, consider performing data
        transformations in warehouse.

        Args:
            query_string (str): Query to execute on the database.
            limit (int, Optional): The number of rows to limit the loaded dataframe to. Defaults
                to 10,000,000.
            **kwargs: Additional query parameters.

        Returns:
            DataFrame: The data frame corresponding to the data returned by the given query.
        """
        print_message = 'Loading data'
        if verbose:
            print_message += ' with query'

            if display_query:
                for line in display_query.split('\n'):
                    print_message += f'\n{line}'
            else:
                print_message += f'\n\n{query_string}\n\n'

        query_string = self._clean_query(query_string)

        with self.printer.print_msg(print_message):
            warnings.filterwarnings('ignore', category=UserWarning)

            return read_sql(
                self._enforce_limit_oracledb(query_string, limit),
                self.conn,
                **kwargs,
            )

    def _enforce_limit_oracledb(self, query: str, limit: int = QUERY_ROW_LIMIT) -> str:
        """
        Modifies SQL SELECT query to enforce a limit on the number of rows returned by the query.
        This method is currently supports Oracledb syntax only.

        Args:
            query (str): The SQL query to modify
            limit (int): The limit on the number of rows to return.

        Returns:
            str: Modified query with limit on row count returned.
        """
        query = query.strip(';')

        return f"""
WITH subquery AS (
    {query}
)

SELECT *
FROM subquery
FETCH FIRST {limit} ROWS ONLY
                """

    def table_exists(self, schema_name: str, table_name: str) -> bool:
        with self.conn.cursor() as cur:
            try:
                cur.execute(f"select * from {table_name} where rownum=1")
            except Exception as exc:
                logger.info(f"Table not existing: {table_name}. Exception: {exc}")
                return False
        return True

    def get_type(self, column: Series, dtype: str) -> str:
        if dtype in (
            PandasTypes.MIXED,
            PandasTypes.UNKNOWN_ARRAY,
            PandasTypes.COMPLEX,
        ):
            return 'CHAR(255)'
        elif dtype in (PandasTypes.DATETIME, PandasTypes.DATETIME64):
            try:
                if column.dt.tz:
                    return 'TIMESTAMP'
            except AttributeError:
                pass
            return 'TIMESTAMP'
        elif dtype == PandasTypes.TIME:
            try:
                if column.dt.tz:
                    return 'TIMESTAMP'
            except AttributeError:
                pass
            return 'TIMESTAMP'
        elif dtype == PandasTypes.DATE:
            return 'DATE'
        elif dtype == PandasTypes.STRING:
            return 'CHAR(255)'
        elif dtype == PandasTypes.CATEGORICAL:
            return 'CHAR(255)'
        elif dtype == PandasTypes.BYTES:
            return 'CHAR(255)'
        elif dtype in (PandasTypes.FLOATING, PandasTypes.DECIMAL, PandasTypes.MIXED_INTEGER_FLOAT):
            return 'NUMBER'
        elif dtype == PandasTypes.INTEGER:
            max_int, min_int = column.max(), column.min()
            if np.int16(max_int) == max_int and np.int16(min_int) == min_int:
                return 'NUMBER'
            elif np.int32(max_int) == max_int and np.int32(min_int) == min_int:
                return 'NUMBER'
            else:
                return 'NUMBER'
        elif dtype == PandasTypes.BOOLEAN:
            return 'CHAR(52)'
        elif dtype in (PandasTypes.TIMEDELTA, PandasTypes.TIMEDELTA64, PandasTypes.PERIOD):
            return 'NUMBER'
        elif dtype == PandasTypes.EMPTY:
            return 'CHAR(255)'
        else:
            print(f'Invalid datatype provided: {dtype}')

        return 'CHAR(255)'

    def export(
        self,
        df: DataFrame,
        schema_name: str = None,
        table_name: str = None,
        if_exists: Union[str, ExportWritePolicy] = ExportWritePolicy.REPLACE,
        unique_conflict_method: str = None,
        unique_constraints: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Exports a data frame to OracleDB. If table doesn't exist, 
        the table is automatically created. Supports insert, append, and upsert operations.
        
        Args:
            df (DataFrame): Data frame to export
            schema_name (str): Schema name (owner in Oracle). Defaults to current user
            table_name (str): Name of the table to export to
            if_exists (Union[str, ExportWritePolicy]): Policy if table exists:
                - 'fail' or ExportWritePolicy.FAIL: Raise error if table exists
                - 'replace' or ExportWritePolicy.REPLACE: Drop and recreate table (default)
                - 'append' or ExportWritePolicy.APPEND: Insert new rows (for upsert use unique_conflict_method)
            unique_conflict_method (str): Method to handle conflicts when appending:
                - 'update': Use MERGE statement to update matching rows (upsert)
            unique_constraints (List[str]): Columns that form the unique key for upsert matching
            **kwargs: Additional parameters passed to BaseSQL.export()
        
        Supported Operations:
            1. INSERT (if_exists='replace'): Create new table or replace existing one
            2. APPEND (if_exists='append'): Add new rows to existing table
            3. UPSERT (if_exists='append' + unique_conflict_method='update'): Insert or update rows based on unique_constraints
        
        Example:
            # Simple insert
            oracle.export(df, table_name='my_table', if_exists='replace')
            
            # Append to existing table
            oracle.export(df, table_name='my_table', if_exists='append')
            
            # Upsert (insert or update based on id)
            oracle.export(df, table_name='my_table', if_exists='append',
                         unique_conflict_method='update', unique_constraints=['id'])
        """
        if table_name is None:
            raise Exception('Please provide a table_name argument in the export method.')
        
        # Convert string if_exists to ExportWritePolicy enum if needed
        if isinstance(if_exists, str):
            if_exists_lower = if_exists.lower()
            if if_exists_lower == 'fail':
                if_exists = ExportWritePolicy.FAIL
            elif if_exists_lower == 'replace':
                if_exists = ExportWritePolicy.REPLACE
            elif if_exists_lower == 'append':
                if_exists = ExportWritePolicy.APPEND
            else:
                if_exists = ExportWritePolicy.REPLACE
        
        # When upsert is needed, force APPEND mode
        if unique_conflict_method and unique_constraints:
            if_exists = ExportWritePolicy.APPEND
        
        # Call parent export with all parameters
        super().export(
            df,
            **kwargs,
            schema_name=schema_name,
            table_name=table_name,
            if_exists=if_exists,
            unique_conflict_method=unique_conflict_method,
            unique_constraints=unique_constraints,
            # Oracle cursor execute will automatically add a semicolon at the end of the query.
            skip_semicolon_at_end=True
        )

    def upload_dataframe(
        self,
        cursor: Any,
        df: DataFrame,
        db_dtypes: List[str],
        dtypes: List[str],
        full_table_name: str,
        buffer: Union[IO, None] = None,
        **kwargs,
    ) -> None:
        def serialize_obj(val):
            if type(val) is dict or type(val) is np.ndarray:
                return simplejson.dumps(
                    val,
                    default=encode_complex,
                    ignore_nan=True,
                )
            elif type(val) is list and len(val) >= 1 and type(val[0]) is dict:
                return simplejson.dumps(
                    val,
                    default=encode_complex,
                    ignore_nan=True,
                )
            return val

        # Create values
        df_ = df.copy()
        columns = df_.columns
        for col in columns:
            dtype = df_[col].dtype
            if dtype == PandasTypes.OBJECT:
                df_[col] = df_[col].apply(lambda x: serialize_obj(x))
            elif dtype in (
                PandasTypes.MIXED,
                PandasTypes.UNKNOWN_ARRAY,
                PandasTypes.COMPLEX,
            ):
                df_[col] = df_[col].astype('string')

            # Remove extraneous surrounding double quotes
            # that get added while performing conversion to string.
            df_[col] = df_[col].apply(lambda x: x.strip('"') if x and isinstance(x, str) else x)
        df_.fillna('', inplace=True)
        values = list(df_.itertuples(index=False, name=None))

        # Create values placeholder
        colmn_names = df.columns.tolist()
        values_placeholder = ""
        for i in range(0, len(colmn_names)):
            values_placeholder += f':{str(i + 1)},'

        insert_sql = f'INSERT INTO {full_table_name} VALUES({values_placeholder.rstrip(",")})'
        cursor.executemany(insert_sql, values)

    def upload_dataframe_fast(
        self,
        df: DataFrame,
        schema_name: str,
        table_name: str,
        if_exists: ExportWritePolicy = None,
        **kwargs,
    ):
        """
        Uploads dataframe to Oracle database with support for upsert operations.
        
        Args:
            df (DataFrame): Data frame to upload
            schema_name (str): Schema name (owner in Oracle). Uses current user if None
            table_name (str): Table name to upload to
            if_exists (ExportWritePolicy): Policy if table exists:
                - REPLACE: Delete existing data and insert new
                - APPEND: Insert new rows only
                - FAIL: Raise error if table exists
            **kwargs: Additional parameters including:
                - unique_conflict_method: 'update' for upsert
                - unique_constraints: List of columns for matching in upsert
        
        Raises:
            Exception: If data insertion fails, transaction is rolled back
        """
        unique_conflict_method = kwargs.get('unique_conflict_method')
        unique_constraints = kwargs.get('unique_constraints')

        full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name

        # Prepare dataframe
        df_ = df.copy()
        columns = list(df_.columns)
        
        # Early return if dataframe is empty
        if df_.empty:
            logger.info(f'DataFrame is empty, skipping upload to {full_table_name}')
            return
        
        # Clean and prepare data for all columns
        for col in columns:
            dtype = df_[col].dtype
            if dtype == PandasTypes.OBJECT:
                df_[col] = df_[col].apply(
                    lambda x: simplejson.dumps(x, default=encode_complex, ignore_nan=True)
                    if type(x) in (dict, np.ndarray, list) else x
                )
            elif dtype in (PandasTypes.MIXED, PandasTypes.UNKNOWN_ARRAY, PandasTypes.COMPLEX):
                df_[col] = df_[col].astype('string')
            
            # Remove extraneous surrounding double quotes
            df_[col] = df_[col].apply(lambda x: x.strip('"') if x and isinstance(x, str) else x)
        
        df_.fillna('', inplace=True)
        values = list(df_.itertuples(index=False, name=None))
        
        try:
            # Handle upsert with MERGE statement
            if unique_conflict_method and unique_constraints and if_exists == ExportWritePolicy.APPEND:
                if UNIQUE_CONFLICT_METHOD_UPDATE == unique_conflict_method:
                    with self.conn.cursor() as cur:
                        # Build MERGE statement for upsert
                        # Match rows based on unique_constraints, update all columns if matched, insert if not matched
                        on_clause = ' AND '.join([f't.{col} = s.{col}' for col in unique_constraints])
                        update_clause = ', '.join([f't.{col} = s.{col}' for col in columns])
                        insert_cols = ', '.join(columns)
                        insert_vals = ', '.join([f's.{col}' for col in columns])
                        
                        # Oracle MERGE requires column aliases in USING clause
                        merge_sql = f"""
MERGE INTO {full_table_name} t
USING (SELECT {', '.join([f':{i+1} as {col}' for i, col in enumerate(columns)])} FROM DUAL) s
ON ({on_clause})
WHEN MATCHED THEN
    UPDATE SET {update_clause}
WHEN NOT MATCHED THEN
    INSERT ({insert_cols}) VALUES ({insert_vals})
                        """
                        
                        # Execute merge for each row
                        row_count = 0
                        for value_tuple in values:
                            cur.execute(merge_sql, value_tuple)
                            row_count += 1
                        
                        self.conn.commit()
                        logger.info(f'Successfully UPSERTED {row_count} rows to {full_table_name}')
                    return
            
            # Standard insert for REPLACE or regular APPEND (no upsert)
            with self.conn.cursor() as cur:
                # Create INSERT statement
                values_placeholder = ', '.join([f':{i+1}' for i in range(len(columns))])
                insert_sql = f'INSERT INTO {full_table_name} ({", ".join(columns)}) VALUES({values_placeholder})'
                
                # Execute insert
                row_count = len(values)
                cur.executemany(insert_sql, values)
                self.conn.commit()
                logger.info(f'Successfully INSERTED {row_count} rows to {full_table_name}')
        
        except Exception as e:
            # Rollback on error to prevent partial inserts
            try:
                self.conn.rollback()
                logger.error(f'Transaction rolled back due to error: {e}')
            except Exception as rollback_error:
                logger.error(f'Error during rollback: {rollback_error}')
            
            raise Exception(f'Failed to upload dataframe to {full_table_name}: {str(e)}')
