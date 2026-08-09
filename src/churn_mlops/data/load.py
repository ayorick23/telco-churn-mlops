from pathlib import Path

import pandas as pd

from churn_mlops.data.schema import raw_telco_schema


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Lee el CSV crudo de Telco Customer Churn y lo valida contra raw_telco_schema.

    Lanza pandera.errors.SchemaError si el CSV no cumple el schema esperado.
    """
    df = pd.read_csv(path)
    validated_df: pd.DataFrame = raw_telco_schema.validate(df)
    return validated_df
