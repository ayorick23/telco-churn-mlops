"""Traduce los nombres de feature que genera
`ColumnTransformer.get_feature_names_out()` (ej.
"categorical__Contract_Month-to-Month", "numeric__Age") a algo legible para
el usuario del dashboard (ej. "Contract: Month-to-Month", "Age"). No son
nombres inventados por FastAPI — son los que arma sklearn internamente al
transformar features (training/preprocessing.py)."""


def humanize_shap_feature_name(name: str, known_columns: list[str]) -> str:
    if "__" not in name:
        return name
    prefix, remainder = name.split("__", 1)
    if prefix == "numeric":
        return remainder
    if prefix == "categorical":
        for column in sorted(known_columns, key=len, reverse=True):
            if remainder == column:
                return column
            if remainder.startswith(f"{column}_"):
                category = remainder[len(column) + 1 :]
                return f"{column}: {category}"
    return remainder
