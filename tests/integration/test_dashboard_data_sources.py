"""load_known_cities() contra el reports/known_cities.json REAL commiteado
al repo, no un archivo fabricado por el test.

tests/unit/dashboard/test_data_sources.py mockea el archivo con tmp_path a
propósito para aislar el parseo — nunca prueba que el archivo real que
efectivamente viaja a docker/dashboard.Dockerfile (COPY reports/ reports/)
existe y es válido. Ese hueco es justo el que causó el bug real del commit
a63c4bd (2026-08-17): el dashboard leía data/processed/train.parquet en
runtime, carpeta que nunca se copia a la imagen, y crasheaba con
FileNotFoundError al abrir la página de predicción — ningún test lo
atrapó porque ninguno tocaba el archivo real."""

from churn_mlops.dashboard.data_sources import load_known_cities


def test_known_cities_json_is_present_and_nonempty() -> None:
    cities = load_known_cities()

    assert len(cities) > 0
    assert all(isinstance(city, str) for city in cities)
