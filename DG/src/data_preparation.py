"""
Preparacion de datos de Argentina para Dynare.

Lee Arg.xlsx (sheet 'data_Arg') y genera argmodel_data.mat
con las 7 variables observables en formato Dynare.

Variables (columnas AC-AI en Excel):
- dy: Crecimiento del PIB
- dc: Crecimiento del consumo
- dinve: Crecimiento de la inversion
- labobs: Horas trabajadas
- pinfobs: Inflacion
- dw: Crecimiento de salarios reales
- robs: Tasa de interes

Periodo: 2004Q2 - 2025Q3 (86 observaciones)
"""

import pandas as pd
import numpy as np
from scipy.io import savemat
from pathlib import Path


def prepare_argentina_data(
    input_file: str = None,
    output_file: str = None,
    sheet_name: str = 'data_Arg'
) -> pd.DataFrame:
    """
    Lee datos de Argentina y los convierte a formato Dynare (.mat).

    Args:
        input_file: Ruta al archivo Excel de entrada (default: ../data/Arg.xlsx)
        output_file: Ruta al archivo .mat de salida (default: ../data/argmodel_data.mat)
        sheet_name: Nombre del sheet en Excel (default: 'data_Arg')

    Returns:
        DataFrame con los datos procesados
    """
    # Determinar rutas por defecto
    base_path = Path(__file__).parent.parent

    if input_file is None:
        input_file = base_path / 'data' / 'Arg.xlsx'
    else:
        input_file = Path(input_file)

    if output_file is None:
        output_file = base_path / 'data' / 'argmodel_data.mat'
    else:
        output_file = Path(output_file)

    print(f"Leyendo datos de: {input_file}")
    print(f"Sheet: {sheet_name}")

    # Leer Excel - estructura del archivo:
    # Fila 0: titulos de seccion (ignorar)
    # Fila 1: nombres de columnas (header)
    # Fila 2: vacia (ignorar)
    # Fila 3+: datos
    # Columnas 28-34: dc, dinve, dy, labobs, pinfobs, dw, robs
    df = pd.read_excel(
        input_file,
        sheet_name=sheet_name,
        header=1,  # Segunda fila como header (indice 1)
        skiprows=[2]  # Saltar fila vacia despues del header
    )

    # Limpiar espacios en nombres de columnas
    df.columns = df.columns.str.strip()

    print(f"Dimensiones del archivo: {df.shape}")
    print(f"Columnas disponibles: {list(df.columns)}")

    # Las 7 variables observables en el orden que espera Dynare
    # El orden DEBE coincidir con varobs en el .mod file
    variables = ['dy', 'dc', 'dinve', 'labobs', 'pinfobs', 'dw', 'robs']

    # Verificar que las variables existen
    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        print(f"Columnas encontradas: {[c for c in df.columns if 'd' in c.lower() or 'obs' in c.lower() or 'pinf' in c.lower()]}")
        raise ValueError(f"Variables faltantes en el archivo: {missing_vars}")

    # Extraer solo las variables observables
    data_df = df[variables].copy()

    # Eliminar filas con NaN
    data_df = data_df.dropna()

    print(f"\nDatos procesados:")
    print(f"  Observaciones: {len(data_df)}")
    print(f"  Variables: {variables}")
    print(f"\nEstadisticas descriptivas:")
    print(data_df.describe())

    # Crear diccionario para .mat
    # Dynare espera cada variable como un vector columna
    data_dict = {}
    for var in variables:
        data_dict[var] = data_df[var].values.reshape(-1, 1).astype(np.float64)

    # Guardar como .mat
    savemat(str(output_file), data_dict)

    print(f"\nDatos guardados en: {output_file}")
    print(f"Variables en .mat: {list(data_dict.keys())}")

    return data_df


def verify_data(mat_file: str = None) -> dict:
    """
    Verifica que el archivo .mat contiene los datos correctamente.

    Args:
        mat_file: Ruta al archivo .mat (default: ../data/argmodel_data.mat)

    Returns:
        Diccionario con los datos cargados
    """
    from scipy.io import loadmat

    if mat_file is None:
        mat_file = Path(__file__).parent.parent / 'data' / 'argmodel_data.mat'

    print(f"Verificando archivo: {mat_file}")

    data = loadmat(str(mat_file))

    # Filtrar variables internas de MATLAB
    variables = {k: v for k, v in data.items() if not k.startswith('__')}

    print(f"\nVariables en el archivo:")
    for var, values in variables.items():
        print(f"  {var}: shape={values.shape}, dtype={values.dtype}")
        print(f"    min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

    return variables


if __name__ == '__main__':
    # Ejecutar preparacion de datos
    print("="*60)
    print("PREPARACION DE DATOS DE ARGENTINA PARA DYNARE")
    print("="*60)

    df = prepare_argentina_data()

    print("\n" + "="*60)
    print("VERIFICACION DE DATOS")
    print("="*60)

    verify_data()

    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
