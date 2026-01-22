# Instrucciones de Instalación

Este documento describe cómo instalar los requisitos externos para ejecutar
la replicación directa de Smets & Wouters (2007) usando oct2py.

## Requisitos

1. **GNU Octave** - Intérprete compatible con MATLAB
2. **Dynare** - Toolkit para modelos DSGE
3. **Sims VARtools** - Funciones BVAR de Christopher Sims

---

## 1. Instalación de GNU Octave (Windows)

### Paso 1: Descargar
- Ir a: https://www.gnu.org/software/octave/download
- Descargar el instalador de Windows (ej: `octave-8.4.0-w64-installer.exe`)

### Paso 2: Instalar
- Ejecutar el instalador
- Ruta sugerida: `C:\Octave\Octave-8.4.0`
- Completar la instalación

### Paso 3: Agregar a PATH
1. Abrir "Variables de entorno del sistema"
2. En "Variables del sistema", seleccionar `Path` → Editar
3. Agregar: `C:\Octave\Octave-8.4.0\mingw64\bin`
4. Aceptar todos los cambios

### Paso 4: Verificar instalación
Abrir CMD o PowerShell y ejecutar:
```
octave --version
```
Debería mostrar: `GNU Octave, version 8.4.0`

---

## 2. Instalación de Dynare (Windows)

### Paso 1: Descargar
- Ir a: https://www.dynare.org/download/
- Descargar la última versión estable para Windows (ej: `dynare-6.2-win.exe`)

### Paso 2: Instalar
- Ejecutar el instalador
- Ruta sugerida: `C:\dynare\6.2`
- Completar la instalación

### Paso 3: Anotar la ruta
La ruta al folder MATLAB de Dynare es necesaria para la configuración.
Ejemplo: `C:\dynare\6.2\matlab`

**Nota**: NO es necesario agregar Dynare al PATH del sistema.
Se configura desde Python usando `oct2py`.

### Paso 4: Verificar instalación
Abrir Octave (GUI o CLI) y ejecutar:
```octave
addpath('C:\dynare\6.2\matlab')
dynare_version
```
Debería mostrar la versión de Dynare.

---

## 3. Descargar Sims VARtools

Las funciones BVAR requieren herramientas de Christopher Sims.

### Paso 1: Descargar archivos
Ir a: http://sims.princeton.edu/yftp/VARtools/

Descargar los siguientes archivos:
- `varprior.m`
- `rfvar3.m`
- `matrictint.m`

### Paso 2: Colocar en el repositorio
Copiar los archivos descargados a:
```
MECTMT11/repo/
```
Junto a los otros archivos .m del modelo original.

---

## 4. Instalar dependencias Python

Desde el directorio del proyecto:
```bash
pip install -r direct_replication/requirements.txt
```

O instalar oct2py directamente:
```bash
pip install oct2py
```

---

## 5. Verificación Completa

Ejecutar el siguiente script Python para verificar la instalación:

```python
from oct2py import Oct2Py
import os

# Configurar rutas (MODIFICAR SEGÚN TU INSTALACIÓN)
DYNARE_PATH = r'C:\dynare\6.2\matlab'
REPO_PATH = r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11\repo'

# Crear sesión Octave
oc = Oct2Py()

# Agregar paths
oc.addpath(DYNARE_PATH)
oc.addpath(REPO_PATH)

# Verificar Dynare
print("Verificando Dynare...")
result = oc.eval('exist("dynare")', nout=1)
if result == 2:
    print("✓ Dynare encontrado")
else:
    print("✗ Dynare NO encontrado")

# Verificar VARtools
print("\nVerificando VARtools...")
for func in ['varprior', 'rfvar3', 'matrictint']:
    result = oc.eval(f'exist("{func}")', nout=1)
    if result == 2:
        print(f"✓ {func}.m encontrado")
    else:
        print(f"✗ {func}.m NO encontrado - Descargar de sims.princeton.edu")

# Cerrar sesión
oc.exit()

print("\n¡Verificación completa!")
```

---

## Solución de Problemas

### Error: "octave-cli not found"
- Verificar que Octave está en el PATH
- Reiniciar la terminal después de modificar PATH

### Error: "dynare command not found"
- Verificar la ruta DYNARE_PATH
- Asegurar que addpath() se ejecute antes de usar dynare

### Error: "undefined function varprior"
- Descargar los archivos de http://sims.princeton.edu/yftp/VARtools/
- Colocarlos en la carpeta repo/

### Error de permisos en Windows
- Ejecutar Octave/Python como administrador la primera vez
- O cambiar la ruta de instalación a una carpeta sin restricciones

---

## Rutas de Ejemplo (para configuration)

```python
# Windows típico
DYNARE_PATH = r'C:\dynare\6.2\matlab'
REPO_PATH = r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11\repo'

# Si Octave no está en PATH
import os
os.environ['OCTAVE_EXECUTABLE'] = r'C:\Octave\Octave-8.4.0\mingw64\bin\octave-cli.exe'
```

---

## Referencias

- [Oct2Py Documentation](https://oct2py.readthedocs.io/)
- [Dynare Manual](https://www.dynare.org/manual/)
- [Sims VARtools](http://sims.princeton.edu/yftp/VARtools/HEADER.html)
