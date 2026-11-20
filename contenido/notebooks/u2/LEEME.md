# Datos de la Unidad 2

Todos proceden del repositorio de aprendizaje automático de la Universidad de California en
Irvine (UCI) y tienen licencia CC BY 4.0. Se han traducido los nombres de las columnas y no se
ha cambiado ningún valor, salvo en `ventas_diarias.csv`, que es un resumen.

| Archivo | Origen | Cita |
|---|---|---|
| `corazon.csv` | Heart Disease, archivo `processed.cleveland.data` (303 pacientes; `?` = dato que falta) | Janosi, Steinbrunn, Pfisterer y Detrano (1989), https://doi.org/10.24432/C52P4X |
| `credito.csv` | Statlog (German Credit Data), archivo `german.data` (1000 clientes; riesgo 1 = bueno, 2 = malo) | Hofmann (1994), https://doi.org/10.24432/C5NC77 |
| `compras.csv` | Online Shoppers Purchasing Intention Dataset (12 330 sesiones), sin cambios | Sakar y Kastro (2018), https://doi.org/10.24432/C5F88Q |
| `ventas_diarias.csv` | Online Retail II (sus dos hojas se solapan del 1 al 9/12/2010: esos días se toman solo de la primera): suma diaria de cantidad × precio, sin facturas anuladas (que empiezan por C) ni líneas con cantidad o precio no positivos; los días sin ventas valen 0 | Chen (2012), https://doi.org/10.24432/C5CG6D |

## Ficheros de código

Los ficheros `.py` de esta carpeta son los listados del libro que preparan un caso
(por ejemplo, `h_ventas.py` construye las variables de la serie de ventas). Los listados de
ampliación del cuaderno los importan (`from h_ventas import X, y`) para no repetir la
preparación. No hay que abrirlos ni modificarlos.
