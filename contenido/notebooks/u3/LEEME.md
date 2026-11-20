# Datos de la Unidad 3

Todos proceden del repositorio de aprendizaje automático de la Universidad de California en
Irvine (UCI) y tienen licencia CC BY 4.0. Se han traducido los nombres de las columnas.
`mayoristas.csv` conserva todos los valores originales; los otros tres archivos son resúmenes
de Online Retail II.

| Archivo | Origen | Cita |
|---|---|---|
| `mayoristas.csv` | Wholesale customers (440 clientes; gasto anual en unidades monetarias; canal 1 = Horeca, 2 = minorista; región 1 = Lisboa, 2 = Oporto, 3 = otras), sin cambios | Cardoso (2013), https://doi.org/10.24432/C5030X |
| `clientes_rfm.csv` | Online Retail II, sin facturas anuladas (que empiezan por C), líneas con cantidad o precio no positivos ni líneas sin cliente. Por cliente: días desde su última compra hasta el 10/12/2011, número de facturas y suma de cantidad × precio | Chen (2012), https://doi.org/10.24432/C5CG6D |
| `cestas.csv` | Online Retail II con los mismos filtros, solo artículos (códigos que empiezan por una cifra, sin gastos de envío ni otros cargos) y solo los 100 presentes en más facturas. Una fila por factura y artículo | Chen (2012), https://doi.org/10.24432/C5CG6D |
| `productos.csv` | Descripción más frecuente de cada uno de esos 100 artículos | Chen (2012), https://doi.org/10.24432/C5CG6D |
