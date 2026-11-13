# Análisis de la comparativa EasyOCR vs SmolVLM

Comparativa sobre el mismo conjunto de **217 imágenes** de matrículas, usando el mismo criterio de evaluación (acierto exacto, errores de carácter, CER y tiempo por imagen). Los datos provienen de `resultados/comparativa.csv` y `resultados/detalle_comparativa.csv`.

## Resultados globales

| Métrica | EasyOCR | SmolVLM |
|---|---|---|
| Imágenes procesadas | 217 | 217 |
| Aciertos exactos | 146 | 191 |
| Accuracy exacta | **67,28 %** | **88,02 %** |
| Errores de carácter | 166 | 121 |
| CER global | 0,1088 (10,88 %) | 0,0793 (7,93 %) |
| Accuracy por carácter | 89,12 % | 92,07 % |
| Lecturas vacías | 7 (3,23 %) | 0 (0 %) |
| Tiempo total | **304,7 s** | 3077,6 s |
| Tiempo medio/imagen | **1,40 s** | 14,18 s |

## Interpretación

**SmolVLM es claramente más preciso.** Acierta 45 imágenes más que EasyOCR (+20,7 puntos de accuracy exacta) y comete 45 errores de carácter menos. La diferencia se mantiene en el CER (7,93 % frente a 10,88 %), por lo que no solo falla menos, sino que cuando falla suele estar más cerca de la matrícula correcta.

**EasyOCR es mucho más rápido.** Tarda unas 10 veces menos por imagen (1,4 s frente a 14,2 s), lo que lo hace más adecuado si el tiempo es crítico o si se procesan muchos fotogramas.

**EasyOCR deja más imágenes sin leer.** Sus 7 lecturas vacías (3,23 %) contrastan con SmolVLM, que siempre devuelve alguna predicción. Aun así, SmolVLM tampoco es perfecto: en algunos casos produce salidas degeneradas (repeticiones de dígitos o texto sin relación con la matrícula).

## Análisis de los fallos

- **18 imágenes fallan con ambos modelos**, por lo que son los casos más difíciles (baja calidad, ángulo, oclusión, etc.).
- **53 imágenes falla solo EasyOCR**: SmolVLM las resuelve correctamente.
- **8 imágenes falla solo SmolVLM**: EasyOCR las acierta.

Los errores de EasyOCR son mayoritariamente:
- caracteres espurios al inicio (`L0192DSP`, `41159FPG`, `11319FSX`),
- confusiones entre caracteres similares (`M`/`N`, `V`/`W`, `H`/`K`, `0`/`O`, `8`/`B`),
- lecturas parciales o vacías.

Los errores de SmolVLM, aunque menos frecuentes, siguen el mismo patrón de sustitución/inserción de un carácter (`0649DHS` → `0649DSH`), con algún caso de salida degenerada (`1777...`, `BENTON`).

La confianza de EasyOCR es un buen indicador: media de **0,797** en global, **0,887** en aciertos y **0,612** en fallos.

## Conclusión

SmolVLM ofrece una precisión notablemente superior (88 % frente a 67 % de acierto exacto) y es más robusto al no dejar lecturas vacías, a costa de ser unas 10 veces más lento. EasyOCR sigue siendo una opción válida cuando prima la velocidad, pero requiere filtrado o corrección posterior por su mayor tasa de error.
