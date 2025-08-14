import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import folium

# LEER IMÁGENES .tif
carpeta_tif = "./img"
resultados = []

for archivo in os.listdir(carpeta_tif):
    if archivo.lower().endswith((".tif", ".tiff")):
        ruta = os.path.join(carpeta_tif, archivo)
        with rasterio.open(ruta) as src:
            img = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                img = np.where(img == nodata, np.nan, img)
            promedio = np.nanmean(img)

        # Extraer nombre y fecha
        nombre_sin_ext = archivo.rsplit(".", 1)[0]
        for i, c in enumerate(nombre_sin_ext):
            if c.isdigit():
                lago = nombre_sin_ext[:i]
                fecha_str = nombre_sin_ext[i:]
                break
        fecha = datetime.strptime(fecha_str, "%Y-%m-%d")

        resultados.append({
            "Lago": lago.capitalize(),
            "Fecha": fecha,
            "Promedio": promedio
        })

# DataFrame base
df = pd.DataFrame(resultados).sort_values(by=["Lago", "Fecha"])
print(df)

# Calcular umbral por lago (mediana histórica)
umbral_por_lago = df.groupby("Lago")["Promedio"].median().to_dict()
df["Contaminado"] = df.apply(lambda row: int(row["Promedio"] > umbral_por_lago[row["Lago"]]), axis=1)
df["Mes"] = df["Fecha"].dt.month

# 10. SERIES TEMPORALES POR LAGO
predicciones_series = []

for lago in df["Lago"].unique():
    datos_lago = df[df["Lago"] == lago][["Fecha", "Promedio"]].rename(columns={"Fecha": "ds", "Promedio": "y"})
    
    # Modelo Prophet más estable
    modelo = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    modelo.add_seasonality(name="mensual", period=30.5, fourier_order=3)
    modelo.fit(datos_lago)

    # Predecir próximos 6 meses
    futuro = modelo.make_future_dataframe(periods=6, freq="MS")  # MS = mes inicio
    pronostico = modelo.predict(futuro)
    pronostico["Lago"] = lago
    predicciones_series.append(pronostico[["ds", "yhat", "Lago"]])

    # Rango ajustado del eje Y (solo ±20% de los datos reales)
    y_min = datos_lago["y"].min() * 0.9
    y_max = datos_lago["y"].max() * 1.1

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(datos_lago["ds"], datos_lago["y"], "bo-", label="Datos reales")
    plt.plot(pronostico["ds"], pronostico["yhat"], "r--", label="Pronóstico", alpha=0.8)
    plt.ylim(y_min, y_max)
    plt.xlabel("Fecha")
    plt.ylabel("Índice de Cianobacteria")
    plt.title(f"Predicción índice cianobacteria - {lago}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Unir todas las predicciones
predicciones_series = pd.concat(predicciones_series).rename(columns={"ds": "Fecha", "yhat": "PrediccionSerie"})

# 11. CLASIFICACIÓN BINARIA POR LAGO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

clasificadores = {}

for lago in df["Lago"].unique():
    datos_lago = df[df["Lago"] == lago]
    X = datos_lago[["Mes", "Promedio"]]
    y = datos_lago["Contaminado"]

    # Validación cruzada
    clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, X, y, cv=3)  
    print(f"=== Validación cruzada - {lago} ===")
    print(f"Precisión promedio: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)
    clasificadores[lago] = clf

    y_pred = clf.predict(X_test)
    print(f"\n=== Clasificación Binaria - {lago} ===")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels = [0, 1])
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Matriz de Confusión - {lago}")
    plt.show()

# 12. MODELO HÍBRIDO POR LAGO
hibridos = {}

df_hibrido = df.merge(predicciones_series, on=["Fecha", "Lago"], how="left")
df_hibrido["Mes"] = df_hibrido["Fecha"].dt.month

for lago in df_hibrido["Lago"].unique():
    datos_lago = df_hibrido[df_hibrido["Lago"] == lago]
    X = datos_lago[["Mes", "PrediccionSerie"]].fillna(datos_lago["Promedio"])
    y = datos_lago["Contaminado"]

    # Validación cruzada
    clf_h = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf_h, X, y, cv=3)
    print(f"=== Validación cruzada híbrido - {lago} ===")
    print(f"Precisión promedio: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf_h.fit(X_train, y_train)
    hibridos[lago] = clf_h

    y_pred_h = clf_h.predict(X_test)
    print(f"\n=== Modelo Híbrido - {lago} ===")
    print(classification_report(y_test, y_pred_h))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_h, labels=[0, 1])
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Matriz de Confusión Híbrido - {lago}")
    plt.show()

# 14. MAPA CON FOLIUM
coordenadas = {
    "Amatitlan": (14.4789, -90.6061),
    "Atitlan": (14.6900, -91.2000)
}

mapa = folium.Map(location=[14.6, -90.9], zoom_start=8)

for _, fila in df_hibrido.iterrows():
    lat, lon = coordenadas[fila["Lago"]]
    color = "red" if fila["Contaminado"] == 1 else "green"
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"{fila['Lago']} - {fila['Fecha'].date()} - Índice: {fila['Promedio']:.2f}"
    ).add_to(mapa)

mapa.save("mapa_cianobacteria.html")
print("Mapa guardado en 'mapa_cianobacteria.html'")
