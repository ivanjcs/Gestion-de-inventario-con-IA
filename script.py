import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ==========================================
# PARTE 1: GENERACIÓN DE DATOS Y ML
# ==========================================

# 1. Configuración de Semilla (Para que siempre de lo mismo)
np.random.seed(42)

# 2. Eje Temporal (Todo el año 2025)
fechas = pd.date_range(start='2025-01-01', end='2025-12-31')
dias = np.arange(len(fechas)) # 0, 1, 2... 364

# 3. Tendencia (Crecimiento lineal del negocio)
# Empezamos vendiendo 100 y terminamos en 150
tendencia = np.linspace(start=100, stop=150, num=len(fechas))

# 4. Estacionalidad (La onda anual)
# Usamos una amplitud de 20 para que se note la diferencia invierno/verano
estacionalidad = 20 * np.sin((2 * np.pi / 365) * dias)

# 5. Ruido (La variable aleatoria)
# Media 0 y desvio 10 para darle realismo
ruido = np.random.normal(loc=0, scale=10, size=len(fechas))

# 6. Ecuación Final de la Demanda
demanda_simulada = tendencia + estacionalidad + ruido

# Las ventas no pueden ser negativas. Si el ruido resta mucho, forzamos a 0.
demanda_simulada = np.maximum(0, demanda_simulada)

# 7. Guardar todo en un DataFrame
df_ventas = pd.DataFrame({
    'Fecha': fechas,
    'Demanda_Total': demanda_simulada.astype(int) # Convertimos a enteros
})

# --- Feature Engineering (Preparar datos para la IA) ---

# Asegurarnos de que 'Fecha' es datetime
df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'])

# Sacamos datos del calendario
df_ventas['Mes'] = df_ventas['Fecha'].dt.month
df_ventas['Dia'] = df_ventas['Fecha'].dt.day
df_ventas['Dia_Semana'] = df_ventas['Fecha'].dt.dayofweek

# Generamos los lags (historia reciente)
# Lag 1: ¿cuanto se vendió ayer?
df_ventas['Ventas_Ayer'] = df_ventas['Demanda_Total'].shift(1)
# Lag 7: ¿Cuanto se vendio hace una semana?
df_ventas['Ventas_Semana_Ant'] = df_ventas['Demanda_Total'].shift(7)

# Promedio móvil (Tendencia suave de la semana)
df_ventas['Promedio_7dias'] = df_ventas['Demanda_Total'].rolling(window=7).mean()

# Borramos los nulos que se generaron al principio por los lags
df_ventas = df_ventas.dropna().reset_index(drop=True)

# --- División Train / Test ---

# Definimos la fecha de corte (probamos con los últimos 2 meses)
fecha_corte = '2025-11-01'

# Entrenamiento: Todo lo de antes
train = df_ventas[df_ventas['Fecha'] < fecha_corte]
# Prueba: Todo lo de después
test = df_ventas[df_ventas['Fecha'] >= fecha_corte]

# Definimos las columnas (X) y el objetivo (y)
feature_cols = ['Mes', 'Dia', 'Dia_Semana', 'Ventas_Ayer', 'Ventas_Semana_Ant', 'Promedio_7dias']
target_col = 'Demanda_Total'

X_train = train[feature_cols]
y_train = train[target_col]

X_test = test[feature_cols]
y_test = test[target_col]

# --- Entrenamiento del Modelo ---

# Usamos 100 árboles
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Le enseñamos al modelo con los datos de entrenamiento
regressor.fit(X_train, y_train)

# Hacemos la predicción sobre el test para ver qué tal le fue
y_pred = regressor.predict(X_test)

# Calculamos el error
mae = mean_absolute_error(y_test, y_pred)
print(f"--> Error del Modelo (MAE): {mae:.2f} unidades")


# ==========================================
# PARTE 2: EL SIMULADOR (SimPy)
# ==========================================

class Deposito:
    def __init__(self, env, stock_inicial, costo_ordenar, costo_mantenimiento, costo_faltante):
        # --- Entorno de Simulación ---
        self.env = env
        
        # --- Variables de Estado ---
        self.stock_actual = stock_inicial
        
        # --- Costos ---
        self.k = costo_ordenar          # Costo fijo por pedido
        self.h = costo_mantenimiento    # Costo por guardar cajas
        self.p = costo_faltante         # Costo por perder ventas
        
        # --- Acumuladores (para ver resultados al final) ---
        self.costo_total_ordenar = 0.0
        self.costo_total_mantenimiento = 0.0
        self.costo_total_faltante = 0.0
        self.total_ventas_perdidas = 0
        self.total_pedidos_realizados = 0 # Contador extra para saber cuantos camiones vinieron
        
        # Log para Power BI
        self.daily_logs = []

    # --- Proceso 1: Cobrar almacenamiento ---
    def control_costos_almacenamiento(self):
        while True:
            # Esperamos 1 día (24hs)
            yield self.env.timeout(1) 

            # Calculamos cuanto nos cuesta guardar lo que sobró hoy
            costo_diario = self.stock_actual * self.h
            self.costo_total_mantenimiento += costo_diario

            # (Opcional) Aquí podrías guardar log simple, pero lo hacemos en operacion_diaria
    
    # --- Proceso 2: Llegada del camión ---
    def llegada_pedido(self, cantidad, lead_time):
        # Simula el tiempo de viaje (espera)
        yield self.env.timeout(lead_time)
        
        # Llegó el camión, sumamos al stock
        self.stock_actual += cantidad
        
        # Sumamos el costo del flete
        self.costo_total_ordenar += self.k
        self.total_pedidos_realizados += 1
        
    # --- Proceso 3: Operación Diaria (Lo principal) ---
    def operacion_diaria(self, df_datos, estrategia_func, estrategia_nombre):
        for i, fila in df_datos.iterrows():
            # A. Demanda del día
            demanda_hoy = fila['Demanda_Total']
            
            # Inicializamos la variable del día en 0 antes de verificar nada
            ventas_perdidas_hoy = 0
            
            # B. Venta
            if self.stock_actual >= demanda_hoy:
                # Hay stock, vendemos todo
                self.stock_actual -= demanda_hoy
            else:
                # No hay suficiente
                ventas_perdidas_hoy = demanda_hoy - self.stock_actual
                self.stock_actual = 0
                
                # Anotamos la pérdida en acumuladores
                self.total_ventas_perdidas += ventas_perdidas_hoy
                self.costo_total_faltante += ventas_perdidas_hoy * self.p

            # C. Decisión 
            # ¿Cuanto pido?
            cantidad_a_pedir = estrategia_func(self.stock_actual, dia_actual=i)
            
            if cantidad_a_pedir > 0:
                # Si dice que pida, llamamos al camión (tarda 3 días)
                self.env.process(self.llegada_pedido(cantidad_a_pedir, lead_time=3))
            
            # Logging Daily Data (Guardar foto del día)
            self.daily_logs.append({
                'Fecha': fila['Fecha'],
                'Estrategia': estrategia_nombre, # Guardamos el nombre "IA" o "Clasica"
                'Demanda': demanda_hoy,
                'Venta_Real': demanda_hoy - ventas_perdidas_hoy,
                'Ventas_Perdidas': ventas_perdidas_hoy,
                'Stock_Final_Dia': self.stock_actual,
                'Pedido_Cantidad': cantidad_a_pedir,
                'Costo_Ordenar_Dia': self.k if cantidad_a_pedir > 0 else 0,
                'Costo_Mantenimiento_Dia': self.stock_actual * self.h, 
                'Costo_Faltante_Dia': ventas_perdidas_hoy * self.p
            })
            
            yield self.env.timeout(1)

# ==========================================
# PARTE 3: ESTRATEGIAS Y EJECUCIÓN
# ==========================================

# Estrategia Clásica (Punto de pedido fijo)
def estrategia_clasica(stock_actual, dia_actual=None):
    # Valores fijos definidos a ojo (heurística)
    punto_pedido = 500
    cantidad_fija = 1000
    
    # Si bajamos del punto de pedido, compramos
    if stock_actual <= punto_pedido:
        return cantidad_fija
    else:
        return 0 # No pedimos nada

# Estrategia IA
def estrategia_ia(stock_actual, dia_actual):
    """
    Usa el Random Forest para estimar la demanda del Lead Time (3 días).
    """
    lead_time = 3
    stock_seguridad = 50 # Un colchón mínimo por si la IA se equivoca un poco
    
    # Validamos no salirnos del calendario
    if dia_actual + lead_time >= len(df_ventas):
        return 0

    # --- PASO CLAVE: PREDICCIÓN ---
    # Miramos las features de los próximos 3 días (t+1, t+2, t+3)
    features_futuras = df_ventas.iloc[dia_actual+1 : dia_actual+1+lead_time][feature_cols]
    
    if len(features_futuras) == 0:
        return 0

    # Le preguntamos al modelo: "¿Cuánto se va a vender?"
    prediccion_demanda = regressor.predict(features_futuras).sum()
    
    # --- CÁLCULO DEL PEDIDO ---
    stock_objetivo = prediccion_demanda + stock_seguridad
    
    # Si lo que tengo no alcanza para cubrir lo que viene...
    if stock_actual < stock_objetivo:
        cantidad_a_pedir = stock_objetivo - stock_actual
        return int(cantidad_a_pedir) # Pedimos la diferencia exacta
    else:
        return 0 # Estamos cubiertos, no pedir nada


# ==========================================
# EJECUCIÓN COMPARATIVA (IA vs CLÁSICA)
# ==========================================
dias_totales = len(df_ventas) # El límite de tiempo para cortar el bucle

print(f"\n--> Corriendo simulaciones por {dias_totales} días...")

# --- SIMULACIÓN 1: ESTRATEGIA IA 🤖 ---
env_ia = simpy.Environment()
deposito_ia = Deposito(env_ia, stock_inicial=1000, costo_ordenar=50, costo_mantenimiento=2, costo_faltante=10)

env_ia.process(deposito_ia.control_costos_almacenamiento())
# Pasamos la función estrategia_ia y el nombre "IA"
env_ia.process(deposito_ia.operacion_diaria(df_ventas, estrategia_ia, "IA"))

# until=dias_totales evita el bucle infinito
env_ia.run(until=dias_totales) 

costo_total_ia = (deposito_ia.costo_total_ordenar + 
                  deposito_ia.costo_total_mantenimiento + 
                  deposito_ia.costo_total_faltante)


# --- SIMULACIÓN 2: ESTRATEGIA CLÁSICA 📉 ---
env_clasica = simpy.Environment()
deposito_clasica = Deposito(env_clasica, stock_inicial=1000, costo_ordenar=50, costo_mantenimiento=2, costo_faltante=10)

env_clasica.process(deposito_clasica.control_costos_almacenamiento())
# Pasamos la función estrategia_clasica y el nombre "Clasica"
env_clasica.process(deposito_clasica.operacion_diaria(df_ventas, estrategia_clasica, "Clasica"))

env_clasica.run(until=dias_totales)

costo_total_clasica = (deposito_clasica.costo_total_ordenar + 
                       deposito_clasica.costo_total_mantenimiento + 
                       deposito_clasica.costo_total_faltante)

# ==========================================
# EXPORTACIÓN DE RESULTADOS
# ==========================================

# 1. Detalle Diario (Para gráficos de tiempo en PBI)
df_logs_ia = pd.DataFrame(deposito_ia.daily_logs)
df_logs_cl = pd.DataFrame(deposito_clasica.daily_logs)
df_total_diario = pd.concat([df_logs_ia, df_logs_cl], ignore_index=True)

# Calculamos el costo total diario sumando las partes
df_total_diario['Costo_Total_Dia'] = (df_total_diario['Costo_Ordenar_Dia'] + 
                                      df_total_diario['Costo_Mantenimiento_Dia'] + 
                                      df_total_diario['Costo_Faltante_Dia'])

# 2. Resumen KPI (Para tarjetas y comparaciones globales)
kpi_data = [
    {
        'Estrategia': 'IA',
        'Pedidos_Total': deposito_ia.total_pedidos_realizados,
        'Ventas_Perdidas_Total': deposito_ia.total_ventas_perdidas,
        'Costo_Ordenar_Total': deposito_ia.costo_total_ordenar,
        'Costo_Mantenimiento_Total': deposito_ia.costo_total_mantenimiento,
        'Costo_Faltante_Total': deposito_ia.costo_total_faltante,
        'Costo_Operativo_Final': costo_total_ia
    },
    {
        'Estrategia': 'Clasica',
        'Pedidos_Total': deposito_clasica.total_pedidos_realizados,
        'Ventas_Perdidas_Total': deposito_clasica.total_ventas_perdidas,
        'Costo_Ordenar_Total': deposito_clasica.costo_total_ordenar,
        'Costo_Mantenimiento_Total': deposito_clasica.costo_total_mantenimiento,
        'Costo_Faltante_Total': deposito_clasica.costo_total_faltante,
        'Costo_Operativo_Final': costo_total_clasica
    }
]
df_kpi = pd.DataFrame(kpi_data)

# Guardar archivos
df_total_diario.to_csv('simulacion_diaria_detalle.csv', index=False)
df_kpi.to_csv('simulacion_kpi_resumen.csv', index=False)

print(f"--> Simulación completada.")
print(f"--> Archivos generados: 'simulacion_diaria_detalle.csv' y 'simulacion_kpi_resumen.csv'")

# Agrupamos por estrategia y miramos el mínimo stock que se registró
print(df_total_diario.groupby('Estrategia')['Stock_Final_Dia'].min())