<div align="left">
  🌐 <a href="#english">English</a> | 🇦🇷 <a href="#español">Español</a>
</div>

<div align="center">
  <h1 id="english">🚀 Digital Twin & ML Inventory Optimization</h1>
  <p><b>Discrete Event Simulation for Supply Chain & Inventory Control</b></p>
  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  ![SimPy](https://img.shields.io/badge/SimPy-Simulation-blue?style=for-the-badge)
</div>

---

## 📌Summary

This project develops a **Digital Twin** of a warehouse to solve the classic problem of inventory management under uncertainty. Using Discrete Event Simulation, it contrasts a **traditional static strategy (Continuous Review s,Q)** against a **dynamic predictive model driven by Machine Learning (Random Forest)**.

**Business Impact:**
* 📉 **73% Reduction in Operating Costs:** Total costs dropped from \$1,089,102 to \$289,465 annually.
* 🔄 **Just-in-Time Transition:** Logistics efficiency increased by executing 192 data-driven micro-orders.
* ⭐ **100% Service Level:** Zero stockouts, proving that extreme financial efficiency does not have to compromise customer satisfaction.

---

## The Problem: Moving Beyond MAE

In Data Science, achieving a low Mean Absolute Error (MAE) in demand forecasting does not guarantee business profitability if the system fails to manage stockout risks. This project bridges the gap between **Machine Learning** and **Operations Research** by building an asynchronous simulated environment to measure the actual financial impact (Holding Costs, Ordering Costs, and Shortage Costs) of algorithmic decisions.

## Architecture & Methodology

1. **Stochastic Demand Modeling:** Generation of synthetic time series incorporating underlying growth trends, cyclic seasonality, and Gaussian stochastic noise.
2. **Machine Learning Forecasting:** Feature engineering (Lags, Rolling Windows, Calendar data) to train a **Random Forest Regressor** capable of predicting demand during supplier Lead Times.
3. **Simulation Engine (SimPy):** Implementation of a Discrete Event Simulator that processes daily sales, replenishments, transport delays, and accounting concurrently.

---

## 📊 Results & Comparative Analysis

### 1. Total Cost of Operation
The predictive model successfully eliminated the unnecessary "inventory safety buffer," achieving superior economic efficiency.

<img width="870" alt="Total Cost Comparison" src="https://github.com/user-attachments/assets/362be07a-f154-471f-afac-22cb3a52dbfe" />

### 2. Cost Breakdown (KPIs)
The AI drastically reduced holding costs (tied-up capital) while deliberately accepting a marginally higher logistics/ordering cost.

<img width="870" alt="KPIs Comparison" src="https://github.com/user-attachments/assets/a844cc9e-acf9-446d-9c78-9443d2d0ca5d" />

### 3. Order Frequency (Paradigm Shift)
A clear transition towards a *Just-in-Time* philosophy. The AI model fragmented purchases, executing 326% more orders to dynamically adjust to actual demand peaks rather than hoarding stock.

<img width="870" alt="Orders Comparison" src="https://github.com/user-attachments/assets/e07f903a-e815-4494-adca-9aabc6c01ce6" />

---

## 🚀 Conclusions

* **Static models act as expensive insurance:** Designing thresholds based on historical averages leads to massive over-stocking in uncertain environments.
* **The Sweet Spot:** The AI effectively anticipated stochastic variability, drastically reducing immobilized capital without missing a single sale.

---

<div align="center">
  <h1 id="español">🚀 Digital Twin & ML Inventory Optimization</h1>
  <p><b>Simulador de Eventos Discretos para la Optimización de la Cadena de Suministro</b></p>
  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  ![SimPy](https://img.shields.io/badge/SimPy-Simulation-blue?style=for-the-badge)
</div>

---

## 📌 Resumen

Este proyecto desarrolla un **Gemelo Digital (Digital Twin)** de un depósito para resolver el problema clásico de gestión de inventarios bajo incertidumbre. Mediante Simulación de Eventos Discretos, se contrastó una **estrategia tradicional estática (Revisión Continua s,Q)** contra un **modelo dinámico predictivo basado en Machine Learning (Random Forest)**.

**Impacto del Proyecto:**
* 📉 **Reducción del 73% en Costos Operativos:** De \$1,089,102 a \$289,465 USD anuales.
* 🔄 **Transición a Just-in-Time:** Aumento de eficiencia logística realizando 192 micro-pedidos guiados por datos.
* ⭐ **100% Nivel de Servicio:** Cero quiebres de stock, demostrando que la eficiencia no compromete la satisfacción del cliente.

---

## El Problema: Más allá del MAE

En Data Science, predecir la demanda con un bajo Error Absoluto Medio (MAE) no garantiza rentabilidad si el sistema no gestiona adecuadamente el riesgo de *stockout*. Este proyecto puentea la brecha entre el **Machine Learning** y la **Investigación de Operaciones**, construyendo un entorno simulado asíncrono para medir el impacto financiero real (Costos de Mantenimiento, Costos de Ordenar y Costos de Faltante) de las decisiones algorítmicas.

## Arquitectura y Metodología

1. **Modelado de Demanda Estocástica:** Generación de series temporales sintéticas incorporando tendencias de crecimiento, estacionalidad cíclica y ruido estocástico gaussiano.
2. **Machine Learning (Forecasting):** Ingeniería de características (Lags, Rolling Windows, Calendario) y entrenamiento de un modelo **Random Forest Regressor** para predecir la demanda durante el *Lead Time* logístico.
3. **Motor de Simulación (SimPy):** Implementación de un simulador asíncrono de eventos discretos que procesa ventas diarias, despachos, demoras de transporte (Lead Time) y cálculos contables en paralelo.

---

## 📊 Resultados y Análisis Comparativo

### 1. Costo Total de Operación
El modelo predictivo eliminó el "seguro de inventario" innecesario, logrando una eficiencia económica superior.

<img width="870" alt="Comparativa Costo Total" src="https://github.com/user-attachments/assets/362be07a-f154-471f-afac-22cb3a52dbfe" />

### 2. Desglose de KPIs (Costos)
La IA redujo drásticamente el costo de mantenimiento de stock (dinero inmovilizado), asumiendo un costo operativo marginalmente mayor en transporte.

<img width="870" alt="Comparativa KPIs" src="https://github.com/user-attachments/assets/a844cc9e-acf9-446d-9c78-9443d2d0ca5d" />

### 3. Frecuencia de Pedidos (Cambio de Paradigma)
Se evidencia una transición hacia la filosofía *Just-in-Time*. La IA fragmentó las compras realizando 326% más pedidos, ajustándose dinámicamente a los picos reales de demanda.

<img width="870" alt="Comparativa pedidos" src="https://github.com/user-attachments/assets/e07f903a-e815-4494-adca-9aabc6c01ce6" />

---

## 🚀 Conclusiones

* **Los modelos estáticos son ineficientes financieramente:** Diseñar umbrales basados en promedios históricos genera niveles masivos de sobre-stock ante la incertidumbre.
* **Equilibrio Óptimo:** La IA logró anticipar la variabilidad estocástica, reduciendo drásticamente el capital inmovilizado sin perder una sola venta.

---
*Desarrollado como proyecto de Simulación de Sistemas.*
