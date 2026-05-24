----------------------------------
Advertencias de Seguridad Previas:
----------------------------------

-Eludimos cualquier responsabilidad derivada del uso inadecuado y no realizado dentro de la plataforma de AliAmalia; de estas librerias de entrenamiento de modelos de Machine Learning, 
asi como de los usos y funciones a las que los mismos puedan derivar, sin limitación en ello.


----------------------------------


## Estas versiones son librerias descargables avanzadas para la creación y uso en sus framework de entrenamiento; solo para los usuarios avanzados de AliAmalia.

Pueden encontrarse los manuales, para la creación directa de estos Framework por Agentes, como Manus; en este Manual conjunto, de la versión común, y de las avanzadas, que entre otras implementan en entrenamiento la generación directa por Llms de sus múltiples opciones y estas. **[Conocer]**




____________________________________________________________________





# Manual de Adaptadores de Entrenamiento — TenMiNaTor

**Versión:** 3.1.3 (tenIII) / 3.2 (tenIII-II)  
**Autor:** Gotham City AI Lab  
**URL:** http://Terminator.com.es  
**Fecha:** Mayo 2026

---

## Índice

1. [Introducción y filosofía de diseño](#1-introducción-y-filosofía-de-diseño)
2. [Métodos implementados en tenIII v3.1.3](#2-métodos-implementados-en-tenii-v313)
3. [Métodos implementados en tenIII-II v3.2 (32-bit)](#3-métodos-implementados-en-teniii-ii-v32-32-bit)
4. [Guía de decisión por hardware](#4-guía-de-decisión-por-hardware)
5. [Integración con GradientMode](#5-integración-con-gradientmode)
6. [Uso desde un framework externo](#6-uso-desde-un-framework-externo)
7. [Comparativa de eficiencia y calidad](#7-comparativa-de-eficiencia-y-calidad)
8. [Casos de uso avanzados](#8-casos-de-uso-avanzados)
9. [Referencia de API](#9-referencia-de-api)

---

## 1. Introducción y filosofía de diseño

El módulo `training` de TenMiNaTor implementa los métodos de ajuste fino (*fine-tuning*) más relevantes del ecosistema moderno de LLMs. La premisa central es que **el framework de entrenamiento que llama a la librería no debe necesitar conocer los detalles internos del adaptador**: basta con especificar el hardware disponible y la prioridad (memoria, velocidad o calidad), y el sistema selecciona automáticamente el método óptimo.

Todos los adaptadores comparten la misma interfaz `BaseAdapter`:

```python
adapter.apply(model_weights)       # Inyectar parámetros entrenables
adapter.forward_delta(name, x)     # Calcular la corrección ΔW·x
adapter.merge(model_weights)       # Fusionar adaptador con el modelo base
adapter.trainable_parameter_count()
adapter.efficiency_ratio()
adapter.summary()
```

Esta uniformidad permite que cualquier framework externo intercambie adaptadores sin modificar su código de entrenamiento.

---

## 2. Métodos implementados en tenIII v3.1.3

### 2.1 LoRA — Low-Rank Adaptation

**Principio:** Descompone la actualización de pesos ΔW en el producto de dos matrices de bajo rango: ΔW = A·B, donde A ∈ ℝ^(d×r) y B ∈ ℝ^(r×k), con r ≪ min(d, k).

**Parámetros clave:**
- `rank` (r): rango de la descomposición. Valores típicos: 4, 8, 16, 32.
- `alpha`: factor de escala. La actualización efectiva es (alpha/rank)·A·B.
- `target_modules`: lista de capas a adaptar (por defecto, todas las capas de atención).

**Cuándo usarlo:** Es el método de referencia. Funciona bien en la mayoría de los escenarios con GPU de 8 GB o más. Con rank=8 y alpha=16, reduce los parámetros entrenables a menos del 1% del modelo original.

```python
from tenIII.training import AdapterConfig, AdapterMethod, AdapterFactory

cfg = AdapterConfig(method=AdapterMethod.LORA, rank=8, alpha=16)
adapter = AdapterFactory.from_config(cfg)
trainable = adapter.apply(model_weights)
```

---

### 2.2 QLoRA — Quantized LoRA

**Principio:** Combina LoRA con cuantización de 4 bits del modelo base. Los pesos del modelo base se almacenan en NF4 (Normal Float 4-bit), mientras que los adaptadores LoRA se entrenan en BF16/FP32. Esto reduce el consumo de VRAM en un 60-75% respecto a LoRA estándar.

**Parámetros clave:**
- `quant_bits`: bits de cuantización del modelo base (4 por defecto).
- `rank`, `alpha`: igual que LoRA.

**Cuándo usarlo:** GPU con 4-8 GB de VRAM. Es el método **por defecto** en TenMiNaTor cuando no se especifica ninguno, ya que ofrece el mejor equilibrio entre memoria y calidad para la mayoría de los usuarios.

```python
# QLoRA es el método por defecto
cfg = AdapterConfig()  # method=QLORA automáticamente
adapter = AdapterFactory.from_config(cfg)
```

---

### 2.3 DoRA — Weight-Decomposed Low-Rank Adaptation

**Principio:** Descompone el peso W en magnitud (m) y dirección (V): W = m · V/‖V‖. Entrena la magnitud directamente y aplica LoRA a la dirección. Esto imita el comportamiento del ajuste fino completo con mayor fidelidad que LoRA estándar.

**Parámetros clave:**
- `rank`, `alpha`: igual que LoRA.
- `dora_magnitude`: vector de magnitudes (inicializado automáticamente).

**Cuándo usarlo:** Cuando la calidad es prioritaria y se dispone de GPU de 12 GB o más. DoRA supera a LoRA en tareas de instrucción y razonamiento, especialmente con modelos de 7B-13B parámetros.

---

### 2.4 PiSSA — Principal Singular Values and Singular Vectors Adaptation

**Principio:** Inicializa los adaptadores usando los vectores singulares principales de W (SVD). Esto hace que los adaptadores comiencen en la dirección de mayor varianza del modelo, acelerando la convergencia.

**Cuándo usarlo:** Cuando se dispone de pocos pasos de entrenamiento (menos de 1000 steps) y se necesita la mejor calidad posible. Requiere GPU de 12 GB o más para el cálculo SVD inicial.

---

### 2.5 LoRA+ — Differential Learning Rate LoRA

**Principio:** Aplica tasas de aprendizaje diferentes a las matrices A y B del adaptador LoRA. La matriz B recibe una tasa de aprendizaje `lr_ratio` veces mayor que A (por defecto, 16×). Esto acelera la convergencia sin coste adicional de memoria.

**Cuándo usarlo:** Cuando la velocidad de convergencia es prioritaria. Especialmente útil en CPU o GPU de baja potencia donde el número de steps es limitado.

```python
cfg = AdapterConfig(method=AdapterMethod.LORA_PLUS, lora_plus_ratio=16.0)
```

---

### 2.6 VeRA — Vector-based Random Matrix Adaptation

**Principio:** Comparte matrices aleatorias fijas A y B entre todas las capas, y solo entrena vectores de escala (d y b) por capa. Reduce los parámetros entrenables a menos del 0.01% del modelo, con una huella de memoria mínima.

**Cuándo usarlo:** CPU sin GPU, o dispositivos con menos de 2 GB de RAM. Es el método más eficiente en memoria de todos los implementados.

---

### 2.7 AdaLoRA — Adaptive Budget Allocation for LoRA

**Principio:** Distribuye el presupuesto de rango entre las capas de forma adaptativa, asignando mayor rango a las capas más importantes y menor rango a las menos relevantes. Usa SVD truncada para la poda de rango.

**Parámetros clave:**
- `adalora_budget`: presupuesto total de rango distribuido entre todas las capas.

**Cuándo usarlo:** Cuando se quiere maximizar la calidad con un presupuesto de parámetros fijo. Requiere GPU de 16 GB o más por el coste del SVD adaptativo.

---

### 2.8 LoRA-FA — Frozen-A LoRA

**Principio:** Congela la matriz A (inicializada aleatoriamente) y solo entrena B. Esto reduce a la mitad los gradientes que hay que almacenar, con una pérdida de calidad mínima respecto a LoRA estándar.

**Cuándo usarlo:** GPU de 6-8 GB donde LoRA estándar no cabe en memoria. Ofrece un equilibrio entre QLoRA (más eficiente) y LoRA (más calidad).

---

## 3. Métodos implementados en tenIII-II v3.2 (32-bit)

El módulo `teniii_ii.training` está diseñado para dispositivos embebidos sin FPU (Cortex-M0/M0+), con FPU limitada (Cortex-M4/M7), y microcontroladores como ESP32 y RISC-V. Todos los cálculos usan aritmética de punto fijo Q8.8 o Q4.12.

### 3.1 MicroLoRA

**Principio:** LoRA con aritmética de punto fijo Q8.8. Las matrices A y B se almacenan como enteros de 16 bits. El producto A·B se calcula con acumuladores de 32 bits para evitar desbordamiento.

**Requisitos mínimos:** 64 KB de RAM, Cortex-M0+.

```python
from teniii_ii.training import AdapterConfig32, AdapterMethod32, AdapterFactory32

cfg = AdapterConfig32(method=AdapterMethod32.MICRO_LORA, rank=2)
adapter = AdapterFactory32.from_config(cfg)
```

---

### 3.2 BitLoRA

**Principio:** Extiende MicroLoRA con cuantización de 2-4 bits para los pesos del modelo base. Compatible con los formatos Q2 y Q4 del módulo `quantization` de tenIII-II.

**Requisitos mínimos:** 32 KB de RAM, Cortex-M0.

---

### 3.3 TinyVeRA

**Principio:** Versión 32-bit de VeRA. Usa matrices aleatorias fijas en Q4.12 y solo entrena vectores de escala de 8 bits. Huella de memoria: menos de 4 KB para modelos de hasta 1M de parámetros.

**Requisitos mínimos:** 16 KB de RAM. Es el único método viable para Cortex-M0 con menos de 32 KB de RAM.

---

### 3.4 SelectiveLoRA

**Principio:** Aplica LoRA solo a las capas más importantes del modelo, seleccionadas por criterio de magnitud o SVD. Reduce el número de parámetros entrenables en un 50-80% respecto a MicroLoRA completo.

**Cuándo usarlo:** ESP32 o Cortex-M4 con RAM entre 64-256 KB.

---

## 4. Guía de decisión por hardware

La siguiente tabla resume la recomendación automática del sistema (`HardwareDetector.recommend()`). El framework puede usarla como referencia o dejar que el sistema decida automáticamente.

### 4.1 tenIII v3.1.3 — Dispositivos con GPU/CPU estándar

| Hardware | VRAM/RAM | Prioridad memoria | Prioridad velocidad | Prioridad calidad |
|---|---|---|---|---|
| GPU 4 GB | 4 GB VRAM | **QLoRA** (4-bit) | **LoRA-FA** | **QLoRA** |
| GPU 8 GB | 8 GB VRAM | **QLoRA** | **LoRA** | **DoRA** |
| GPU 12 GB | 12 GB VRAM | **LoRA** | **LoRA+** | **DoRA** |
| GPU 24 GB | 24 GB VRAM | **LoRA** | **LoRA+** | **PiSSA** |
| GPU 40 GB+ | 40+ GB VRAM | **LoRA** | **AdaLoRA** | **AdaLoRA** |
| Multi-GPU | 80+ GB VRAM | **LoRA** | **AdaLoRA** | **AdaLoRA** |
| CPU only | Sistema RAM | **VeRA** | **LoRA+** | **PiSSA** |

**Regla general para GPU:** Con menos de 8 GB de VRAM, QLoRA es casi siempre la mejor opción. Con 12-24 GB, DoRA ofrece la mejor calidad. Con 40 GB o más, AdaLoRA maximiza el uso del presupuesto de parámetros.

**Regla general para CPU:** VeRA para memoria, LoRA+ para velocidad, PiSSA para calidad con pocos steps.

### 4.2 tenIII-II v3.2 — Dispositivos embebidos 32-bit

| Dispositivo | RAM disponible | Recomendación |
|---|---|---|
| Cortex-M0 / M0+ | < 16 KB | **TinyVeRA** (único viable) |
| Cortex-M0 / M0+ | 16-64 KB | **BitLoRA** (rank=1) |
| Cortex-M4 / M7 | 64-256 KB | **MicroLoRA** (rank=2-4) |
| ESP32 | 256-512 KB | **SelectiveLoRA** (rank=4) |
| RISC-V (HiFive) | 512 KB+ | **MicroLoRA** (rank=4-8) |
| Raspberry Pi Zero | 512 MB | **MicroLoRA** (rank=8) |

**Nota sobre FPU:** Los dispositivos Cortex-M0/M0+ no tienen FPU. TenMiNaTor usa aritmética de punto fijo Q8.8 automáticamente cuando detecta `DeviceClass.CORTEX_M0`. Los dispositivos Cortex-M4/M7 con FPU pueden usar Q8.8 o FP32 según la configuración.

### 4.3 Árbol de decisión rápido

```
¿Tienes GPU?
├── Sí
│   ├── VRAM < 8 GB → QLoRA (por defecto)
│   ├── VRAM 8-16 GB
│   │   ├── Prioridad calidad → DoRA
│   │   ├── Prioridad velocidad → LoRA+
│   │   └── Prioridad memoria → LoRA-FA
│   └── VRAM > 16 GB
│       ├── Prioridad calidad → AdaLoRA o PiSSA
│       └── Prioridad velocidad → LoRA+
└── No (CPU / embebido)
    ├── RAM > 4 GB → LoRA+ o PiSSA
    ├── RAM 256 MB - 4 GB → LoRA o VeRA
    ├── RAM 64-256 KB → MicroLoRA o SelectiveLoRA (tenIII-II)
    ├── RAM 16-64 KB → BitLoRA (tenIII-II)
    └── RAM < 16 KB → TinyVeRA (tenIII-II)
```

---

## 5. Integración con GradientMode

Todos los adaptadores de TenMiNaTor se integran con los 3 modos de `GradientMode` implementados en `SimulationGRPOModel`. Esta integración se gestiona a través de `AdaptedTrainer` (tenIII) y `AdaptedTrainer32` (tenIII-II).

### 5.1 Los 3 modos y su efecto sobre el fine-tuning

| Modo | Comportamiento en `zero_grad` | Caso de uso con adaptadores |
|---|---|---|
| `RESET` | Reinicia gradientes a cero | Entrenamiento estándar por epochs. Compatible con todos los adaptadores. |
| `ACCUMULATE` | No reinicia (acumula entre pasos) | Gradient accumulation para batches pequeños. Útil con QLoRA en GPU 4 GB. |
| `CONTINUOUS_DESCENT` | Aplica decaimiento exponencial (×0.9) | Entrenamiento continuo sin reinicio. Ideal para fine-tuning incremental con LoRA. |

### 5.2 Combinaciones recomendadas

La siguiente tabla muestra las combinaciones más efectivas entre adaptador y GradientMode:

| Adaptador | GradientMode recomendado | Razón |
|---|---|---|
| QLoRA | ACCUMULATE | Los gradientes de 4-bit se benefician de la acumulación |
| LoRA | RESET | Comportamiento estándar, máxima estabilidad |
| DoRA | RESET | La actualización de magnitud requiere gradientes limpios |
| PiSSA | RESET | La inicialización SVD ya proporciona el momentum inicial |
| LoRA+ | CONTINUOUS_DESCENT | El ratio diferencial de LR se combina bien con el decaimiento |
| VeRA | ACCUMULATE | Los vectores de escala son muy pequeños; la acumulación ayuda |
| AdaLoRA | RESET | La poda adaptativa requiere gradientes precisos por paso |
| MicroLoRA | ACCUMULATE | En MCU, los batches son de 1-4 muestras; la acumulación es necesaria |

### 5.3 Configuración desde un framework externo

```python
from tenIII.training import AdaptedTrainer, AdaptedTrainingConfig, AdapterMethod

# Configuración completa
cfg = AdaptedTrainingConfig(
    # Adaptador
    auto_select_adapter=False,          # Selección manual
    adapter_config=AdapterConfig(
        method=AdapterMethod.QLORA,
        rank=8,
        alpha=16,
        quant_bits=4,
    ),
    # Hiperparámetros
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,     # Acumular 8 pasos antes de actualizar
    # GradientMode
    gradient_mode="accumulate",        # Acumulación para QLoRA
    gradient_decay_factor=0.9,         # Solo relevante para continuous_descent
    # Fusión al finalizar
    merge_on_finish=True,
)

trainer = AdaptedTrainer(cfg)
trainer.setup(model_weights)
metrics = trainer.train(train_data)
print(f"Adaptador usado: {metrics['adapter_method']}")
print(f"Parámetros entrenables: {trainer.adapter.trainable_parameter_count():,}")
```

---

## 6. Uso desde un framework externo

TenMiNaTor está diseñado para ser llamado desde cualquier framework de entrenamiento. La API es estable y no requiere conocer los detalles internos.

### 6.1 Selección automática (recomendada)

```python
from tenIII.training import AdaptedTrainer, AdaptedTrainingConfig

# El sistema detecta el hardware y selecciona el mejor adaptador
cfg = AdaptedTrainingConfig(
    auto_select_adapter=True,
    model_size_b=7.0,          # Tamaño del modelo en miles de millones de parámetros
    hardware_priority="memory", # "memory" | "speed" | "quality" | "balanced"
)
trainer = AdaptedTrainer(cfg)
trainer.setup(model_weights)
```

### 6.2 Selección manual con lista de métodos disponibles

```python
from tenIII.training import AdapterFactory, AdapterMethod

# Ver todos los métodos disponibles
for method_info in AdapterFactory.list_methods():
    print(f"{method_info['method']}: {method_info['description']}")
    print(f"  VRAM mínima: {method_info['min_vram_gb']} GB")
    print(f"  Parámetros: {method_info['typical_params_pct']}%")
```

### 6.3 Integración con PyTorch (framework externo)

Cuando el framework usa PyTorch real, `AdaptedTrainer` actúa como gestor de adaptadores. El framework llama a `apply()` para obtener los parámetros entrenables y los pasa al optimizador de PyTorch:

```python
import torch
from tenIII.training import AdapterFactory, AdapterConfig, AdapterMethod

# Crear adaptador
cfg = AdapterConfig(method=AdapterMethod.LORA, rank=8, alpha=16)
adapter = AdapterFactory.from_config(cfg)

# Aplicar al modelo (obtener parámetros numpy)
trainable_np = adapter.apply(model_weights_np)

# Convertir a tensores PyTorch con requires_grad=True
trainable_torch = {
    k: torch.tensor(v, requires_grad=True)
    for k, v in trainable_np.items()
}

# Optimizador PyTorch solo sobre los parámetros del adaptador
optimizer = torch.optim.AdamW(
    list(trainable_torch.values()),
    lr=2e-4,
    weight_decay=0.01,
)

# Ciclo de entrenamiento estándar
for batch in dataloader:
    loss = model_forward_with_adapter(batch, trainable_torch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Al finalizar: fusionar adaptador con el modelo base
merged_weights = adapter.merge(model_weights_np)
```

### 6.4 Integración con tenIII-II en dispositivos embebidos

```python
from teniii_ii.training import AdaptedTrainer32, AdaptedTrainingConfig32

cfg = AdaptedTrainingConfig32(
    auto_select=True,
    ram_kb=128,               # RAM disponible en KB
    has_fpu=True,             # Cortex-M4 con FPU
    gradient_mode="accumulate",
    batch_size=1,             # En MCU, batch_size=1 es lo habitual
    gradient_accumulation_steps=16,
)
trainer = AdaptedTrainer32(cfg)
trainer.setup(model_weights_int16)
```

---

## 7. Comparativa de eficiencia y calidad

La siguiente tabla resume las características de cada método para ayudar en la decisión:

### 7.1 tenIII v3.1.3

| Método | Parámetros entrenables | VRAM mínima | Velocidad convergencia | Calidad relativa |
|---|---|---|---|---|
| LoRA | ~0.5-1% | 8 GB | Media | ★★★★☆ |
| **QLoRA** | ~0.5-1% | **4 GB** | Media | ★★★★☆ |
| DoRA | ~0.5-1% | 12 GB | Media-lenta | ★★★★★ |
| PiSSA | ~0.5-1% | 12 GB | **Rápida** | ★★★★★ |
| LoRA+ | ~0.5-1% | 8 GB | **Muy rápida** | ★★★★☆ |
| VeRA | **~0.01%** | CPU | Lenta | ★★★☆☆ |
| AdaLoRA | ~0.5-1% | 16 GB | Lenta | ★★★★★ |
| LoRA-FA | ~0.25-0.5% | 6 GB | Media | ★★★★☆ |

> **Nota:** "Calidad relativa" se refiere a la calidad del ajuste fino respecto al fine-tuning completo (full fine-tuning), medida en benchmarks de instrucción y razonamiento.

### 7.2 tenIII-II v3.2 (embebidos)

| Método | RAM mínima | Precisión | Velocidad inferencia | Calidad relativa |
|---|---|---|---|---|
| MicroLoRA | 64 KB | Q8.8 | Media | ★★★☆☆ |
| BitLoRA | 32 KB | Q4.12 | Rápida | ★★☆☆☆ |
| TinyVeRA | **16 KB** | Q4.12 | **Muy rápida** | ★★☆☆☆ |
| SelectiveLoRA | 64 KB | Q8.8 | Media | ★★★☆☆ |

---

## 8. Casos de uso avanzados

### 8.1 Fine-tuning incremental con CONTINUOUS_DESCENT

El modo `CONTINUOUS_DESCENT` permite reanudar el entrenamiento desde el punto exacto donde se detuvo, conservando el momentum de los gradientes anteriores. Esto es especialmente útil para el fine-tuning incremental donde se añaden nuevos datos periódicamente:

```python
# Primera sesión de entrenamiento
cfg = AdaptedTrainingConfig(
    auto_select_adapter=True,
    gradient_mode="continuous_descent",
    gradient_decay_factor=0.95,  # Conservar 95% del momentum anterior
    merge_on_finish=False,        # No fusionar: queremos continuar entrenando
)
trainer = AdaptedTrainer(cfg)
trainer.setup(model_weights)
trainer.train(batch_1)

# Guardar estado del adaptador
adapter_state = trainer.get_trainable_params()
gradient_state = trainer._gradients

# Segunda sesión (nuevos datos)
trainer2 = AdaptedTrainer(cfg)
trainer2.setup(model_weights)
trainer2._trainable_params = adapter_state
trainer2._gradients = gradient_state  # Restaurar gradientes con momentum
trainer2.train(batch_2)               # Continúa desde donde se dejó
```

### 8.2 Combinación QLoRA + ACCUMULATE para GPU 4 GB

Con una GPU de 4 GB, el batch_size efectivo suele ser 1-2. La acumulación de gradientes permite simular batches más grandes:

```python
cfg = AdaptedTrainingConfig(
    auto_select_adapter=False,
    adapter_config=AdapterConfig(
        method=AdapterMethod.QLORA,
        rank=4,
        alpha=8,
        quant_bits=4,
    ),
    batch_size=1,
    gradient_accumulation_steps=32,  # Simula batch_size=32
    gradient_mode="accumulate",
    learning_rate=1e-4,
)
```

### 8.3 AdaLoRA para máxima calidad con presupuesto fijo

```python
from tenIII.training import AdapterConfig, AdapterMethod

cfg_adalora = AdapterConfig(
    method=AdapterMethod.ADALORA,
    rank=8,
    alpha=16,
    adalora_budget=144,  # Presupuesto total de rango para todas las capas
)
# AdaLoRA distribuirá el presupuesto de 144 entre las capas más importantes
adapter = AdapterFactory.from_config(cfg_adalora)
```

### 8.4 Selección automática con detección de hardware

```python
from tenIII.training import HardwareDetector, HardwareProfile

# Detectar hardware automáticamente
profile = HardwareDetector.detect()
print(f"Hardware detectado: {profile.value}")

# Obtener recomendación
method, reason = HardwareDetector.recommend(
    profile=profile,
    model_size_b=7.0,
    priority="balanced",
)
print(f"Método recomendado: {method.value}")
print(f"Razón: {reason}")
```

### 8.5 Fine-tuning en Cortex-M4 con tenIII-II

```python
from teniii_ii.training import (
    AdaptedTrainer32, AdaptedTrainingConfig32,
    AdapterMethod32, AdapterConfig32
)

# Cortex-M4 con FPU y 256 KB de RAM
cfg = AdaptedTrainingConfig32(
    auto_select=False,
    adapter_config=AdapterConfig32(
        method=AdapterMethod32.MICRO_LORA,
        rank=4,
        fixed_point_bits=8,  # Q8.8
    ),
    ram_kb=256,
    has_fpu=True,
    batch_size=2,
    gradient_accumulation_steps=8,
    gradient_mode="accumulate",
    learning_rate=1e-3,  # LR más alto en MCU por menor número de steps
)

trainer = AdaptedTrainer32(cfg)
trainer.setup(model_weights_int16)
metrics = trainer.train(micro_dataset)
```

---

## 9. Referencia de API

### 9.1 `AdapterConfig` (tenIII v3.1.3)

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `method` | `AdapterMethod` | `QLORA` | Método de adaptación |
| `rank` | `int` | `8` | Rango de la descomposición |
| `alpha` | `float` | `16.0` | Factor de escala (alpha/rank) |
| `target_modules` | `List[str]` | `["q_proj","v_proj"]` | Capas a adaptar |
| `dropout` | `float` | `0.05` | Dropout en el adaptador |
| `quant_bits` | `int` | `4` | Bits de cuantización (QLoRA) |
| `lora_plus_ratio` | `float` | `16.0` | Ratio LR B/A (LoRA+) |
| `adalora_budget` | `int` | `144` | Presupuesto total de rango (AdaLoRA) |

### 9.2 `AdapterMethod` (tenIII v3.1.3)

| Valor | Descripción |
|---|---|
| `LORA` | LoRA estándar |
| `QLORA` | LoRA con cuantización 4-bit (por defecto) |
| `DORA` | Weight-Decomposed LoRA |
| `PISSA` | Principal Singular Values Adaptation |
| `LORA_PLUS` | LoRA con ratio diferencial de LR |
| `VERA` | Vector-based Random Matrix Adaptation |
| `ADALORA` | Adaptive Budget Allocation LoRA |
| `LORA_FA` | Frozen-A LoRA |

### 9.3 `AdapterMethod32` (tenIII-II v3.2)

| Valor | Descripción | RAM mínima |
|---|---|---|
| `MICRO_LORA` | LoRA en punto fijo Q8.8 | 64 KB |
| `BIT_LORA` | LoRA con cuantización 2-4 bits | 32 KB |
| `TINY_VERA` | VeRA ultra-ligero Q4.12 | 16 KB |
| `SELECTIVE_LORA` | LoRA selectivo por importancia | 64 KB |

### 9.4 `HardwareProfile` (tenIII v3.1.3)

| Valor | Descripción |
|---|---|
| `GPU_4GB` | GPU con 4 GB de VRAM |
| `GPU_8GB` | GPU con 8 GB de VRAM |
| `GPU_12GB` | GPU con 12 GB de VRAM |
| `GPU_24GB` | GPU con 24 GB de VRAM |
| `GPU_40GB_PLUS` | GPU con 40+ GB de VRAM |
| `MULTI_GPU` | Configuración multi-GPU |
| `CPU_ONLY` | Sin GPU, solo CPU |

### 9.5 `DeviceClass32` (tenIII-II v3.2)

| Valor | Descripción |
|---|---|
| `CORTEX_M0` | ARM Cortex-M0/M0+, sin FPU |
| `CORTEX_M4` | ARM Cortex-M4/M7, con FPU |
| `ESP32` | ESP32 / ESP32-S3 |
| `RISCV_32` | RISC-V 32-bit |
| `GENERIC_32BIT` | Cualquier dispositivo 32-bit genérico |

---

*Manual generado por Gotham City AI Lab — TenMiNaTor v3.1.3 / tenIII-II v3.2*  
*http://Terminator.com.es*
