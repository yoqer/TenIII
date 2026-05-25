# Manual Unificado de Framework de Entrenamiento — TenMiNaTor

**Versión:** tenIII v3.1.3 / tenIII-II v3.2  
**Autor:** Gotham City AI Lab  
**URL:** http://Terminator.com.es  
**Fecha:** Mayo 2026

---

> Este manual unifica en un único documento tres guías complementarias: la guía base de creación de frameworks de entrenamiento (válida para todos los usuarios), la guía avanzada de adaptadores LLM/QLoRA (exclusiva para tenIII v3.1.3 y tenIII-II v3.2), y la guía de creación de frameworks de entrenamiento usando tenIII desde PyPI.

---

## Índice

**Parte 1 — Framework Base (Todos los usuarios)**

1. [Arquitectura del sistema de entrenamiento](#1-arquitectura-del-sistema-de-entrenamiento)
2. [GradientMode: los tres modos de gradiente](#2-gradientmode-los-tres-modos-de-gradiente)
3. [Construcción de un framework de entrenamiento básico](#3-construcción-de-un-framework-de-entrenamiento-básico)
4. [Integración con TerminaTodo (almacenamiento)](#4-integración-con-terminatodo-almacenamiento)
5. [Integración con Terminator AliAmalia (agente)](#5-integración-con-terminator-aliamalia-agente)

**Parte 2 — Adaptadores LLM Avanzados (SOLO tenIII v3.1.3 y tenIII-II v3.2)**

6. [Introducción a los adaptadores de fine-tuning](#6-introducción-a-los-adaptadores-de-fine-tuning)
7. [Adaptadores en tenIII v3.1.3 (GPU/CPU estándar)](#7-adaptadores-en-tenii-v313-gpucpu-estándar)
8. [Adaptadores en tenIII-II v3.2 (dispositivos embebidos 32-bit)](#8-adaptadores-en-teniii-ii-v32-dispositivos-embebidos-32-bit)
9. [Guía de decisión por hardware](#9-guía-de-decisión-por-hardware)
10. [Integración de adaptadores con GradientMode](#10-integración-de-adaptadores-con-gradientmode)
11. [Casos de uso avanzados](#11-casos-de-uso-avanzados)
12. [Referencia de API de adaptadores](#12-referencia-de-api-de-adaptadores)

**Parte 3 — Creación de Frameworks con tenIII desde PyPI (SOLO tenIII PyPI)**

13. [Instalación y estructura del proyecto](#13-instalación-y-estructura-del-proyecto)
14. [Subclasificación de AdaptedTrainer](#14-subclasificación-de-adaptedtrainer)
15. [Configuración de AdapterConfig y GradientMode](#15-configuración-de-adapterconfig-y-gradientmode)
16. [Exportación del modelo entrenado](#16-exportación-del-modelo-entrenado)
17. [Ejemplo mínimo funcional completo](#17-ejemplo-mínimo-funcional-completo)

---

# Parte 1 — Framework Base

> Esta parte es válida para todos los usuarios del ecosistema TenMiNaTor, independientemente del hardware o la versión instalada.

---

## 1. Arquitectura del sistema de entrenamiento

El ecosistema TenMiNaTor está compuesto por cuatro paquetes con responsabilidades bien delimitadas. Comprender la arquitectura general es el primer paso para construir cualquier framework de entrenamiento sobre él.

| Paquete | Versión | Responsabilidad principal |
|---|---|---|
| **tenIII** | v3.1.3 | Framework LLM completo: entrenamiento, razonamiento, cuantización, exportación |
| **tenIII-II** | v3.2 | Inferencia ultra-ligera para dispositivos 32-bit y MCU embebidos |
| **TerminaTodo** | v1.0.0 | Descubrimiento y gestión de backends de almacenamiento (USB, S3, FTP, Cloud) |
| **Terminator AliAmalia** | v1.0.0 | Framework de agentes autónomos (*Exclusivo AliAmalia*) |

La separación de responsabilidades permite que un framework de entrenamiento externo orqueste los cuatro paquetes de forma independiente. El framework llama a tenIII para el entrenamiento, a TerminaTodo para guardar los pesos, y opcionalmente a Terminator para automatizar decisiones durante el proceso.

El punto de entrada central para el entrenamiento es el módulo `tenIII.reasoning`, que contiene `SimulationGRPOModel`. Este modelo implementa el algoritmo GRPO (*Group Relative Policy Optimization*) con soporte para tres backends de cálculo (PyTorch, TensorFlow y Simulación) y tres modos de gestión de gradientes (`GradientMode`).

```
tenIII/
├── core/           ← Arquitectura base del modelo
├── reasoning/      ← GRPO + GradientMode (punto de entrada principal)
│   └── grpo_backprop.py
├── training/       ← Adaptadores de fine-tuning (Parte 2 de este manual)
│   ├── adapters.py
│   └── adapter_trainer.py
├── constitution/   ← Constitutional AI
├── quantization/   ← Q1-Q8, TurboQuant, ANT4, NVFP4
├── storage/        ← Integración con TerminaTodo
├── multimodal/     ← Texto, imagen, audio, video
└── packaging/      ← Exportación USB, GGUF, chip, unikernel
```

---

## 2. GradientMode: los tres modos de gradiente

`GradientMode` es el mecanismo central que controla cómo se gestionan los gradientes entre pasos de entrenamiento. Está implementado en `tenIII.reasoning.grpo_backprop` y afecta directamente al comportamiento de `SimulationGRPOModel.zero_grad()`.

### 2.1 Los tres modos

| Modo | Comportamiento en `zero_grad()` | Caso de uso principal |
|---|---|---|
| `RESET` | Reinicia gradientes a cero en cada paso | Entrenamiento estándar por epochs; máxima estabilidad |
| `ACCUMULATE` | No reinicia; acumula gradientes entre pasos | Gradient accumulation para batches pequeños o GPU con poca VRAM |
| `CONTINUOUS_DESCENT` | Aplica decaimiento exponencial (×`decay_factor`, por defecto 0.9) | Entrenamiento continuo e incremental sin reinicio completo |

### 2.2 Importación y uso básico

```python
from tenIII.reasoning.grpo_backprop import SimulationGRPOModel, GradientMode

# Modo RESET (comportamiento clásico de PyTorch)
model_reset = SimulationGRPOModel(
    gradient_mode=GradientMode.RESET
)

# Modo ACCUMULATE (acumulación de gradientes)
model_accum = SimulationGRPOModel(
    gradient_mode=GradientMode.ACCUMULATE
)

# Modo CONTINUOUS_DESCENT (decaimiento exponencial)
model_cont = SimulationGRPOModel(
    gradient_mode=GradientMode.CONTINUOUS_DESCENT,
    decay_factor=0.9  # Conserva el 90% del momentum anterior
)
```

### 2.3 Comportamiento interno de `zero_grad()`

El método `zero_grad()` de `SimulationGRPOModel` se comporta de forma diferente según el modo activo:

```python
def zero_grad(self):
    if self.gradient_mode == GradientMode.RESET:
        # Reinicio completo: gradientes = 0
        self._gradients = {k: np.zeros_like(v) for k, v in self._weights.items()}

    elif self.gradient_mode == GradientMode.ACCUMULATE:
        # No hace nada: los gradientes se acumulan entre llamadas
        pass

    elif self.gradient_mode == GradientMode.CONTINUOUS_DESCENT:
        # Decaimiento exponencial: gradientes *= decay_factor
        self._gradients = {
            k: v * self.decay_factor
            for k, v in self._gradients.items()
        }
```

### 2.4 Cuándo usar cada modo

**RESET** es el modo por defecto y el más seguro. Debe usarse cuando se entrena con batches de tamaño normal (≥16 muestras) y cuando la estabilidad del entrenamiento es prioritaria. Es el equivalente directo de `optimizer.zero_grad()` en PyTorch.

**ACCUMULATE** debe usarse cuando el hardware no permite batches grandes. Con una GPU de 4 GB y un modelo de 7B parámetros, el batch_size efectivo puede ser 1 o 2. Acumular gradientes durante 8-32 pasos antes de actualizar los pesos simula un batch_size de 8-64, mejorando la estabilidad del entrenamiento sin aumentar el consumo de memoria.

**CONTINUOUS_DESCENT** es el modo más experimental. Su principal ventaja es que permite reanudar el entrenamiento desde el punto exacto donde se detuvo, conservando el momentum de los gradientes anteriores. Es ideal para el fine-tuning incremental donde se añaden nuevos datos periódicamente, o para el entrenamiento continuo en dispositivos embebidos donde el proceso puede interrumpirse en cualquier momento.

---

## 3. Construcción de un framework de entrenamiento básico

Esta sección describe cómo construir un framework de entrenamiento mínimo usando `SimulationGRPOModel` como motor de gradientes. El framework es independiente del hardware y no requiere GPU.

### 3.1 Estructura mínima del framework

Un framework de entrenamiento sobre TenMiNaTor tiene cuatro componentes:

1. **Inicialización del modelo** con los pesos y el modo de gradiente.
2. **Ciclo de entrenamiento** que llama a `forward()`, calcula la pérdida, y llama a `backward()`.
3. **Gestión de gradientes** mediante `zero_grad()` y `step()`.
4. **Guardado del estado** al finalizar o en puntos de control.

```python
import numpy as np
from tenIII.reasoning.grpo_backprop import SimulationGRPOModel, GradientMode

class MiFrameworkEntrenamiento:
    """Framework de entrenamiento mínimo sobre TenMiNaTor."""

    def __init__(self, model_weights: dict, gradient_mode: GradientMode = GradientMode.RESET,
                 learning_rate: float = 1e-4, decay_factor: float = 0.9):
        self.model = SimulationGRPOModel(
            gradient_mode=gradient_mode,
            decay_factor=decay_factor,
        )
        self.model.initialize_weights(model_weights)
        self.lr = learning_rate
        self.history = []

    def train_step(self, batch: list) -> float:
        """Ejecuta un paso de entrenamiento y devuelve la pérdida."""
        # 1. Forward pass
        output = self.model.forward(batch)

        # 2. Calcular pérdida (MSE simple como ejemplo)
        target = np.ones_like(output) * 0.5
        loss = float(np.mean((output - target) ** 2))

        # 3. Backward pass (calcula gradientes)
        self.model.backward(loss)

        # 4. Actualizar pesos con los gradientes
        self.model.step(self.lr)

        # 5. Gestionar gradientes según el modo
        self.model.zero_grad()

        return loss

    def train(self, dataset: list, num_epochs: int = 3) -> dict:
        """Entrena el modelo durante num_epochs epochs."""
        for epoch in range(num_epochs):
            epoch_losses = []
            for batch in dataset:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
            avg_loss = float(np.mean(epoch_losses))
            self.history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{num_epochs} — Pérdida: {avg_loss:.6f}")

        return {
            "final_loss": self.history[-1],
            "history": self.history,
            "weights": self.model.get_weights(),
        }

    def get_weights(self) -> dict:
        """Devuelve los pesos actuales del modelo."""
        return self.model.get_weights()
```

### 3.2 Uso del framework

```python
# Pesos iniciales del modelo (ejemplo con 3 capas)
model_weights = {
    "layer_0": np.random.randn(64, 64).astype(np.float32) * 0.01,
    "layer_1": np.random.randn(64, 32).astype(np.float32) * 0.01,
    "layer_2": np.random.randn(32, 1).astype(np.float32) * 0.01,
}

# Dataset de ejemplo (listas de arrays numpy)
dataset = [
    [np.random.randn(8, 64).astype(np.float32) for _ in range(4)]
    for _ in range(10)  # 10 batches por epoch
]

# Crear y entrenar el framework
framework = MiFrameworkEntrenamiento(
    model_weights=model_weights,
    gradient_mode=GradientMode.ACCUMULATE,
    learning_rate=1e-4,
)

results = framework.train(dataset, num_epochs=5)
print(f"Pérdida final: {results['final_loss']:.6f}")
```

### 3.3 Framework con gradient accumulation manual

Cuando se usa `GradientMode.ACCUMULATE`, el framework debe controlar cuándo actualizar los pesos. El patrón estándar es acumular durante `N` pasos y luego llamar a `step()`:

```python
class FrameworkConAcumulacion:
    def __init__(self, model_weights, accumulation_steps=8, lr=1e-4):
        self.model = SimulationGRPOModel(gradient_mode=GradientMode.ACCUMULATE)
        self.model.initialize_weights(model_weights)
        self.accumulation_steps = accumulation_steps
        self.lr = lr
        self._step_count = 0

    def train_step(self, batch) -> float:
        output = self.model.forward(batch)
        loss = float(np.mean((output - 0.5) ** 2))
        self.model.backward(loss)
        self._step_count += 1

        # Actualizar pesos solo cada N pasos
        if self._step_count % self.accumulation_steps == 0:
            self.model.step(self.lr)
            self.model.zero_grad()  # En ACCUMULATE, esto no hace nada,
                                    # pero es buena práctica llamarlo

        return loss
```

---

## 4. Integración con TerminaTodo (almacenamiento)

TerminaTodo v1.0.0 proporciona un sistema de descubrimiento automático de backends de almacenamiento. Permite guardar y recuperar los pesos del modelo en USB, S3, FTP o almacenamiento local sin cambiar el código del framework.

### 4.1 Descubrimiento automático de backend

```python
from terminatodo import select_best_backend, list_backends

# Ver todos los backends disponibles
backends = list_backends()
for name, available in backends.items():
    print(f"  {name}: {'✓' if available else '✗'}")

# Seleccionar automáticamente el mejor backend disponible
backend = select_best_backend()
print(f"Backend seleccionado: {backend.name}")
```

### 4.2 Guardado de pesos del modelo

```python
import pickle
from terminatodo import select_best_backend

def guardar_modelo(weights: dict, nombre: str = "modelo_entrenado") -> str:
    """Guarda los pesos del modelo en el mejor backend disponible."""
    backend = select_best_backend()

    # Serializar pesos
    data = pickle.dumps(weights)

    # Guardar en el backend
    ruta = f"{nombre}.pkl"
    backend.save(ruta, data)

    print(f"Modelo guardado en {backend.name}: {ruta}")
    return ruta

def cargar_modelo(nombre: str = "modelo_entrenado") -> dict:
    """Carga los pesos del modelo desde el backend disponible."""
    backend = select_best_backend()
    ruta = f"{nombre}.pkl"
    data = backend.load(ruta)
    return pickle.loads(data)
```

### 4.3 Guardado en USB (despliegue portátil)

TerminaTodo detecta automáticamente dispositivos USB montados. Si hay un USB disponible, `select_best_backend()` lo priorizará sobre el almacenamiento local:

```python
from terminatodo.backends.usb import USBBackend

# Verificar si hay USB disponible
usb = USBBackend()
if usb.is_available():
    print(f"USB detectado: {usb.name}")
    usb.save("modelo_final.pkl", pickle.dumps(weights))
else:
    print("No hay USB disponible, usando almacenamiento local")
```

### 4.4 Integración en el ciclo de entrenamiento

```python
from terminatodo import select_best_backend
import pickle

class FrameworkConAlmacenamiento:
    def __init__(self, model_weights, checkpoint_interval=100):
        self.model = SimulationGRPOModel(gradient_mode=GradientMode.RESET)
        self.model.initialize_weights(model_weights)
        self.storage = select_best_backend()
        self.checkpoint_interval = checkpoint_interval
        self._global_step = 0

    def train_step(self, batch) -> float:
        output = self.model.forward(batch)
        loss = float(np.mean((output - 0.5) ** 2))
        self.model.backward(loss)
        self.model.step(lr=1e-4)
        self.model.zero_grad()
        self._global_step += 1

        # Guardar checkpoint periódicamente
        if self._global_step % self.checkpoint_interval == 0:
            self._save_checkpoint()

        return loss

    def _save_checkpoint(self):
        state = {
            "step": self._global_step,
            "weights": self.model.get_weights(),
            "gradient_mode": self.model.gradient_mode.value,
        }
        data = pickle.dumps(state)
        self.storage.save(f"checkpoint_step_{self._global_step}.pkl", data)
        print(f"Checkpoint guardado en paso {self._global_step} ({self.storage.name})")
```

---

## 5. Integración con Terminator AliAmalia (agente)

> **Nota:** Terminator AliAmalia es un producto *Exclusivo AliAmalia*. Esta sección describe la integración del agente con el framework de entrenamiento para automatizar decisiones durante el proceso.

Terminator AliAmalia v1.0.0 proporciona un framework de agentes autónomos que puede integrarse con el ciclo de entrenamiento para automatizar decisiones como: ajustar el learning rate, cambiar el modo de gradiente, o detener el entrenamiento cuando se detecta divergencia.

### 5.1 Inicialización del agente

```python
from terminator_aliamalia import Terminator, PersonalityProfile

# Crear agente con perfil analítico
agente = Terminator(personality=PersonalityProfile.ANALYTICAL)
```

### 5.2 Uso del agente para analizar métricas de entrenamiento

```python
# Analizar el historial de pérdidas y obtener un plan de acción
historial_perdidas = [0.85, 0.72, 0.68, 0.71, 0.73, 0.75]  # Divergencia detectada

analisis = agente.analyze(
    f"El historial de pérdidas del entrenamiento es: {historial_perdidas}. "
    f"La pérdida aumentó en los últimos 3 pasos. ¿Qué acción recomendar?"
)

plan = agente.plan(analisis)
print(f"Plan del agente: {plan}")
# Ejemplo de salida: "Reducir learning rate a 5e-5 y cambiar a GradientMode.ACCUMULATE"
```

### 5.3 Integración completa en el framework

```python
from terminator_aliamalia import Terminator, PersonalityProfile

class FrameworkConAgente:
    def __init__(self, model_weights, lr=1e-4):
        self.model = SimulationGRPOModel(gradient_mode=GradientMode.RESET)
        self.model.initialize_weights(model_weights)
        self.agente = Terminator(personality=PersonalityProfile.ANALYTICAL)
        self.lr = lr
        self.history = []

    def _evaluar_con_agente(self) -> str:
        """Pide al agente que evalúe el estado del entrenamiento."""
        if len(self.history) < 5:
            return "continuar"

        ultimas = self.history[-5:]
        contexto = (
            f"Últimas 5 pérdidas: {ultimas}. "
            f"Learning rate actual: {self.lr}. "
            f"Modo de gradiente: {self.model.gradient_mode.value}. "
            "¿Debo continuar, reducir LR, cambiar modo o detener?"
        )
        analisis = self.agente.analyze(contexto)
        return self.agente.plan(analisis)

    def train(self, dataset, num_epochs=5):
        for epoch in range(num_epochs):
            epoch_losses = []
            for batch in dataset:
                output = self.model.forward(batch)
                loss = float(np.mean((output - 0.5) ** 2))
                self.model.backward(loss)
                self.model.step(self.lr)
                self.model.zero_grad()
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            self.history.append(avg_loss)

            # Consultar al agente cada epoch
            decision = self._evaluar_con_agente()
            print(f"Epoch {epoch+1}: pérdida={avg_loss:.6f} | Agente: {decision}")

            # Almacenar decisión del agente
            self.agente.store("ultima_decision", decision)
```

---

# Parte 2 — Adaptadores LLM Avanzados

> ⚠️ **SOLO USUARIOS AVANZADOS** — Esta parte requiere tenIII v3.1.3 (GPU/CPU estándar) o tenIII-II v3.2 (dispositivos embebidos 32-bit). No es aplicable a versiones anteriores ni a otros paquetes del ecosistema.

---

## 6. Introducción a los adaptadores de fine-tuning

El módulo `tenIII.training` implementa los métodos de ajuste fino (*fine-tuning*) más relevantes del ecosistema moderno de LLMs. La premisa central es que **el framework de entrenamiento que llama a la librería no necesita conocer los detalles internos del adaptador**: basta con especificar el hardware disponible y la prioridad (memoria, velocidad o calidad), y el sistema selecciona automáticamente el método óptimo.

Todos los adaptadores comparten la misma interfaz `BaseAdapter`:

```python
adapter.apply(model_weights)           # Inyectar parámetros entrenables
adapter.forward_delta(name, x)         # Calcular la corrección ΔW·x
adapter.merge(model_weights)           # Fusionar adaptador con el modelo base
adapter.trainable_parameter_count()    # Número de parámetros entrenables
adapter.efficiency_ratio()             # Ratio de eficiencia (params_adapter / params_base)
adapter.summary()                      # Resumen del adaptador
```

Esta uniformidad permite que cualquier framework externo intercambie adaptadores sin modificar su código de entrenamiento.

---

## 7. Adaptadores en tenIII v3.1.3 (GPU/CPU estándar)

### 7.1 LoRA — Low-Rank Adaptation

**Principio:** Descompone la actualización de pesos ΔW en el producto de dos matrices de bajo rango: ΔW = A·B, donde A ∈ ℝ^(d×r) y B ∈ ℝ^(r×k), con r ≪ min(d, k). La actualización efectiva aplicada es (alpha/rank)·A·B.

Con `rank=8` y `alpha=16`, los parámetros entrenables se reducen a menos del 1% del modelo original. Es el método de referencia y funciona bien en la mayoría de los escenarios con GPU de 8 GB o más.

```python
from tenIII.training import AdapterConfig, AdapterMethod, AdapterFactory

cfg = AdapterConfig(method=AdapterMethod.LORA, rank=8, alpha=16)
adapter = AdapterFactory.from_config(cfg)
trainable = adapter.apply(model_weights)
```

### 7.2 QLoRA — Quantized LoRA (método por defecto)

**Principio:** Combina LoRA con cuantización de 4 bits del modelo base. Los pesos del modelo base se almacenan en NF4 (Normal Float 4-bit), mientras que los adaptadores LoRA se entrenan en BF16/FP32. Esto reduce el consumo de VRAM en un 60-75% respecto a LoRA estándar.

QLoRA es el **método por defecto** en TenMiNaTor cuando no se especifica ninguno, ya que ofrece el mejor equilibrio entre memoria y calidad para la mayoría de los usuarios. Es el adaptador recomendado para GPU con 4-8 GB de VRAM.

```python
# QLoRA es el método por defecto: AdapterConfig() sin argumentos lo selecciona
cfg = AdapterConfig()  # method=QLORA automáticamente
adapter = AdapterFactory.from_config(cfg)
```

### 7.3 DoRA — Weight-Decomposed Low-Rank Adaptation

**Principio:** Descompone el peso W en magnitud (m) y dirección (V): W = m · V/‖V‖. Entrena la magnitud directamente y aplica LoRA a la dirección. Esto imita el comportamiento del ajuste fino completo con mayor fidelidad que LoRA estándar.

DoRA supera a LoRA en tareas de instrucción y razonamiento, especialmente con modelos de 7B-13B parámetros. Requiere GPU de 12 GB o más. Debe usarse cuando la calidad es prioritaria y el hardware lo permite.

### 7.4 PiSSA — Principal Singular Values and Singular Vectors Adaptation

**Principio:** Inicializa los adaptadores usando los vectores singulares principales de W (SVD). Esto hace que los adaptadores comiencen en la dirección de mayor varianza del modelo, acelerando la convergencia. Es especialmente útil cuando se dispone de pocos pasos de entrenamiento (menos de 1000 steps). Requiere GPU de 12 GB o más para el cálculo SVD inicial.

### 7.5 LoRA+ — Differential Learning Rate LoRA

**Principio:** Aplica tasas de aprendizaje diferentes a las matrices A y B del adaptador LoRA. La matriz B recibe una tasa de aprendizaje `lr_ratio` veces mayor que A (por defecto, 16×). Esto acelera la convergencia sin coste adicional de memoria.

```python
cfg = AdapterConfig(method=AdapterMethod.LORA_PLUS, lora_plus_ratio=16.0)
```

Es especialmente útil en CPU o GPU de baja potencia donde el número de steps es limitado.

### 7.6 VeRA — Vector-based Random Matrix Adaptation

**Principio:** Comparte matrices aleatorias fijas A y B entre todas las capas, y solo entrena vectores de escala (d y b) por capa. Reduce los parámetros entrenables a menos del 0.01% del modelo, con una huella de memoria mínima. Es el método más eficiente en memoria de todos los implementados. Recomendado para CPU sin GPU o dispositivos con menos de 2 GB de RAM.

### 7.7 AdaLoRA — Adaptive Budget Allocation for LoRA

**Principio:** Distribuye el presupuesto de rango entre las capas de forma adaptativa, asignando mayor rango a las capas más importantes y menor rango a las menos relevantes. Usa SVD truncada para la poda de rango.

```python
from tenIII.training import AdapterConfig, AdapterMethod

cfg_adalora = AdapterConfig(
    method=AdapterMethod.ADALORA,
    rank=8,
    alpha=16,
    adalora_budget=144,  # Presupuesto total de rango para todas las capas
)
```

Requiere GPU de 16 GB o más por el coste del SVD adaptativo. Maximiza la calidad con un presupuesto de parámetros fijo.

### 7.8 LoRA-FA — Frozen-A LoRA

**Principio:** Congela la matriz A (inicializada aleatoriamente) y solo entrena B. Esto reduce a la mitad los gradientes que hay que almacenar, con una pérdida de calidad mínima respecto a LoRA estándar. Recomendado para GPU de 6-8 GB donde LoRA estándar no cabe en memoria.

---

## 8. Adaptadores en tenIII-II v3.2 (dispositivos embebidos 32-bit)

El módulo `teniii_ii.training` está diseñado para dispositivos embebidos sin FPU (Cortex-M0/M0+), con FPU limitada (Cortex-M4/M7), y microcontroladores como ESP32 y RISC-V. Todos los cálculos usan aritmética de punto fijo Q8.8 o Q4.12.

### 8.1 MicroLoRA

**Principio:** LoRA con aritmética de punto fijo Q8.8. Las matrices A y B se almacenan como enteros de 16 bits. El producto A·B se calcula con acumuladores de 32 bits para evitar desbordamiento. Requiere un mínimo de 64 KB de RAM y procesador Cortex-M0+.

```python
from teniii_ii.training import AdapterConfig32, AdapterMethod32, AdapterFactory32

cfg = AdapterConfig32(method=AdapterMethod32.MICRO_LORA, rank=2)
adapter = AdapterFactory32.from_config(cfg)
```

### 8.2 BitLoRA

**Principio:** Extiende MicroLoRA con cuantización de 2-4 bits para los pesos del modelo base. Compatible con los formatos Q2 y Q4 del módulo `quantization` de tenIII-II. Requisito mínimo: 32 KB de RAM, Cortex-M0.

### 8.3 TinyVeRA

**Principio:** Versión 32-bit de VeRA. Usa matrices aleatorias fijas en Q4.12 y solo entrena vectores de escala de 8 bits. Huella de memoria inferior a 4 KB para modelos de hasta 1M de parámetros. Requiere un mínimo de 16 KB de RAM. Es el único método viable para Cortex-M0 con menos de 32 KB de RAM.

### 8.4 SelectiveLoRA

**Principio:** Aplica LoRA solo a las capas más importantes del modelo, seleccionadas por criterio de magnitud o SVD. Reduce el número de parámetros entrenables en un 50-80% respecto a MicroLoRA completo. Recomendado para ESP32 o Cortex-M4 con RAM entre 64-256 KB.

---

## 9. Guía de decisión por hardware

### 9.1 tenIII v3.1.3 — GPU/CPU estándar

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

### 9.2 tenIII-II v3.2 — Dispositivos embebidos 32-bit

| Dispositivo | RAM disponible | Recomendación |
|---|---|---|
| Cortex-M0 / M0+ | < 16 KB | **TinyVeRA** (único viable) |
| Cortex-M0 / M0+ | 16-64 KB | **BitLoRA** (rank=1) |
| Cortex-M4 / M7 | 64-256 KB | **MicroLoRA** (rank=2-4) |
| ESP32 | 256-512 KB | **SelectiveLoRA** (rank=4) |
| RISC-V (HiFive) | 512 KB+ | **MicroLoRA** (rank=4-8) |
| Raspberry Pi Zero | 512 MB | **MicroLoRA** (rank=8) |

> **Nota sobre FPU:** Los dispositivos Cortex-M0/M0+ no tienen FPU. TenMiNaTor usa aritmética de punto fijo Q8.8 automáticamente cuando detecta `DeviceClass.CORTEX_M0`. Los dispositivos Cortex-M4/M7 con FPU pueden usar Q8.8 o FP32 según la configuración.

### 9.3 Árbol de decisión rápido

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

### 9.4 Selección automática por código

```python
from tenIII.training import HardwareDetector

# Detectar hardware y obtener recomendación automática
profile = HardwareDetector.detect()
method, reason = HardwareDetector.recommend(
    profile=profile,
    model_size_b=7.0,
    priority="balanced",  # "memory" | "speed" | "quality" | "balanced"
)
print(f"Hardware: {profile.value}")
print(f"Método recomendado: {method.value}")
print(f"Razón: {reason}")
```

---

## 10. Integración de adaptadores con GradientMode

Todos los adaptadores de TenMiNaTor se integran con los 3 modos de `GradientMode` a través de `AdaptedTrainer` (tenIII) y `AdaptedTrainer32` (tenIII-II).

### 10.1 Combinaciones recomendadas

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

### 10.2 Configuración completa con AdaptedTrainer

```python
from tenIII.training import (
    AdaptedTrainer, AdaptedTrainingConfig,
    AdapterConfig, AdapterMethod
)

cfg = AdaptedTrainingConfig(
    # Adaptador
    auto_select_adapter=False,
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
    gradient_accumulation_steps=8,
    # GradientMode
    gradient_mode="accumulate",
    gradient_decay_factor=0.9,
    # Fusión al finalizar
    merge_on_finish=True,
)

trainer = AdaptedTrainer(cfg)
trainer.setup(model_weights)
metrics = trainer.train(train_data)
print(f"Adaptador usado: {metrics['adapter_method']}")
print(f"Parámetros entrenables: {trainer.adapter.trainable_parameter_count():,}")
```

### 10.3 Configuración para dispositivos embebidos (tenIII-II)

```python
from teniii_ii.training import AdaptedTrainer32, AdaptedTrainingConfig32

cfg = AdaptedTrainingConfig32(
    auto_select=True,
    ram_kb=128,
    has_fpu=True,
    gradient_mode="accumulate",
    batch_size=1,
    gradient_accumulation_steps=16,
)
trainer = AdaptedTrainer32(cfg)
trainer.setup(model_weights_int16)
```

---

## 11. Casos de uso avanzados

### 11.1 Fine-tuning incremental con CONTINUOUS_DESCENT

El modo `CONTINUOUS_DESCENT` permite reanudar el entrenamiento conservando el momentum de los gradientes anteriores. Es ideal para el fine-tuning incremental donde se añaden nuevos datos periódicamente:

```python
# Primera sesión de entrenamiento
cfg = AdaptedTrainingConfig(
    auto_select_adapter=True,
    gradient_mode="continuous_descent",
    gradient_decay_factor=0.95,
    merge_on_finish=False,
)
trainer = AdaptedTrainer(cfg)
trainer.setup(model_weights)
trainer.train(batch_1)

# Guardar estado del adaptador y los gradientes
adapter_state = trainer.get_trainable_params()
gradient_state = trainer._gradients

# Segunda sesión (nuevos datos)
trainer2 = AdaptedTrainer(cfg)
trainer2.setup(model_weights)
trainer2._trainable_params = adapter_state
trainer2._gradients = gradient_state  # Restaurar gradientes con momentum
trainer2.train(batch_2)               # Continúa desde donde se dejó
```

### 11.2 QLoRA + ACCUMULATE para GPU 4 GB

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

### 11.3 Fine-tuning en Cortex-M4 con tenIII-II

```python
from teniii_ii.training import (
    AdaptedTrainer32, AdaptedTrainingConfig32,
    AdapterMethod32, AdapterConfig32
)

cfg = AdaptedTrainingConfig32(
    auto_select=False,
    adapter_config=AdapterConfig32(
        method=AdapterMethod32.MICRO_LORA,
        rank=4,
        fixed_point_bits=8,
    ),
    ram_kb=256,
    has_fpu=True,
    batch_size=2,
    gradient_accumulation_steps=8,
    gradient_mode="accumulate",
    learning_rate=1e-3,
)

trainer = AdaptedTrainer32(cfg)
trainer.setup(model_weights_int16)
metrics = trainer.train(micro_dataset)
```

---

## 12. Referencia de API de adaptadores

### 12.1 `AdapterConfig` (tenIII v3.1.3)

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

### 12.2 `AdapterMethod` (tenIII v3.1.3)

| Valor | Descripción | VRAM mínima | Parámetros entrenables |
|---|---|---|---|
| `LORA` | LoRA estándar | 8 GB | ~0.5-1% |
| `QLORA` | LoRA con cuantización 4-bit (por defecto) | 4 GB | ~0.5-1% |
| `DORA` | Weight-Decomposed LoRA | 12 GB | ~0.5-1% |
| `PISSA` | Principal Singular Values Adaptation | 12 GB | ~0.5-1% |
| `LORA_PLUS` | LoRA con ratio diferencial de LR | 8 GB | ~0.5-1% |
| `VERA` | Vector-based Random Matrix Adaptation | CPU | ~0.01% |
| `ADALORA` | Adaptive Budget Allocation LoRA | 16 GB | ~0.5-1% |
| `LORA_FA` | Frozen-A LoRA | 6 GB | ~0.25-0.5% |

### 12.3 `AdapterMethod32` (tenIII-II v3.2)

| Valor | Descripción | RAM mínima | Precisión |
|---|---|---|---|
| `MICRO_LORA` | LoRA en punto fijo Q8.8 | 64 KB | Q8.8 |
| `BIT_LORA` | LoRA con cuantización 2-4 bits | 32 KB | Q4.12 |
| `TINY_VERA` | VeRA ultra-ligero Q4.12 | 16 KB | Q4.12 |
| `SELECTIVE_LORA` | LoRA selectivo por importancia | 64 KB | Q8.8 |

### 12.4 `HardwareProfile` (tenIII v3.1.3)

| Valor | Descripción |
|---|---|
| `GPU_4GB` | GPU con 4 GB de VRAM |
| `GPU_8GB` | GPU con 8 GB de VRAM |
| `GPU_12GB` | GPU con 12 GB de VRAM |
| `GPU_24GB` | GPU con 24 GB de VRAM |
| `GPU_40GB_PLUS` | GPU con 40+ GB de VRAM |
| `MULTI_GPU` | Configuración multi-GPU |
| `CPU_ONLY` | Sin GPU, solo CPU |

### 12.5 `DeviceClass32` (tenIII-II v3.2)

| Valor | Descripción |
|---|---|
| `CORTEX_M0` | ARM Cortex-M0/M0+, sin FPU |
| `CORTEX_M4` | ARM Cortex-M4/M7, con FPU |
| `ESP32` | ESP32 / ESP32-S3 |
| `RISCV_32` | RISC-V 32-bit |
| `GENERIC_32BIT` | Cualquier dispositivo 32-bit genérico |

---

# Parte 3 — Creación de Frameworks con tenIII desde PyPI

> Esta parte está dirigida exclusivamente a usuarios que instalan tenIII a través de PyPI (`pip install tenIII`). Describe cómo construir un framework de entrenamiento completo y distribuible usando tenIII como dependencia de librería.

> **Importante:** tenIII v3.1.3 se distribuye **exclusivamente desde la web** (http://Terminator.com.es). Las versiones anteriores disponibles en PyPI son compatibles con esta guía, pero las funcionalidades descritas en la Parte 2 (módulo `training/`) son exclusivas de v3.1.3.

---

## 13. Instalación y estructura del proyecto

### 13.1 Instalación desde PyPI

```bash
pip install tenIII
```

Para verificar la instalación:

```python
import tenIII
print(tenIII.__version__)  # Debe mostrar la versión instalada

# Verificar que el módulo de entrenamiento está disponible
from tenIII.training import AdaptedTrainer, AdapterConfig
from tenIII.reasoning.grpo_backprop import SimulationGRPOModel, GradientMode
print("Módulos de entrenamiento disponibles ✓")
```

### 13.2 Estructura recomendada del proyecto

Un framework de entrenamiento basado en tenIII debe seguir esta estructura de proyecto:

```
mi_framework_entrenamiento/
├── pyproject.toml              ← Metadatos del paquete y dependencias
├── mi_framework/
│   ├── __init__.py             ← Exportaciones públicas del framework
│   ├── trainer.py              ← Clase principal del trainer (subclase de AdaptedTrainer)
│   ├── config.py               ← Configuración del framework
│   ├── data.py                 ← Carga y preprocesamiento de datos
│   ├── export.py               ← Exportación del modelo entrenado
│   └── cli.py                  ← Interfaz de línea de comandos (opcional)
└── tests/
    ├── test_trainer.py
    ├── test_config.py
    └── test_export.py
```

### 13.3 `pyproject.toml` mínimo

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "mi-framework-entrenamiento"
version = "1.0.0"
description = "Framework de entrenamiento LLM basado en tenIII"
requires-python = ">=3.9"
dependencies = [
    "tenIII>=3.1.0",
    "numpy>=1.24",
    "terminatodo>=1.0.0",   # Opcional: para almacenamiento
]

[project.scripts]
mi-framework = "mi_framework.cli:main"

[project.urls]
Homepage = "http://Terminator.com.es"
```

---

## 14. Subclasificación de AdaptedTrainer

`AdaptedTrainer` es la clase base de tenIII para el entrenamiento con adaptadores. Para crear un framework propio, se debe subclasificar y sobrescribir los métodos de ciclo de vida.

### 14.1 Métodos de ciclo de vida disponibles

| Método | Cuándo se llama | Uso típico |
|---|---|---|
| `on_train_begin(config)` | Al inicio del entrenamiento | Inicializar logging, callbacks |
| `on_epoch_begin(epoch)` | Al inicio de cada epoch | Ajustar LR, shuffle de datos |
| `on_batch_begin(batch_idx)` | Al inicio de cada batch | Preprocesamiento específico |
| `on_batch_end(batch_idx, loss)` | Al final de cada batch | Logging, gradient clipping |
| `on_epoch_end(epoch, metrics)` | Al final de cada epoch | Guardar checkpoint, early stopping |
| `on_train_end(metrics)` | Al final del entrenamiento | Exportar modelo, limpiar recursos |

### 14.2 Implementación de un trainer personalizado

```python
# mi_framework/trainer.py

import numpy as np
from typing import Optional, Dict, Any, List
from tenIII.training import AdaptedTrainer, AdaptedTrainingConfig, AdapterConfig, AdapterMethod
from tenIII.reasoning.grpo_backprop import GradientMode


class MiTrainer(AdaptedTrainer):
    """
    Trainer personalizado basado en tenIII.AdaptedTrainer.

    Añade: logging detallado, early stopping, y guardado automático
    de checkpoints en TerminaTodo.
    """

    def __init__(
        self,
        config: AdaptedTrainingConfig,
        patience: int = 3,
        min_delta: float = 1e-4,
        log_interval: int = 10,
    ):
        super().__init__(config)
        self.patience = patience
        self.min_delta = min_delta
        self.log_interval = log_interval

        # Estado interno
        self._best_loss: float = float("inf")
        self._patience_counter: int = 0
        self._log: List[Dict[str, Any]] = []
        self._storage = None

    def on_train_begin(self, config: AdaptedTrainingConfig) -> None:
        """Inicializa el almacenamiento y el log al inicio del entrenamiento."""
        print(f"[MiTrainer] Iniciando entrenamiento")
        print(f"  Adaptador: {config.adapter_config.method.value if config.adapter_config else 'auto'}")
        print(f"  GradientMode: {config.gradient_mode}")
        print(f"  Epochs: {config.num_epochs}")

        # Inicializar almacenamiento con TerminaTodo
        try:
            from terminatodo import select_best_backend
            self._storage = select_best_backend()
            print(f"  Almacenamiento: {self._storage.name}")
        except ImportError:
            print("  Almacenamiento: TerminaTodo no disponible, usando memoria")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Guarda checkpoint y evalúa early stopping al final de cada epoch."""
        loss = metrics.get("loss", float("inf"))
        self._log.append({"epoch": epoch, "loss": loss})

        print(f"[MiTrainer] Epoch {epoch}: loss={loss:.6f}")

        # Early stopping
        if loss < self._best_loss - self.min_delta:
            self._best_loss = loss
            self._patience_counter = 0
            self._save_best_checkpoint(epoch, loss)
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.patience:
                print(f"[MiTrainer] Early stopping en epoch {epoch} "
                      f"(sin mejora en {self.patience} epochs)")
                raise StopIteration("early_stopping")

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        """Exporta el modelo al finalizar el entrenamiento."""
        print(f"[MiTrainer] Entrenamiento finalizado")
        print(f"  Mejor pérdida: {self._best_loss:.6f}")
        print(f"  Historial: {[round(e['loss'], 4) for e in self._log]}")

    def _save_best_checkpoint(self, epoch: int, loss: float) -> None:
        """Guarda el mejor checkpoint en el backend de almacenamiento."""
        if self._storage is None:
            return
        import pickle
        state = {
            "epoch": epoch,
            "loss": loss,
            "weights": self.get_trainable_params(),
            "adapter_method": self.adapter.summary() if self.adapter else None,
        }
        self._storage.save(f"best_checkpoint_epoch_{epoch}.pkl", pickle.dumps(state))
        print(f"  Checkpoint guardado: epoch {epoch}, loss={loss:.6f}")

    def get_training_log(self) -> List[Dict[str, Any]]:
        """Devuelve el historial completo de entrenamiento."""
        return self._log
```

---

## 15. Configuración de AdapterConfig y GradientMode

### 15.1 Configuración por hardware detectado automáticamente

El patrón más sencillo para un framework distribuible es dejar que tenIII detecte el hardware y seleccione el adaptador óptimo:

```python
# mi_framework/config.py

from tenIII.training import AdaptedTrainingConfig, HardwareDetector

def crear_config_automatica(
    model_size_b: float = 7.0,
    priority: str = "balanced",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
) -> AdaptedTrainingConfig:
    """
    Crea una configuración de entrenamiento adaptada al hardware disponible.

    Args:
        model_size_b: Tamaño del modelo en miles de millones de parámetros.
        priority: "memory" | "speed" | "quality" | "balanced"
        num_epochs: Número de epochs de entrenamiento.
        learning_rate: Tasa de aprendizaje inicial.

    Returns:
        AdaptedTrainingConfig lista para usar con MiTrainer.
    """
    profile = HardwareDetector.detect()
    method, reason = HardwareDetector.recommend(
        profile=profile,
        model_size_b=model_size_b,
        priority=priority,
    )

    print(f"Hardware detectado: {profile.value}")
    print(f"Adaptador seleccionado: {method.value} — {reason}")

    return AdaptedTrainingConfig(
        auto_select_adapter=True,
        model_size_b=model_size_b,
        hardware_priority=priority,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=4,
        gradient_accumulation_steps=8,
        gradient_mode="accumulate",
        merge_on_finish=True,
    )
```

### 15.2 Configuración manual por perfil de hardware

Para frameworks que necesitan control explícito sobre el adaptador:

```python
from tenIII.training import (
    AdaptedTrainingConfig, AdapterConfig, AdapterMethod
)
from tenIII.reasoning.grpo_backprop import GradientMode

# Perfil para GPU 4 GB (QLoRA + ACCUMULATE)
CONFIG_GPU_4GB = AdaptedTrainingConfig(
    auto_select_adapter=False,
    adapter_config=AdapterConfig(
        method=AdapterMethod.QLORA,
        rank=4, alpha=8, quant_bits=4,
    ),
    learning_rate=1e-4,
    num_epochs=3,
    batch_size=1,
    gradient_accumulation_steps=32,
    gradient_mode="accumulate",
    merge_on_finish=True,
)

# Perfil para GPU 24 GB (DoRA + RESET)
CONFIG_GPU_24GB = AdaptedTrainingConfig(
    auto_select_adapter=False,
    adapter_config=AdapterConfig(
        method=AdapterMethod.DORA,
        rank=16, alpha=32,
    ),
    learning_rate=2e-4,
    num_epochs=5,
    batch_size=8,
    gradient_accumulation_steps=4,
    gradient_mode="reset",
    merge_on_finish=True,
)

# Perfil para CPU (VeRA + ACCUMULATE)
CONFIG_CPU = AdaptedTrainingConfig(
    auto_select_adapter=False,
    adapter_config=AdapterConfig(
        method=AdapterMethod.VERA,
        rank=8, alpha=16,
    ),
    learning_rate=5e-5,
    num_epochs=10,
    batch_size=2,
    gradient_accumulation_steps=16,
    gradient_mode="accumulate",
    merge_on_finish=True,
)

PERFILES = {
    "gpu_4gb": CONFIG_GPU_4GB,
    "gpu_24gb": CONFIG_GPU_24GB,
    "cpu": CONFIG_CPU,
}
```

---

## 16. Exportación del modelo entrenado

Tras el entrenamiento, tenIII ofrece múltiples formatos de exportación a través del módulo `quantization.export`.

### 16.1 Formatos de exportación disponibles

| Formato | Módulo | Uso principal |
|---|---|---|
| GGUF | `quantization.export` | Inferencia con llama.cpp, Ollama |
| safetensors | `quantization.export` | Hugging Face, interoperabilidad |
| ONNX | `quantization.export` | Inferencia multiplataforma |
| Chip | `quantization.export` | Despliegue en hardware dedicado |
| Unikernel | `quantization.export` | Despliegue en sistemas embebidos |
| USB | `packaging.usb_packaging` | Despliegue portátil en pendrive |

### 16.2 Exportación a GGUF

```python
# mi_framework/export.py

from tenIII.quantization.export import ExportManager, ExportFormat
from tenIII.quantization.formats import QuantizationConfig, QuantizationFormat


def exportar_gguf(
    weights: dict,
    output_path: str,
    quantization: str = "Q4_K_M",
) -> str:
    """
    Exporta los pesos del modelo a formato GGUF.

    Args:
        weights: Diccionario de pesos del modelo (post-merge del adaptador).
        output_path: Ruta del archivo de salida (sin extensión).
        quantization: Nivel de cuantización GGUF ("Q4_K_M", "Q5_K_M", "Q8_0", etc.)

    Returns:
        Ruta completa del archivo GGUF generado.
    """
    quant_map = {
        "Q4_K_M": QuantizationFormat.Q4,
        "Q5_K_M": QuantizationFormat.Q5,
        "Q8_0": QuantizationFormat.Q8,
    }

    quant_cfg = QuantizationConfig(
        format=quant_map.get(quantization, QuantizationFormat.Q4),
        calibration_samples=128,
    )

    manager = ExportManager(weights, quant_cfg)
    result = manager.export(ExportFormat.GGUF, output_path)
    print(f"Modelo exportado a GGUF: {result}")
    return result


def exportar_safetensors(weights: dict, output_path: str) -> str:
    """Exporta los pesos a formato safetensors."""
    manager = ExportManager(weights)
    result = manager.export(ExportFormat.SAFETENSORS, output_path)
    print(f"Modelo exportado a safetensors: {result}")
    return result


def exportar_usb(weights: dict, model_name: str = "modelo") -> bool:
    """
    Exporta el modelo a un dispositivo USB conectado.
    Usa TerminaTodo para detectar el USB automáticamente.
    """
    from tenIII.packaging.usb_packaging import USBPackager
    packager = USBPackager()
    success = packager.package(weights, model_name)
    if success:
        print(f"Modelo '{model_name}' guardado en USB ✓")
    else:
        print("No se encontró dispositivo USB disponible")
    return success
```

### 16.3 Pipeline completo de exportación

```python
from mi_framework.export import exportar_gguf, exportar_safetensors, exportar_usb

def pipeline_exportacion(trainer, output_dir: str = "./output"):
    """Pipeline completo: merge del adaptador + exportación en múltiples formatos."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Obtener pesos con el adaptador fusionado
    merged_weights = trainer.adapter.merge(trainer._base_weights)

    # 2. Exportar a GGUF (para inferencia con llama.cpp)
    exportar_gguf(merged_weights, f"{output_dir}/modelo_final", quantization="Q4_K_M")

    # 3. Exportar a safetensors (para Hugging Face)
    exportar_safetensors(merged_weights, f"{output_dir}/modelo_final")

    # 4. Intentar exportar a USB si hay uno disponible
    exportar_usb(merged_weights, model_name="modelo_final")

    print(f"\nExportación completada en: {output_dir}/")
```

---

## 17. Ejemplo mínimo funcional completo

Esta sección presenta un ejemplo completo y ejecutable de un framework de entrenamiento construido sobre tenIII desde PyPI. El ejemplo integra todas las partes anteriores: configuración automática, trainer personalizado, entrenamiento, y exportación.

### 17.1 Código completo del framework

```python
#!/usr/bin/env python3
"""
ejemplo_framework_completo.py
Framework de entrenamiento mínimo basado en tenIII.
Instalación: pip install tenIII terminatodo

Uso:
    python ejemplo_framework_completo.py --epochs 5 --priority memory
"""

import argparse
import numpy as np
from typing import Dict, List, Optional

# ── Imports de tenIII ──────────────────────────────────────────────────────────
from tenIII.reasoning.grpo_backprop import SimulationGRPOModel, GradientMode
from tenIII.training import (
    AdaptedTrainer,
    AdaptedTrainingConfig,
    AdapterConfig,
    AdapterMethod,
    HardwareDetector,
)

# ── Imports opcionales ─────────────────────────────────────────────────────────
try:
    from terminatodo import select_best_backend
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def crear_config(
    priority: str = "balanced",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
) -> AdaptedTrainingConfig:
    """Detecta el hardware y crea la configuración óptima."""
    profile = HardwareDetector.detect()
    method, reason = HardwareDetector.recommend(
        profile=profile, model_size_b=7.0, priority=priority
    )
    print(f"[Config] Hardware: {profile.value} → {method.value} ({reason})")

    return AdaptedTrainingConfig(
        auto_select_adapter=True,
        model_size_b=7.0,
        hardware_priority=priority,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=4,
        gradient_accumulation_steps=8,
        gradient_mode="accumulate",
        merge_on_finish=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINER PERSONALIZADO
# ══════════════════════════════════════════════════════════════════════════════

class MiTrainer(AdaptedTrainer):
    """Trainer con early stopping, logging y guardado automático."""

    def __init__(self, config: AdaptedTrainingConfig, patience: int = 3):
        super().__init__(config)
        self.patience = patience
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._log: List[Dict] = []
        self._storage = select_best_backend() if STORAGE_AVAILABLE else None

    def on_train_begin(self, config):
        adapter_name = (
            config.adapter_config.method.value
            if config.adapter_config else "auto"
        )
        print(f"[Trainer] Iniciando — adaptador: {adapter_name}, "
              f"modo: {config.gradient_mode}, epochs: {config.num_epochs}")
        if self._storage:
            print(f"[Trainer] Almacenamiento: {self._storage.name}")

    def on_epoch_end(self, epoch: int, metrics: Dict):
        loss = metrics.get("loss", float("inf"))
        self._log.append({"epoch": epoch, "loss": loss})
        print(f"[Trainer] Epoch {epoch:3d} | loss: {loss:.6f} "
              f"| mejor: {self._best_loss:.6f}")

        if loss < self._best_loss - 1e-4:
            self._best_loss = loss
            self._patience_counter = 0
            self._guardar_checkpoint(epoch, loss)
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.patience:
                print(f"[Trainer] Early stopping (sin mejora en {self.patience} epochs)")
                raise StopIteration("early_stopping")

    def on_train_end(self, metrics: Dict):
        print(f"[Trainer] Finalizado — mejor pérdida: {self._best_loss:.6f}")

    def _guardar_checkpoint(self, epoch: int, loss: float):
        if not self._storage:
            return
        import pickle
        state = {
            "epoch": epoch, "loss": loss,
            "params": self.get_trainable_params(),
        }
        self._storage.save(f"ckpt_epoch_{epoch:04d}.pkl", pickle.dumps(state))
        print(f"[Trainer] Checkpoint guardado (epoch {epoch})")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATOS DE EJEMPLO
# ══════════════════════════════════════════════════════════════════════════════

def generar_dataset(
    num_batches: int = 20,
    batch_size: int = 4,
    input_dim: int = 64,
) -> List[List[np.ndarray]]:
    """Genera un dataset sintético de ejemplo."""
    return [
        [np.random.randn(batch_size, input_dim).astype(np.float32)]
        for _ in range(num_batches)
    ]


def generar_pesos_modelo(input_dim: int = 64) -> Dict[str, np.ndarray]:
    """Genera pesos iniciales de ejemplo para el modelo."""
    rng = np.random.default_rng(42)
    return {
        "attention.q_proj": rng.standard_normal((input_dim, input_dim)).astype(np.float32) * 0.02,
        "attention.v_proj": rng.standard_normal((input_dim, input_dim)).astype(np.float32) * 0.02,
        "ffn.up_proj":      rng.standard_normal((input_dim, input_dim * 4)).astype(np.float32) * 0.02,
        "ffn.down_proj":    rng.standard_normal((input_dim * 4, input_dim)).astype(np.float32) * 0.02,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPORTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def exportar_modelo(trainer: MiTrainer, output_dir: str = "./output_modelo"):
    """Exporta el modelo entrenado a GGUF y safetensors."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Obtener pesos con adaptador fusionado
    if trainer.adapter:
        merged = trainer.adapter.merge(trainer._base_weights)
    else:
        merged = trainer._base_weights

    # Exportar a GGUF
    try:
        from tenIII.quantization.export import ExportManager, ExportFormat
        from tenIII.quantization.formats import QuantizationConfig, QuantizationFormat
        quant_cfg = QuantizationConfig(format=QuantizationFormat.Q4)
        mgr = ExportManager(merged, quant_cfg)
        mgr.export(ExportFormat.GGUF, f"{output_dir}/modelo_final")
        print(f"[Export] GGUF guardado en {output_dir}/modelo_final.gguf")
    except Exception as e:
        print(f"[Export] GGUF no disponible: {e}")

    # Exportar a safetensors
    try:
        from tenIII.quantization.export import ExportManager, ExportFormat
        mgr = ExportManager(merged)
        mgr.export(ExportFormat.SAFETENSORS, f"{output_dir}/modelo_final")
        print(f"[Export] safetensors guardado en {output_dir}/modelo_final.safetensors")
    except Exception as e:
        print(f"[Export] safetensors no disponible: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Framework de entrenamiento basado en tenIII")
    parser.add_argument("--epochs", type=int, default=5, help="Número de epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--priority", choices=["memory", "speed", "quality", "balanced"],
        default="balanced", help="Prioridad de hardware"
    )
    parser.add_argument("--patience", type=int, default=3, help="Paciencia para early stopping")
    parser.add_argument("--export", action="store_true", help="Exportar modelo al finalizar")
    args = parser.parse_args()

    print("=" * 60)
    print("  Framework de Entrenamiento — tenIII PyPI")
    print("  Gotham City AI Lab — http://Terminator.com.es")
    print("=" * 60)

    # 1. Crear configuración
    config = crear_config(
        priority=args.priority,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )

    # 2. Generar datos y pesos de ejemplo
    model_weights = generar_pesos_modelo()
    dataset = generar_dataset(num_batches=20)

    # 3. Crear y ejecutar el trainer
    trainer = MiTrainer(config, patience=args.patience)
    trainer.setup(model_weights)

    try:
        metrics = trainer.train(dataset)
        print(f"\nResultados finales:")
        print(f"  Pérdida final:    {metrics.get('final_loss', 'N/A'):.6f}")
        print(f"  Adaptador usado:  {metrics.get('adapter_method', 'N/A')}")
        print(f"  Parámetros:       {metrics.get('trainable_params', 'N/A'):,}")
    except StopIteration:
        print("Entrenamiento detenido por early stopping")

    # 4. Exportar si se solicitó
    if args.export:
        exportar_modelo(trainer)

    print("\n¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
```

### 17.2 Ejecución del ejemplo

```bash
# Instalación de dependencias
pip install tenIII terminatodo

# Ejecución básica (detección automática de hardware)
python ejemplo_framework_completo.py

# Con exportación del modelo
python ejemplo_framework_completo.py --epochs 10 --priority quality --export

# En GPU con poca VRAM
python ejemplo_framework_completo.py --priority memory --epochs 3

# En CPU
python ejemplo_framework_completo.py --priority memory --epochs 20 --lr 5e-5
```

### 17.3 Salida esperada

```
============================================================
  Framework de Entrenamiento — tenIII PyPI
  Gotham City AI Lab — http://Terminator.com.es
============================================================
[Config] Hardware: cpu_only → vera (CPU sin GPU: VeRA minimiza memoria)
[Trainer] Iniciando — adaptador: auto, modo: accumulate, epochs: 5
[Trainer] Almacenamiento: local_storage
[Trainer] Epoch   1 | loss: 0.251234 | mejor: inf
[Trainer] Checkpoint guardado (epoch 1)
[Trainer] Epoch   2 | loss: 0.198765 | mejor: 0.251234
[Trainer] Checkpoint guardado (epoch 2)
[Trainer] Epoch   3 | loss: 0.187432 | mejor: 0.198765
[Trainer] Checkpoint guardado (epoch 3)
[Trainer] Epoch   4 | loss: 0.189012 | mejor: 0.187432
[Trainer] Epoch   5 | loss: 0.191234 | mejor: 0.187432
[Trainer] Finalizado — mejor pérdida: 0.187432

Resultados finales:
  Pérdida final:    0.191234
  Adaptador usado:  vera
  Parámetros:       1,024
```

---

## Resumen del manual

Este manual ha cubierto tres niveles de uso del ecosistema TenMiNaTor:

| Parte | Audiencia | Contenido principal |
|---|---|---|
| **Parte 1 — Framework Base** | Todos los usuarios | GradientMode (RESET/ACCUMULATE/CONTINUOUS_DESCENT), construcción de frameworks básicos, integración con TerminaTodo y Terminator AliAmalia |
| **Parte 2 — Adaptadores Avanzados** | Usuarios de tenIII v3.1.3 y tenIII-II v3.2 | 8 adaptadores GPU/CPU (QLoRA por defecto), 4 adaptadores MCU, guía de decisión por hardware, combinaciones con GradientMode |
| **Parte 3 — Framework con PyPI** | Desarrolladores que usan `pip install tenIII` | Estructura de proyecto, subclasificación de AdaptedTrainer, configuración por hardware, exportación GGUF/safetensors/USB, ejemplo completo ejecutable |

Para descargar tenIII v3.1.3 (versión completa con módulo `training/`), visita: **http://Terminator.com.es**

---

*Manual generado por Gotham City AI Lab · TenMiNaTor v3.1.3 / tenIII-II v3.2 · Mayo 2026*  
*http://Terminator.com.es*
