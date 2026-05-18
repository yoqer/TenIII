# Manual de Usuario — Ten III v3.1.1
## Framework Avanzado de Entrenamiento de Deep Learning
### Ecosistema Gotham City

---

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Conceptos Fundamentales](#conceptos-fundamentales)
3. [Funciones Principales](#funciones-principales)
4. [Invocaciones Básicas](#invocaciones-básicas)
5. [Integración de LLMs](#integración-de-llms)
6. [Framework de Entrenamiento](#framework-de-entrenamiento)
7. [Ejemplos Prácticos](#ejemplos-prácticos)
8. [Resolución de Problemas](#resolución-de-problemas)

---

## Instalación

### Opción 1: Desde PyPI (Recomendado)

```bash
# Instalación básica (Core Training)
pip install tenIII

# Con soporte para PyTorch
pip install tenIII[torch]

# Con soporte para TensorFlow
pip install tenIII[tensorflow]

# Con todos los módulos optativos
pip install tenIII[all]

# Módulos específicos
pip install tenIII[blocks]           # Carga por bloques
pip install tenIII[constitution]     # Constitutional AI
pip install tenIII[multimodal]       # Voz y KAME
pip install tenIII[packaging]        # USB portátil
```

### Opción 2: Desde archivo descargado

```bash
# Descargar desde terminator.com.es
wget http://terminator.com.es

# Instalar
pip install tenIII-3.1.2.tar.gz
```

### Verificar instalación

```python
import tenIII
print(f"tenIII v{tenIII.__version__}")
print(f"Ecosistema: {tenIII.__ecosystem__}")

# Verificar módulos disponibles
from tenIII import Trainer, TrainingConfig
from tenIII.nn import Linear, Conv2d, Transformer
print("✓ Instalación correcta")
```

---

## Conceptos Fundamentales

### 1. TrainingConfig
Objeto de configuración que define todos los parámetros del entrenamiento.

```python
from tenIII import TrainingConfig, EarlyStoppingMode, TrainingMode

config = TrainingConfig(
    # Modelo
    model_type="transformer",        # "transformer", "cnn", "rnn", "custom"
    hidden_dim=768,                  # Dimensión de capas ocultas
    num_layers=12,                   # Número de capas
    
    # Entrenamiento
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    
    # Early Stopping (Sistema 10×12)
    early_stopping_mode=EarlyStoppingMode.IMPROVED,  # CLASSIC, IMPROVED, ADAPTIVE
    early_stopping_patience=12,      # Iteraciones sin mejora antes de parar
    
    # Control continuo
    continuous_training=True,        # Nunca se detiene
    adaptive_learning_rate=True,     # Ajusta LR automáticamente
    
    # Backend
    backend="pytorch",               # "pytorch", "tensorflow", "auto"
    device="cuda",                   # "cuda", "cpu", "auto"
)
```

### 2. Trainer
Clase principal que gestiona el entrenamiento.

```python
from tenIII import Trainer

trainer = Trainer(config)

# Construir modelo
model = trainer.build_model()

# Entrenar
history = trainer.train(train_loader, val_loader)

# Guardar
trainer.save("modelo.pt")

# Cargar
trainer.load("modelo.pt")
```

### 3. EarlyStoppingMode
Tres modos de detención temprana:

| Modo | Descripción | Cuándo usar |
|---|---|---|
| **CLASSIC** | Umbral absoluto (min_delta=1e-6) | Entrenamientos rápidos |
| **IMPROVED** | Umbral relativo (0.1% de mejora) | Entrenamientos largos |
| **ADAPTIVE** | Ajusta umbral según volatilidad | Entrenamientos inestables |

---

## Funciones Principales

### Core Training

#### `Trainer.build_model()`
Construye el modelo según la configuración.

```python
trainer = Trainer(config)
model = trainer.build_model()
# Retorna: modelo PyTorch o TensorFlow según backend
```

#### `Trainer.train(train_loader, val_loader, max_epochs=None)`
Ejecuta el bucle de entrenamiento.

```python
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    max_epochs=100,
)

# history contiene:
# - losses: lista de pérdidas por época
# - val_losses: pérdidas de validación
# - metrics: métricas adicionales
# - early_stop_reason: razón de parada (si aplica)
```

#### `Trainer.save(path)`
Guarda el modelo y su estado.

```python
trainer.save("mi_modelo.pt")
# Guarda: modelo, config, optimizer state, checkpoint
```

#### `Trainer.load(path)`
Carga un modelo previamente guardado.

```python
trainer.load("mi_modelo.pt")
# Restaura: modelo, config, optimizer state
```

### Block System (Opcional)

#### `BlockSplitter.split_model(model, block_size=2)`
Divide un modelo en bloques independientes.

```python
from tenIII.blocks import BlockSplitter

splitter = BlockSplitter(model)
blocks = splitter.split_model(block_size=2)  # 2 capas por bloque

# blocks es una lista de modelos independientes
for i, block in enumerate(blocks):
    print(f"Bloque {i}: {block}")
```

#### `BlockLoader.load_block(block_id)`
Carga un bloque específico en memoria.

```python
from tenIII.blocks import BlockLoader

loader = BlockLoader("modelo_bloques/")
block = loader.load_block(block_id=0)
# Solo el bloque 0 está en memoria
```

#### `LayerFreezer.freeze_layers(model, layer_indices)`
Congela capas específicas.

```python
from tenIII.blocks import LayerFreezer

freezer = LayerFreezer(model)
freezer.freeze_layers([0, 1, 2])  # Congela capas 0, 1, 2
# Solo las capas 3+ se entrenarán
```

### Constitutional AI (Opcional)

#### `ConstitutionManager.create_constitution(name, policy_names, weights)`
Crea una constitución con políticas ponderadas.

```python
from tenIII.constitution import ConstitutionManager, HonestPolicy, FairnessPolicy

manager = ConstitutionManager()
manager.register_policy(HonestPolicy(), "honesty")
manager.register_policy(FairnessPolicy(), "fairness")

constitution = manager.create_constitution(
    name="safe_assistant",
    policy_names=["honesty", "fairness"],
    weights={"honesty": 0.6, "fairness": 0.4}
)
```

#### `ConstitutionManager.evaluate_response(response, constitution_name)`
Evalúa una respuesta según la constitución.

```python
score = manager.evaluate_response(
    response="Respuesta del modelo",
    constitution_name="safe_assistant"
)
print(f"Puntuación de seguridad: {score:.2f}")
```

### Multimodal con Voz (Opcional)

#### `VoiceDataProcessor.process_audio(audio_path)`
Procesa archivos de audio para entrenamiento.

```python
from tenIII.multimodal import VoiceDataProcessor

processor = VoiceDataProcessor()
features = processor.process_audio("audio.wav")
# features: espectrograma mel de 128 dimensiones
```

#### `VoiceInferenceEngine.generate_with_voice(prompt, output_format="audio")`
Genera respuestas con salida de voz.

```python
from tenIII.multimodal import VoiceInferenceEngine

engine = VoiceInferenceEngine(model)
audio = engine.generate_with_voice(
    prompt="¿Cuál es el capital de Francia?",
    output_format="audio"  # "audio", "text", "both"
)
# Guarda audio.wav con la respuesta
```

---

## Invocaciones Básicas

### Entrenamiento Simple

```python
from tenIII import Trainer, TrainingConfig
import torch
from torch.utils.data import DataLoader, TensorDataset

# Crear datos de ejemplo
X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32)

# Configurar
config = TrainingConfig(
    model_type="transformer",
    hidden_dim=128,
    num_layers=4,
    learning_rate=1e-3,
    num_epochs=10,
)

# Entrenar
trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, train_loader)

# Guardar
trainer.save("modelo_simple.pt")
```

### Entrenamiento con Bloques

```python
from tenIII import Trainer, TrainingConfig
from tenIII.blocks import BlockSplitter, BlockTrainer

# Entrenar modelo completo primero
config = TrainingConfig(model_type="transformer", hidden_dim=256, num_layers=8)
trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader, max_epochs=5)

# Dividir en bloques
splitter = BlockSplitter(model)
blocks = splitter.split_model(block_size=2)

# Entrenar bloques individualmente
block_trainer = BlockTrainer()
for i, block in enumerate(blocks):
    print(f"Entrenando bloque {i}...")
    block_trainer.train_block(block, train_loader, epochs=3)
    block_trainer.save_block(block, f"bloque_{i}.pt")
```

### Entrenamiento con Constitutional AI

```python
from tenIII import Trainer, TrainingConfig
from tenIII.constitution import ConstitutionManager, HonestPolicy, HelpfulnessPolicy

# Crear constitución
manager = ConstitutionManager()
manager.register_policy(HonestPolicy(threshold=0.8), "honesty")
manager.register_policy(HelpfulnessPolicy(threshold=0.7), "helpfulness")

constitution = manager.create_constitution(
    name="helpful_honest",
    policy_names=["honesty", "helpfulness"],
    weights={"honesty": 0.5, "helpfulness": 0.5}
)

# Entrenar con constitución
config = TrainingConfig(
    model_type="transformer",
    constitution_name="helpful_honest",
    continuous_training=True,
)

trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader)
```

---

## Integración de LLMs

### ¿Por qué integrar LLMs en entrenamiento?

Los LLMs pueden:
1. **Generar datos sintéticos** para aumentar dataset
2. **Evaluar calidad** de respuestas del modelo
3. **Proporcionar feedback** durante entrenamiento (RL)
4. **Guiar razonamiento** del modelo (pensamiento distribuido)

### Opción 1: LLM Local (Ollama, llama.cpp)

#### Instalación

```bash
# Descargar Ollama desde https://ollama.ai
# O usar llama.cpp

# Ejecutar modelo local
ollama run llama2:7b
# O
./main -m modelo.gguf -p "prompt"
```

#### Integración en tenIII

```python
from tenIII.llm import LocalLLMClient
from tenIII import Trainer, TrainingConfig

# Conectar con LLM local
llm_client = LocalLLMClient(
    model_name="llama2:7b",
    base_url="http://localhost:11434",  # Ollama default
    # O para llama.cpp:
    # base_url="http://localhost:8000"
)

# Usar en entrenamiento
config = TrainingConfig(
    model_type="transformer",
    llm_client=llm_client,
    llm_mode="evaluation",  # "evaluation", "generation", "feedback", "reasoning"
)

trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader)

# El modelo usará el LLM local para evaluar respuestas
```

### Opción 2: LLM por API (OpenAI, Anthropic, etc.)

#### Instalación

```bash
pip install tenIII[llm-api]
```

#### Integración

```python
from tenIII.llm import APILLMClient
from tenIII import Trainer, TrainingConfig

# Conectar con API
llm_client = APILLMClient(
    provider="openai",  # "openai", "anthropic", "cohere", "custom"
    api_key="sk-...",
    model_name="gpt-4",
)

# O Anthropic
llm_client = APILLMClient(
    provider="anthropic",
    api_key="sk-ant-...",
    model_name="claude-3-opus",
)

# Usar en entrenamiento
config = TrainingConfig(
    model_type="transformer",
    llm_client=llm_client,
    llm_mode="feedback",  # Feedback del LLM durante RL
)

trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader)
```

### Opción 3: LLM Híbrido (Local + API)

```python
from tenIII.llm import HybridLLMClient
from tenIII import Trainer, TrainingConfig

# Usar local para evaluación rápida, API para razonamiento profundo
llm_client = HybridLLMClient(
    local_model="llama2:7b",
    local_base_url="http://localhost:11434",
    api_provider="openai",
    api_key="sk-...",
    api_model="gpt-4",
    strategy="fallback",  # "fallback", "ensemble", "routing"
)

config = TrainingConfig(
    model_type="transformer",
    llm_client=llm_client,
    llm_mode="reasoning",  # Razonamiento con pensamiento distribuido
)

trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader)
```

### Modos de Uso de LLM

#### 1. Evaluation (Evaluación de Respuestas)

```python
config = TrainingConfig(
    llm_mode="evaluation",
    llm_eval_prompt="Evalúa la calidad de esta respuesta en escala 0-10: {response}",
    llm_eval_threshold=7.0,  # Solo acepta respuestas > 7
)

# El LLM evalúa cada respuesta del modelo durante entrenamiento
```

#### 2. Generation (Generación de Datos)

```python
config = TrainingConfig(
    llm_mode="generation",
    llm_generation_prompt="Genera una pregunta sobre {topic}",
    llm_num_samples=100,  # Generar 100 ejemplos
)

# El LLM genera datos sintéticos para aumentar dataset
```

#### 3. Feedback (Retroalimentación en RL)

```python
config = TrainingConfig(
    llm_mode="feedback",
    training_mode=TrainingMode.REINFORCEMENT_LEARNING,
    llm_reward_prompt="¿Es esta respuesta correcta? Responde con puntuación 0-1: {response}",
)

# El LLM proporciona recompensas para RL
```

#### 4. Reasoning (Razonamiento Distribuido)

```python
config = TrainingConfig(
    llm_mode="reasoning",
    llm_reasoning_steps=5,  # Pasos de razonamiento
    llm_reasoning_prompt="Piensa paso a paso: {question}",
)

# El LLM genera pasos de razonamiento que guían al modelo
```

---

## Framework de Entrenamiento

### Arquitectura General

```
┌─────────────────────────────────────────────────────┐
│           Framework de Entrenamiento                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  Datos       │  │  Modelo      │  │  LLM     │ │
│  │  (Local)     │  │  (Local)     │  │  (Local  │ │
│  │              │  │              │  │   o API) │ │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘ │
│         │                 │                │       │
│         └─────────────────┼────────────────┘       │
│                           │                        │
│                    ┌──────▼──────┐                 │
│                    │  Trainer    │                 │
│                    │  (tenIII)   │                 │
│                    └──────┬──────┘                 │
│                           │                        │
│         ┌─────────────────┼─────────────────┐     │
│         │                 │                 │     │
│    ┌────▼────┐    ┌──────▼──────┐   ┌─────▼──┐  │
│    │ Forward │    │ Evaluation  │   │ Update │  │
│    │ Pass    │    │ (LLM)       │   │ Weights│  │
│    └────┬────┘    └──────┬──────┘   └─────┬──┘  │
│         │                │                 │     │
│         └────────────────┼─────────────────┘     │
│                          │                       │
│                   ┌──────▼──────┐                │
│                   │ Checkpoint  │                │
│                   │ & Logging   │                │
│                   └─────────────┘                │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Flujo de Entrenamiento con LLM

```python
# 1. Cargar datos
train_loader = DataLoader(train_dataset, batch_size=32)

# 2. Crear LLM client
llm_client = LocalLLMClient(model_name="llama2:7b")

# 3. Crear config
config = TrainingConfig(
    model_type="transformer",
    llm_client=llm_client,
    llm_mode="feedback",
    continuous_training=True,
)

# 4. Crear trainer
trainer = Trainer(config)
model = trainer.build_model()

# 5. Entrenar (bucle automático)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    max_epochs=100,
)

# 6. Analizar resultados
print(f"Pérdida final: {history['losses'][-1]:.4f}")
print(f"Razón de parada: {history['early_stop_reason']}")

# 7. Guardar
trainer.save("modelo_con_llm.pt")
```

### Integración Personalizada

Si necesitas control total sobre cómo se usa el LLM:

```python
from tenIII import Trainer, TrainingConfig
from tenIII.llm import LocalLLMClient

class CustomTrainer(Trainer):
    def __init__(self, config, llm_client):
        super().__init__(config)
        self.llm_client = llm_client
    
    def custom_training_step(self, batch):
        """Paso de entrenamiento personalizado"""
        x, y = batch
        
        # Forward pass
        logits = self.model(x)
        
        # Usar LLM para evaluar
        response = self.model.generate(x[0])
        llm_score = self.llm_client.evaluate(response)
        
        # Combinar pérdida estándar con puntuación LLM
        loss = self.criterion(logits, y)
        llm_loss = 1.0 - llm_score
        total_loss = 0.7 * loss + 0.3 * llm_loss
        
        return total_loss

# Usar
config = TrainingConfig(model_type="transformer")
llm_client = LocalLLMClient(model_name="llama2:7b")
trainer = CustomTrainer(config, llm_client)
history = trainer.train(train_loader, val_loader)
```

---

## Ejemplos Prácticos

### Ejemplo 1: Entrenamiento Básico

```python
from tenIII import Trainer, TrainingConfig
import torch
from torch.utils.data import DataLoader, TensorDataset

# Datos
X = torch.randn(1000, 20)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

# Configurar
config = TrainingConfig(
    model_type="transformer",
    hidden_dim=128,
    num_layers=4,
    learning_rate=1e-3,
)

# Entrenar
trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(loader, loader, max_epochs=20)

print(f"✓ Entrenamiento completado")
print(f"  Pérdida final: {history['losses'][-1]:.4f}")
```

### Ejemplo 2: Fine-tuning con LLM Local

```python
from tenIII import Trainer, TrainingConfig, TrainingMode
from tenIII.llm import LocalLLMClient

# LLM local
llm_client = LocalLLMClient(
    model_name="mistral:7b",
    base_url="http://localhost:11434"
)

# Config para fine-tuning
config = TrainingConfig(
    model_type="transformer",
    training_mode=TrainingMode.FINE_TUNING,
    freeze_layers=[0, 1, 2],  # Congelar primeras 3 capas
    llm_client=llm_client,
    llm_mode="evaluation",
)

# Entrenar
trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader)
trainer.save("modelo_fine_tuned.pt")
```

### Ejemplo 3: Entrenamiento con RL y LLM API

```python
from tenIII import Trainer, TrainingConfig, TrainingMode
from tenIII.llm import APILLMClient

# LLM por API
llm_client = APILLMClient(
    provider="openai",
    api_key="sk-...",
    model_name="gpt-4"
)

# Config para RL
config = TrainingConfig(
    model_type="transformer",
    training_mode=TrainingMode.REINFORCEMENT_LEARNING,
    llm_client=llm_client,
    llm_mode="feedback",
    llm_reward_prompt="¿Es correcta esta respuesta? (0-1): {response}",
)

# Entrenar
trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader, max_epochs=50)
trainer.save("modelo_rl.pt")
```

### Ejemplo 4: Entrenamiento Continuo con Bloques

```python
from tenIII import Trainer, TrainingConfig
from tenIII.blocks import BlockSplitter, BlockTrainer
from tenIII.llm import LocalLLMClient

# Entrenar modelo base
config = TrainingConfig(model_type="transformer", hidden_dim=256, num_layers=8)
trainer = Trainer(config)
model = trainer.build_model()
history = trainer.train(train_loader, val_loader, max_epochs=10)

# Dividir en bloques
splitter = BlockSplitter(model)
blocks = splitter.split_model(block_size=2)

# LLM para evaluación de bloques
llm_client = LocalLLMClient(model_name="llama2:7b")

# Entrenar cada bloque
block_trainer = BlockTrainer(llm_client=llm_client)
for i, block in enumerate(blocks):
    print(f"\n📦 Entrenando bloque {i}...")
    block_trainer.train_block(
        block=block,
        train_loader=train_loader,
        epochs=5,
        llm_eval=True,  # Evaluar con LLM
    )
    block_trainer.save_block(block, f"bloque_{i}.pt")

print("\n✓ Todos los bloques entrenados")
```

---

## Resolución de Problemas

### Problema: "ModuleNotFoundError: No module named 'tenIII'"

**Solución:**
```bash
pip install tenIII
# O si está descargado:
pip install /ruta/a/tenIII_PyPI_v3.0.1.zip
```

### Problema: "CUDA out of memory"

**Solución:**
```python
config = TrainingConfig(
    batch_size=8,  # Reducir batch size
    device="cpu",  # O usar CPU
    gradient_checkpointing=True,  # Reducir memoria
)
```

### Problema: "LLM client connection refused"

**Solución:**
```bash
# Asegurar que Ollama está corriendo
ollama serve

# O en otra terminal
ollama run llama2:7b

# Verificar conexión
curl http://localhost:11434/api/tags
```

### Problema: "Early stopping muy rápido"

**Solución:**
```python
config = TrainingConfig(
    early_stopping_mode=EarlyStoppingMode.IMPROVED,  # Cambiar a IMPROVED
    early_stopping_patience=20,  # Aumentar paciencia
    min_delta=0.001,  # Aumentar umbral mínimo
)
```

### Problema: "LLM API muy lento"

**Solución:**
```python
# Usar LLM local para evaluación rápida
llm_client = HybridLLMClient(
    local_model="mistral:7b",
    api_provider="openai",
    strategy="routing",  # Usar local para tareas rápidas
)
```

---

## Referencia Rápida de Comandos

```python
# Importar
from tenIII import Trainer, TrainingConfig, EarlyStoppingMode, TrainingMode
from tenIII.blocks import BlockSplitter, BlockTrainer
from tenIII.constitution import ConstitutionManager
from tenIII.multimodal import VoiceInferenceEngine
from tenIII.llm import LocalLLMClient, APILLMClient, HybridLLMClient

# Crear config
config = TrainingConfig(model_type="transformer", hidden_dim=768)

# Crear trainer
trainer = Trainer(config)

# Construir modelo
model = trainer.build_model()

# Entrenar
history = trainer.train(train_loader, val_loader)

# Guardar/Cargar
trainer.save("modelo.pt")
trainer.load("modelo.pt")

# Bloques
splitter = BlockSplitter(model)
blocks = splitter.split_model(block_size=2)

# Constitutional AI
manager = ConstitutionManager()
manager.register_policy(HonestPolicy(), "honesty")

# LLM
llm_client = LocalLLMClient(model_name="llama2:7b")

# Voz
engine = VoiceInferenceEngine(model)
audio = engine.generate_with_voice("¿Hola?")
```

---

## Soporte y Comunidad

- **Documentación:** http://terminator.com.es
- **GitHub:** https://github.com/yoqer/tenIII
- **Issues:** https://github.com/yoqer/tenIII/issues
- **Email:** AI@terminator.com.es

---

**tenIII v3.0.1 — Ecosistema Gotham City**  
*Framework Avanzado de Entrenamiento de Deep Learning*
