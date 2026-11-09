# Proyecto RL: Navegación hacia la lava well room de una Nether Fortress

**Curso:** Aprendizaje por Refuerzo
**Integrantes:** Vicente Hernández · Vicente Wiederhold
**Periodo:** Proyecto semestral 2026

---

## 1. Resumen ejecutivo

El proyecto propone entrenar y comparar agentes de **Aprendizaje por Refuerzo
directamente en Minecraft Java 1.21.0** para navegar desde una entrada exterior
conocida de una Nether Fortress hasta la **lava well room**. Se evaluará si
generalizan a fortalezas/seeds no vistas, en vez de memorizar un mapa.

Comparamos una **baseline aleatoria**, un **DQN implementado por nosotros** y un
**PPO de Stable-Baselines3**; además usamos **DQN de SB3** como referencia para
validar nuestra implementación. El entorno principal es **CraftGround** con un
wrapper Gymnasium propio. Un simulador ligero se mantiene como respaldo si el
entrenamiento directo no ofrece suficiente estabilidad o throughput.

---

## 2. Motivación

- En el speedrun, una parte del *Nether split* es recorrer la fortaleza hasta la
  **lava well room**, que conecta las secciones interiores y exteriores.
- El alcance inicial **no incluye localizar la fortaleza en el Nether**: el
  episodio comienza en una entrada exterior conocida.
- Es una tarea humana concreta, verificable y con **estructura procedural
  determinista por seed**, lo que la vuelve un excelente banco de pruebas de RL.
- Interés científico: ¿puede RL aprender una **política de navegación que capture
  las reglas de generación** y generalice, en lugar de memorizar? ¿Qué algoritmo
  lo hace mejor con presupuesto limitado?

---

## 3. Definición del problema

> ¿Puede un agente entrenado directamente en Minecraft navegar desde una entrada
> exterior conocida hasta la *lava well room* y generalizar a seeds no vistas?

Formulado como un **POMDP** (decisión parcialmente observable):

| Elemento | Definición |
|---|---|
| **Observación** `o_t` | CraftGround: bloques locales 3×3×3 y frame stacking; RGB disponible como complemento si el piloto lo requiere. Sin mapa completo ni coordenadas de la meta. |
| **Acciones** `a_t` | Macros discretas: avanzar, retroceder, desplazamiento lateral y giro, cada una aplicada durante ticks calibrados. Sin combate; inicialmente sin salto. |
| **Transición** `P` | Minecraft real vía CraftGround; estocástica por la física y el control. |
| **Recompensa** `r_t` | Éxito `+1`; pasos, muerte por lava/caída y límite de pasos `0`. Recompensa dispersa inicial. |
| **Objetivo** | Maximizar `E[Σ γᵗ r_t]`; el episodio termina al llegar a la sala, morir o alcanzar el límite. |

### Hipótesis

- **H1 — Generalización:** la tasa de éxito se mantiene por encima de Random en
  fortalezas de seeds reservadas.
- **H2 — Algoritmos:** a igual presupuesto de interacciones, DQN y PPO difieren en
  eficiencia de muestras y estabilidad; DQN podría ser más eficiente y PPO menos
  variable, pero la comparación determinará el resultado.
- **H3 — Recompensa dispersa:** el agente puede aprender directamente en Minecraft
  con recompensa terminal, o el progreso requerirá una recompensa auxiliar.

### Supuestos clave

- Los mundos se generan en Minecraft Java 1.21.0 y la seed se registra.
- El agente recibe información local; por ello se trata como un **POMDP**. El
  wrapper usa las coordenadas de la meta solo para reward/evaluación, no como
  observación de la política.

---

## 4. Estrategia de entorno: CraftGround como entorno principal

- **Experimento principal:** entrenamiento directo en Minecraft Java 1.21.0
  mediante CraftGround y un wrapper Gymnasium desarrollado por el grupo.
- **Respaldo:** simulador propio compacto para depurar o completar la comparación
  si CraftGround resulta demasiado lento/inestable.
- **Scope:** el episodio empieza en una entrada exterior conocida y termina en
  la lava well room; no incluye buscar la fortaleza desde cualquier punto del Nether.
- **Preparación:** comandos administrativos pueden crear/resetear episodios,
  pero no se exponen a la política como observaciones ni acciones.

### Prueba mínima ya realizada

- Instalado CraftGround 2.7.4 en Python 3.11 con JDK 21.
- CraftGround entregó `pov` RGB y `surrounding_blocks` de 3×3×3 (27 bloques).
- `reset()`, pasos no-op y acciones de avanzar funcionaron en un mundo de prueba.
- El primer wrapper V2 está en `src/fortress_rl/env.py`; la factory CraftGround
  completó un reset y la macro forward cambió la posición del jugador.
- La calibración inicial en el spawn piloto fijó `action_repeat=4`; un giro
  ±90° produjo aproximadamente ±89.9°, y forward recorrió cerca de 0.75 bloques
  por macro en esa ruta. Estos valores aún requieren pruebas en otras piezas.
- Diez pruebas unitarias pasan. Se verificaron cuatro resets consecutivos en la
  fortress: el jugador reapareció en el spawn, yaw 0 y sobre el suelo; `close()`
  dejó cero procesos del cliente.
- Para la seed piloto `12345`, `/locate` devolvió una fortaleza cerca de
  `X=112, Z=672`; el NBT guardado incluye 142 piezas de la fortaleza.
- La pieza `minecraft:nece` marca la lava well room: AABB inclusiva
  `(66,55,668)`–`(78,68,680)`; se verificó lava en `(72,60,674)`.
- Se probó el wrapper en esa fortress con spawn candidato `(95.5,61,736.5)`, yaw
  `0`; avanzó por el corredor y un spawn dentro del AABB dio reward `+1` terminal.
- Baseline aleatorio en 10 episodios, límite de 100 decisiones: 950 decisiones
  ejecutadas, 0/10 éxitos y una muerte al paso 50. Throughput: 5.03 decisiones/s
  durante interacción, 4.92/s steady-state incluyendo resets y 3.69/s end-to-end
  (el arranque inicial tomó 64.2 s). Es una referencia corta, no permite concluir
  que un agente entrenado no pueda aprender.
- Para estabilizar el smoke test se usó `initial_extra_commands=["tick rate 20"]`;
  el tick rate predeterminado de 10.000 produjo retraso extremo en esta máquina.
- La prueba se ejecutó en un equipo que reporta Intel UHD 620 y PyTorch CPU; no
  se detectó NVIDIA/CUDA. Esto no impide el smoke test, pero limita el entrenamiento
  acelerado en ese equipo.

### Descartado explícitamente

| Descartado | Razón |
|---|---|
| Buscar la fortaleza desde el Nether abierto | Fuera del alcance inicial; se empieza dentro/en una entrada conocida |
| Entrenar con recompensa densa desde el inicio | Primero se probará la hipótesis de recompensa terminal dispersa |
| Pie chart / profiler (radar de spawners) | Señal sin dirección, de rango corto y ruidosa; fuera del alcance |
| Muertes por mobs | No controlables sin acción de combate; añaden ruido |

---

## 5. Preparación de mundos y seeds

Para entrenar directamente en Minecraft, el agente interactúa con el mundo
CraftGround; no es necesario cosechar primero una matriz completa para entrenar.
Sí necesitamos un pequeño manifiesto reproducible por seed con versión,
coordenadas de entrada, orientación inicial y coordenadas internas de la meta.
Amulet/cubiomes pueden apoyar la inspección o preparar ese ground truth, pero no
son el entorno de entrenamiento.

La división acordada es **70% entrenamiento, 15% validación y 15% prueba**, por
seed. El conjunto de prueba queda reservado hasta la evaluación final.

---

## 6. Observación y acciones en CraftGround

La política no recibe una etiqueta de sala ni coordenadas absolutas de la meta.
La documentación/instalación probada de CraftGround expone `pov` (RGB) y un
mensaje estructurado con `surrounding_blocks` (3×3×3), posición, yaw/pitch,
`is_dead`, estado de lava y otros campos.

- **Entrada inicial elegida:** usar los bloques cercanos 3×3×3 con frame stacking;
  RGB se evaluará si el contexto local no basta.
- **Privacidad de la meta:** la coordenada de objetivo puede usarse internamente
  para reward y evaluación, pero se excluye de la observación de la política.
- **Acciones:** wrapper Gymnasium con macros discretas de movimiento/rotación y
  duración fija en ticks, calibrada experimentalmente.
- **Control de versión:** mantener Minecraft 1.21.0 y el mismo diccionario de
  bloques/configuración durante train, validación y test.

---

## 7. Recompensa y manejo de la muerte

**Los mobs están desactivados** (`difficulty=peaceful`, `gamerule doMobSpawning false`),
por lo que **no hay muerte por mobs**. La única muerte es **lava/caída**
(controlable con movimiento).

| Evento | Recompensa |
|---|---|
| Pasos intermedios | `0` |
| **Éxito** (entrar en la región objetivo) | `+1` terminal, señal interna del wrapper |
| **Muerte** (lava/caída) | `0` terminal |
| Límite de pasos | `0` y truncación |
| Muerte por mob | **No se modela**; mobs desactivados |

Se empieza deliberadamente con recompensa terminal dispersa. Si no aparece
aprendizaje, se evaluará *reward shaping* como contingencia/ablación y se
documentará como cambio experimental.

---

## 8. Diseño experimental (fijado)

| # | Método | Implementación | Rol |
|---|---|---|---|
| 1 | **Random** | Wrapper propio | Baseline |
| 2 | **DQN propio** | PyTorch puro (off-policy) | Método implementado por el grupo |
| 3 | **PPO** | Stable-Baselines3 (on-policy) | Comparación entre familias |
| 4 | **DQN de SB3** | SB3 | Referencia para validar el DQN propio |

### Regla de comparación justa

- Mismo Minecraft 1.21.0, wrapper, observación y macros discretas.
- Mismas seeds.
- Mismo presupuesto de **timesteps**.
- Arquitecturas comparables según observación (voxels y/o RGB).
- **Mismo esfuerzo de búsqueda de hiperparámetros** para todos.
- Evaluación sobre **seeds held-out** (no vistas en entrenamiento).

El experimento #4 demuestra que nuestra implementación de DQN "reproduce" la
referencia de la librería, lo que aporta rigor.

---

## 9. Evaluación

**Métricas:**
- Tasa de éxito (llegar a la lava well room).
- Pasos hasta la meta.
- Varianza entre seeds.
- **Generalización a seeds no vistas** (prueba clave de H1).
- Curvas de aprendizaje (recompensa media ± desviación).

**Ablaciones candidatas:**
- Bloques locales frente a RGB/combinación, si CraftGround throughput lo permite.
- Recompensa terminal frente a *reward shaping*, si la primera no produce aprendizaje.

**Validez de la prueba de "aprender las reglas":** evaluar en seeds que no
aparecieron durante entrenamiento ni tuning y comparar contra Random.

---

## 10. Stack técnico

- **Lenguaje / DL:** Python 3.11 · PyTorch
- **RL:** Stable-Baselines3 · Gymnasium; DQN propio en PyTorch.
- **Entorno principal:** CraftGround 2.7.4 · Minecraft Java 1.21.0.
- **Runtime:** Python 3.11 · JDK 21 · CMake/OpenGL; Xvfb solo si se ejecuta headless.
- **RL:** PyTorch · Stable-Baselines3. La prueba actual usa PyTorch CPU; se debe
  confirmar hardware CUDA en el equipo destinado a entrenar.
- **Respaldo:** simulador Gymnasium propio; Amulet/cubiomes son herramientas
  auxiliares para inspeccionar seeds, entradas y coordenadas.

---

## 11. Revisión bibliográfica (para la presentación)

| Trabajo | Aporte | Relación con nuestro proyecto |
|---|---|---|
| MineRL (Guss et al., 2019) | Dataset y entorno a escala, navegación con recompensa escasa | Muestra la dificultad; inspira simplificación de acciones |
| MineDojo (Fan et al., 2022) | Agentes abiertos, obs multimodal (voxels) | Justifica observación estructurada en vez de píxeles |
| CraftGround (Yang et al., 2025) | Entorno RL de Minecraft, API tipo Gymnasium, observación/acciones configurables | Entorno principal de entrenamiento directo |
| OpenAI VPT (Baker et al., 2022) | Preentrenamiento desde video | Alternativa de transferencia (contexto) |
| DQN (Mnih et al., 2015) | Value-based off-policy | Método principal propio |
| PPO (Schulman et al., 2017) | Policy-gradient on-policy | Método de comparación |
| Domain randomization (Tobin et al., 2017) | Variación de dominios de entrenamiento | Contexto para generalizar entre seeds |
| Reward shaping (Ng et al., 1999) | Shaping potential-based | Contingencia si la recompensa terminal no basta |

**Gap identificado:** trabajos previos estudian Minecraft/RL y generalización en
entornos procedurales, pero no esta tarea acotada de navegación intra-fortaleza a
la lava well room con comparación Random/DQN/PPO entrenada directamente en
Minecraft Java 1.21.0 y evaluación sobre seeds reservadas.

---

## 12. Outline de la presentación (7 min)

| Slide | Contenido | Ponderación |
|---|---|---|
| S1 | Portada | — |
| S2 | Tema y motivación | 15% |
| S3–S4 | Revisión bibliográfica + gap | 25% |
| S5 | Definición formal del problema (POMDP + hipótesis) | — |
| S6 | Métodos y resultados esperados | 20% |
| S7 | Planificación y asignación | 20% |
| S8 | Cierre | — |

*(Tiempo 10% y respuestas a preguntas 20%.)*

---

## 13. Cronograma (por completar)

| Semana | Hito | Responsable |
|---|---|---|
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

---

## 14. Glosario

- **Lava well room:** sala del pozo de lava que conecta secciones interiores y exteriores de la fortaleza.
- **Grid voxel:** representación 3D por celdas de bloques.
- **Observable:** información reconstruible en el juego real (bloques, pose).
- **Privilegiado:** información de ground-truth (etiqueta de sala), no visible por un jugador.
- **Ground truth:** coordenadas internas de la meta usadas para recompensa/evaluación, no observadas por la política.
- **Frame stacking:** entregar a la política varias observaciones recientes como memoria corta.
- **Reward shaping:** recompensa auxiliar que podría probarse si la recompensa terminal dispersa no permite aprender.
