# To-do — Navegación RL en Nether Fortress

Proyecto: Minecraft Java 1.21.0 + CraftGround
Integrantes: Vicente Hernández y Vicente Wiederhold

## Avance actual

- [x] Instalar Miniforge y crear el entorno `craftground-mc121` (Python 3.11, JDK 21).
- [x] Instalar CraftGround 2.7.4 y PyTorch CPU.
- [x] Comprobar que CraftGround puede devolver RGB y una observación estructurada
  con 27 bloques cercanos (3×3×3).
- [x] Ejecutar `reset()` y acciones no-op/forward en un mundo de prueba.
- [x] Usar `initial_extra_commands=["tick rate 20"]` para evitar el retraso visto
  con el tick rate predeterminado de 10.000.
- [x] Crear el primer wrapper Gymnasium en `src/fortress_rl/env.py`.
- [x] Pasar 5 pruebas unitarias y `gymnasium.check_env`.
- [x] Probar la factory del wrapper con CraftGround real y verificar que forward
  cambia la posición del jugador.

## Próximos pasos — Hito 2: prueba de factibilidad

- [x] Elegir seed piloto de Minecraft Java 1.21.0: `12345`.
- [x] Consultar `/locate` en el Nether: referencia de Nether Fortress en
  `X=112, Z=672` (chunk de inicio `7,42`).
- [x] Leer los datos de la estructura del mundo guardado con Amulet-core y
  confirmar la seed `12345` y la versión 1.21.0.
- [x] Identificar la lava well room por pieza de estructura `minecraft:nece`:
  AABB inclusiva `(66,55,668)`–`(78,68,680)`; el bloque de lava detectado está en
  `(72,60,674)`.
- [x] Identificar una entrada exterior candidata en pieza `minecraft:nebef`,
  AABB `(93,56,731)`–`(97,65,738)`, conectada a un corredor `minecraft:nesc`.
  Spawn piloto propuesto: `(95.5,61,736.5)`, yaw `0` (hacia `+Z`/interior).
- [x] Probar el spawn candidato con el wrapper: teleport a
  `(95.5,61,736.5)`, `is_on_ground` y ocho macros forward con
  `action_repeat=4` avanzaron por el corredor.
- [x] Probar la región objetivo con el wrapper: un spawn seguro dentro del AABB
  produjo reward `+1` y terminación exitosa.
- [x] Confirmar en el guardado de prueba que `difficulty=peaceful` y
  `doMobSpawning=false` quedaron aplicados.
- [x] Aprobado por el grupo como spawn piloto: `(95.5,61,736.5)`, yaw `0`.
- [x] Calibración inicial: `action_repeat=4`; giro ±90° produjo yaw ±89.9°.
  Forward avanzó ~2.98 bloques en 4 macros (~0.75 por macro); strafing avanzó
  ~1.2–1.5 bloques en 2 macros.
- [x] Se observó que retroceder desde el extremo del puente puede llevar a una
  caída; se mantiene la acción y se tratará la muerte/caída como fallo terminal.
- [x] Verificar cuatro `reset()` consecutivos en la fortress: mismo spawn, yaw 0,
  jugador en el suelo; `close()` dejó cero procesos de esa instancia.
- [x] Verificar ciclo de éxito real: reward `+1`/`terminated=True`, luego `reset()`
  al spawn piloto y otro `step()` válido.
- [x] Verificar ciclo tras muerte: el episodio aleatorio 5 terminó por muerte al
  paso 50 y el episodio 6 hizo reset y continuó hasta el límite.
- [ ] Si se necesita una figura para la presentación, conseguir una captura/mapa
  visual cargado del punto de inicio; no bloquea el piloto del wrapper.
- [x] Verificar cuatro resets reales en la entrada: posición, yaw y suelo válidos.
- [ ] Revisar los bloques 3×3×3, la transformación a observación egocéntrica y
  frame stacking en la fortaleza real.
- [ ] Refinar la calibración en varias zonas de la fortaleza y probar escaleras,
  colisiones y desplazamiento seguro; los valores actuales son piloto.
- [ ] Verificar muerte por lava y timeout dentro de CraftGround (la muerte normal
  y el reset posterior ya se comprobaron).
- [x] Medición CraftGround: 10 episodios, límite 100 decisiones (950 ejecutadas);
  arranque inicial 64.2 s, resets posteriores 0.05–0.90 s, throughput de
  interacciones 5.03 decisiones/s, steady state 4.92/s, end-to-end 3.69/s.
- [ ] Decidir si CraftGround ofrece throughput suficiente para entrenar directamente;
  si no, activar el simulador propio como respaldo.

## Entrenamiento y evaluación

- [ ] Preparar mundos/seeds y separar por seed: 70% train, 15% validación, 15% test.
- [x] Implementar runner reproducible de baseline aleatorio con métricas de
  éxito, duración, distribución de acciones y throughput.
- [x] Ejecutar baseline aleatorio: 0/10 éxitos; 1 episodio terminó en muerte al
  paso 50 y los otros 9 llegaron al límite de 100. Esto es una referencia, no
  evidencia estadística de que el agente entrenado no pueda aprender.
- [ ] Decidir si el throughput medido permite entrenar directamente y fijar el
  presupuesto de interacciones.
- [ ] Implementar runner de PPO (Stable-Baselines3) con Monitor/logs, evaluación
  periódica, checkpoints y cierre limpio del entorno; hacer un smoke test corto.
- [ ] Definir presupuesto y criterio de parada del piloto PPO. Referencia inicial:
  10.000 decisiones ≈34 min a 4.92 decisiones/s steady-state, más ~64 s del primer
  arranque; revisar la estimación durante el entrenamiento.
- [ ] Implementar DQN propio en PyTorch.
- [x] Implementar runner PPO de Stable-Baselines3 sobre CraftGround, con checkpoints.
- [x] Preparar runner piloto DQN de Stable-Baselines3 con medición de progreso.
- [x] Ejecutar piloto DQN SB3 sparse (19,571 pasos, seed `12345`): 0 éxitos,
  mejor reducción de distancia a la meta 7% (proxy, no ruta real).
- [x] Probar DQN SB3 con shaping potencial (20k pasos, seed `12345`): 0 éxitos,
  mejor reducción de distancia 31% (proxy; no equivale al 31% del camino real).
- [x] Ejecutar piloto PPO SB3 en Minecraft (103,569 pasos acumulados, seed `12345`):
  0 éxitos con recompensa terminal dispersa; requiere análisis/recompensa alternativa.
- [x] Probar PPO ajustado (20,480 pasos; `gamma=0.995`, `gae_lambda=0.98`,
  `ent_coef=0.03`, `learning_rate=1e-4`): 0 éxitos con recompensa sparse.
- [x] Probar reward shaping potencial como ablación separada.
- [ ] Ejecutar DQN de SB3 como referencia para validar el DQN propio.
- [ ] Mantener iguales observación, acciones, presupuesto de interacciones y seeds.
- [ ] Evaluar tasa de éxito, pasos a la meta, variabilidad, curvas de aprendizaje y
  generalización en seeds reservadas.
- [ ] Si la recompensa terminal no produce aprendizaje, probar *reward shaping* como
  contingencia/ablación.
- [ ] Si hace falta, implementar simulador Gymnasium compacto como respaldo.
- [ ] Documentar resultados, limitaciones y reproducibilidad.

## Pendiente para continuar

La seed, el AABB de lava well room y el spawn piloto están identificados y
aprobados. El ciclo terminal→reset se verificó para éxito y muerte. Falta
implementar el runner PPO, fijar su presupuesto y verificar lava/timeout.
