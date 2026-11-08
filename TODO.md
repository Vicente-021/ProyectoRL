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
- [ ] Si se necesita una figura para la presentación, conseguir una captura/mapa
  visual cargado del punto de inicio; no bloquea el piloto del wrapper.
- [ ] Verificar que el jugador aparece en el Nether en la entrada tras varios resets.
- [ ] Revisar los bloques 3×3×3, la transformación a observación egocéntrica y
  frame stacking en la fortaleza real.
- [ ] Refinar la calibración en varias zonas de la fortaleza y probar escaleras,
  colisiones y desplazamiento seguro; los valores actuales son piloto.
- [ ] Verificar muerte por lava/caída y timeout dentro de CraftGround.
- [ ] Medir demora de arranque/reset, pasos por segundo y longitud razonable de episodio.
- [ ] Decidir si CraftGround ofrece throughput suficiente para entrenar directamente;
  si no, activar el simulador propio como respaldo.

## Entrenamiento y evaluación

- [ ] Preparar mundos/seeds y separar por seed: 70% train, 15% validación, 15% test.
- [ ] Implementar y evaluar baseline Random en CraftGround.
- [ ] Implementar DQN propio en PyTorch.
- [ ] Entrenar PPO de Stable-Baselines3.
- [ ] Ejecutar DQN de SB3 como referencia para validar el DQN propio.
- [ ] Mantener iguales observación, acciones, presupuesto de interacciones y seeds.
- [ ] Evaluar tasa de éxito, pasos a la meta, variabilidad, curvas de aprendizaje y
  generalización en seeds reservadas.
- [ ] Si la recompensa terminal no produce aprendizaje, probar *reward shaping* como
  contingencia/ablación.
- [ ] Si hace falta, implementar simulador Gymnasium compacto como respaldo.
- [ ] Documentar resultados, limitaciones y reproducibilidad.

## Pendiente para continuar

La seed, el AABB de lava well room y un spawn candidato están identificados en
los datos del mundo. Falta confirmar visualmente el spawn antes de fijarlo como
inicio oficial de los episodios.
