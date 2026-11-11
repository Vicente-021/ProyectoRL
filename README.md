# ProyectoRL

Agente de RL para navegar desde una entrada conocida de una Nether Fortress
hacia la lava well room. El entorno principal es Minecraft Java 1.21.0 mediante
CraftGround; el simulador propio se reserva como respaldo.

## Wrapper inicial

El wrapper `fortress_rl.NetherFortressEnv` convierte CraftGround en un entorno
Gymnasium con seis macros discretas, observación local de bloques (one-hot),
frame stacking y recompensa terminal dispersa. La coordenada de meta solo se
usa internamente para recompensa/evaluación, no se entrega a la política.

La máquina que crea CraftGround debe incluir `tick rate 20` en sus comandos de
inicialización. Los smoke tests del proyecto encontraron que el tick rate
predeterminado de 10.000 puede producir retrasos severos en este equipo.

Ejemplo de creación (usar coordenadas medidas para una fortress real de la
misma seed y versión):

```python
from fortress_rl import (
    GoalBounds,
    NetherFortressEnv,
    SpawnPoint,
    make_craftground_env,
)

env = make_craftground_env(
    world_seed=seed,
    goal_bounds=GoalBounds((xmin, ymin, zmin), (xmax, ymax, zmax)),
    spawn_points=[SpawnPoint(x, y, z, yaw=90)],
)
try:
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(
        NetherFortressEnv.ACTION_FORWARD
    )
finally:
    env.close()
```

`close()` termina CraftGround y limpia también el cliente/Gradle asociado a esta
instancia, sin buscar ni cerrar otras instancias de Minecraft. Las entradas y
coordenadas objetivo todavía deben prepararse para cada seed.
Las pruebas unitarias del wrapper se ejecutan con:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Baseline aleatorio y benchmark

El runner ejecuta acciones uniformemente aleatorias y reporta tasa de éxito,
duración de episodios, acciones por tipo y decisiones por segundo. El seed
controla tanto la secuencia aleatoria como los resets. Con el entorno
`craftground-mc121` activo, se puede correr un piloto corto en la fortress
aprobada así:

```bash
PYTHONPATH=src python -m fortress_rl.baseline \
  --episodes 5 --max-episode-steps 100 --seed 0 \
  --output results/random_baseline.json
```

Por defecto usa el mundo seed `12345`, spawn `(95.5,61,736.5)` y la región meta
`(66,55,668)`–`(78,68,680)`. Se pueden cambiar con `--world-seed`, `--spawn`,
`--yaw`, `--goal-min` y `--goal-max`. Los informes bajo `results/` se ignoran
en Git; `env.close()` se ejecuta también si el benchmark falla.

## Entrenamiento PPO en Minecraft

El script entrena PPO de Stable-Baselines3 **directamente en Minecraft Java
1.21.0**, lanzado y controlado por CraftGround; no usa el simulador de respaldo.
Durante `learn()`, PPO muestrea estocásticamente una de las seis acciones desde
su distribución categórica; no elige siempre la acción de mayor probabilidad.
La duración de cada macro sigue fija (`action_repeat=4`); lo estocástico es qué
movimiento selecciona la política.
La configuración predeterminada es el mundo piloto documentado: seed `12345`,
entrada `(95.5, 61, 736.5)` mirando al norte (`yaw=0`) y meta
`(66,55,668)`–`(78,68,680)`. Si se cambia la seed, también hay que indicar las
coordenadas de entrada y meta correspondientes a esa seed. El cliente usa por
defecto resolución visible de `640×360`; `--image-size 64 64` la reduce para
priorizar throughput.

En Windows, usar Miniforge/Conda con Python 3.11, JDK 21 y las bibliotecas
nativas necesarias. CraftGround compila su módulo C++ al primer arranque; se
requieren Visual Studio con las herramientas de C++ y CMake 4.2 o superior
(Visual Studio 18 necesita CMake 4.2+).

```bash
conda create -n craftground-mc121 python=3.11 -y
conda activate craftground-mc121
conda install -c conda-forge openjdk=21 cmake=4.4 glew libpng libzlib -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[minecraft,training]"
```

Después de instalar Miniforge o ejecutar `conda init powershell`, cierra y abre
una terminal nueva para que `conda activate` quede disponible.

El primer arranque de CraftGround descarga y compila Minecraft; puede tardar
varios minutos. El runner muestra por defecto el progreso de Gradle. Usa
`--quiet-gradle` solo si quieres ocultarlo.
CraftGround minimiza inicialmente su cliente. En Windows, el runner muestra por
defecto una ventana de vista previa en vivo con el POV que recibe el agente;
usa `--no-preview` para desactivarla. `--hide-window` deja además minimizado el
cliente nativo de CraftGround.
El audio está silenciado por defecto para evitar fallos del backend OpenAL de
Windows; puedes activarlo con `--audio`.

Iniciar el entrenamiento con los valores piloto:

```bash
python -m fortress_rl.train_ppo --total-timesteps 100000
```

Para comprobar rápidamente la integración Minecraft–PPO sin ejecutar 1024 pasos
de rollout, usa una actualización pequeña:

```bash
python -m fortress_rl.train_ppo --total-timesteps 64 --n-steps 64 \
  --batch-size 16 --n-epochs 1 --device cpu
```

También queda disponible el comando `fortress-train-ppo`. Cada ejecución crea
un directorio fechado bajo `runs/` y guarda allí el modelo final, checkpoints,
configuración y registro de episodios; TensorBoard puede abrirse con
`tensorboard --logdir runs`. Para una prueba corta del pipeline se puede usar
`--total-timesteps 2048`. Se puede reanudar desde un checkpoint con
`--resume-from ruta/al/checkpoint.zip`; en ese caso, `--total-timesteps` indica
cuántas acciones nuevas entrenar. La recompensa actual es dispersa (solo se
recibe al llegar a la sala), así que una ejecución corta comprueba la
integración, no que el agente ya haya aprendido la ruta.

Para el piloto PPO ajustado, un comando de ejemplo es:

```bash
python -m fortress_rl.train_ppo --total-timesteps 20000 \
  --gamma 0.995 --gae-lambda 0.98 --ent-coef 0.03 --learning-rate 0.0001
```

## Piloto DQN de Stable-Baselines3

Para comparar otra familia de algoritmos en el mismo entorno y con la misma
recompensa sparse:

```bash
python -m fortress_rl.train_dqn_sb3 --total-timesteps 20000
```

El piloto informa también la mayor reducción de distancia a la AABB de la meta.
Es un proxy de progreso, no la longitud exacta del camino. Si no alcanza el 25%,
el siguiente experimento será una ablación separada con reward shaping:

```bash
python -m fortress_rl.train_dqn_sb3 --total-timesteps 20000 --reward-shaping
```

El shaping potencial solo usa posición y meta para modificar la recompensa; no
añade esas coordenadas a la observación de la política.
