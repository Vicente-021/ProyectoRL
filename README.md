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
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    NetherFortressEnv.ACTION_FORWARD
)
```

Las entradas y coordenadas objetivo todavía deben prepararse para cada seed.
Las pruebas unitarias del wrapper se ejecutan con:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
