# Hermes Action Transducer

Repo autonome pour une architecture:

`Hermes -> ActionIR -> Compiler -> robot runtime`

But:

- garder Hermes libre au niveau cognition
- court-circuiter l'obligation de verbaliser toute l'action en texte
- compiler un etat d'action riche vers une surface runtime robot

Ce repo ne depend pas de `dimos-v0`.

## Pieces

- `HermesEncoder`: produit un etat Hermes compact a partir d'une observation
- `ActionTransducer`: convertit l'etat Hermes en `ActionIR`
- `EmbodimentCompiler`: compile `ActionIR` vers une commande runtime
- `RobotProfile`: specialise la compilation pour `arm`, `go2`, `g1`

## Action IR

`ActionIR` melange:

- structure explicite
- cibles et contraintes
- prior de skills
- latents continus simples

Le texte reste utile pour debug, mais n'est plus l'interface finale de l'action.

## Lancer

```bash
cd /Users/robertbadinter/Downloads/latent-relay/separate-repos/hermes-action-transducer
python scripts/run_transducer.py --task "Pick up the mug and place it on the coaster" --robot-profile arm
```

## Hermes reel via Hugging Face

La `v1` peut maintenant utiliser un vrai encodeur Hermes en inference-only pour extraire des hidden states compacts.

Installe d'abord les dependances:

```bash
pip install -e ".[hermes]"
```

Extraction simple d'un `HermesState`:

```bash
python scripts/extract_hermes_state.py \
  --task "Pick up the mug and place it on the coaster" \
  --state-text "mug left of gripper; coaster on right" \
  --robot-profile arm \
  --model-id NousResearch/Hermes-4.3-36B
```

Tu peux aussi utiliser directement le pipeline avec le vrai encodeur:

```bash
python scripts/run_transducer.py \
  --task "Pick up the mug and place it on the coaster" \
  --state-text "mug left of gripper; coaster on right" \
  --robot-profile arm \
  --encoder-backend hermes_hf \
  --model-id NousResearch/Hermes-4.3-36B
```

Note:

- le modele cible par defaut est `NousResearch/Hermes-4.3-36B`
- l'encodeur extrait les `hidden_states` et les compresse en `thought_vector` et `intent_vector`
- cette etape est inference-only pour l'instant, pas encore du training sur hidden states Hermes

## Entrainement supervise

Bootstrappe d'abord un petit dataset de depart:

```bash
cd /Users/robertbadinter/Downloads/latent-relay/separate-repos/hermes-action-transducer
python scripts/bootstrap_dataset.py
```

Puis entraine un premier transducteur appris:

```bash
python scripts/train_supervised.py \
  --dataset data/bootstrap_train.jsonl \
  --checkpoint checkpoints/action_ir.pt \
  --epochs 20
```

Evaluation:

```bash
python scripts/eval_supervised.py \
  --dataset data/bootstrap_train.jsonl \
  --checkpoint checkpoints/action_ir.pt
```

Sur RunPod, le plus simple est d'utiliser directement:

```bash
bash scripts/runpod_train.sh
```

Ce v1 d'entrainement apprend deja:

- `mode`
- `tool prior principal`
- `horizon`
- `speed`
- `confidence`
- `caution`
- `force_limit`
- `motion_latent`
- `safety_latent`

Note:

- les modules d'entrainement PyTorch sont separes du package principal pour eviter de casser l'inference minimale
- dans mon sandbox local, l'import `torch` plante au niveau OpenMP, donc j'ai pu ajouter le code de train mais pas executer le training ici

## Datasets dedies

Les seeds hardcodes ne sont qu'un bootstrap minimal. Le chemin plus serieux est de convertir des datasets robotiques dedies vers le format supervise du repo.

La voie recommandee ici est le format LeRobot/Hugging Face, qui couvre bien:

- `lerobot/droid_100`
- `lerobot/droid_1.0.1`
- `ZibinDong/bridgedatav2_train`

Installe d'abord les dependances de conversion:

```bash
pip install -e ".[convert]"
```

Puis convertis un dataset vers notre JSONL supervise:

```bash
python scripts/convert_hf_dataset.py \
  --dataset-id lerobot/droid_100 \
  --split train \
  --robot-profile arm \
  --out data/droid_100_supervised.jsonl \
  --max-rows 5000 \
  --max-episodes 200
```

Exemple BridgeData V2:

```bash
python scripts/convert_hf_dataset.py \
  --dataset-id ZibinDong/bridgedatav2_train \
  --split train \
  --robot-profile arm \
  --out data/bridgedatav2_supervised.jsonl \
  --max-rows 10000 \
  --max-episodes 300
```

Si le dataset expose seulement `task_index`, tu peux aussi fournir un mapping `tasks.jsonl` via `--tasks-jsonl`.

Tu peux aussi enchaîner directement conversion + train + eval:

```bash
HF_DATASET_ID=lerobot/droid_100 \
ROBOT_PROFILE=arm \
DATASET_PATH=data/droid_100_supervised.jsonl \
CHECKPOINT_PATH=checkpoints/droid_100_action_ir.pt \
bash scripts/convert_and_train.sh
```

Et sur RunPod, [runpod_train.sh](/Users/robertbadinter/Downloads/latent-relay/separate-repos/hermes-action-transducer/scripts/runpod_train.sh) sait maintenant faire la conversion automatiquement si tu lui passes `HF_DATASET_ID`:

```bash
HF_DATASET_ID=lerobot/droid_100 \
ROBOT_PROFILE=arm \
DATASET_PATH=/workspace/data/droid_100_supervised.jsonl \
CHECKPOINT_PATH=/workspace/checkpoints/droid_100_action_ir.pt \
bash scripts/runpod_train.sh
```

## Tests

```bash
cd /Users/robertbadinter/Downloads/latent-relay/separate-repos/hermes-action-transducer
pytest -q
```
