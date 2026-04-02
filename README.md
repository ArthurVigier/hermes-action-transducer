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

Tu peux choisir trois formes de features Hermes:

- `vanilla`: baseline sans latents Hermes, seulement observation/task/profile/proprio
- `compact`: seulement la forme 8+8 classique
- `rich`: `8+8` + `hidden_projection` + resume agrege des couches
- `per_layer`: `8+8` + `hidden_projection` + concat explicite des projections par couche
- `full`: `8+8` + `hidden_projection` + resume agrege + projections explicites par couche

Exemple:

```bash
python scripts/train_supervised.py \
  --dataset data/droid_100_hermes_supervised.jsonl \
  --checkpoint checkpoints/droid_100_hermes_per_layer.pt \
  --feature-mode per_layer \
  --rich-projection-dim 32 \
  --per-layer-projection-dim 64 \
  --max-layer-projections 3 \
  --epochs 20 \
  --device cuda
```

## Benchmark

Le repo sait aussi lancer:

- un benchmark complet: `vanilla`, `compact`, `rich`, `per_layer`, `full`
- un benchmark par paires: par exemple `vanilla,full`
- avec mesures de latence d'inference:
  - `avg_feature_build_latency_ms`
  - `avg_model_forward_latency_ms`
  - `avg_end_to_end_latency_ms`
  - `p50_end_to_end_latency_ms`
  - `p95_end_to_end_latency_ms`
- un `layer_sweep` pour comparer `vanilla` vs `compact` couche par couche

Script direct:

```bash
python scripts/benchmark_feature_modes.py \
  --dataset data/droid_100_hermes_supervised.jsonl \
  --checkpoint-dir benchmarks/checkpoints \
  --results-out benchmarks/results.json \
  --benchmark-mode complete \
  --epochs 20 \
  --latency-samples 64 \
  --latency-warmup 5 \
  --device cuda
```

## Evaluation CaP-X

J'ai ajoute une integration dediee a l'eval officielle `CaP-X / CaP-Bench`.

Important:

- cette integration lance le vrai runner officiel `capx/envs/launch.py`
- elle parse ensuite le resume officiel `success_rate/avg_reward/task_completed`
- elle evalue donc un endpoint agent code-gen compatible OpenAI
- elle ne transforme pas encore automatiquement le transducteur `ActionIR` en agent `CaP-X`

Le module s'appuie sur les conventions officielles du repo `capgym/cap-x`:

- 8 tiers `S1-S4 / M1-M4`
- configs YAML dans `env_configs/<suite>/`
- regression officielle sur `cube_stack`

Exemple benchmark complet sur `cube_stack`:

```bash
python scripts/benchmark_capx.py \
  --capx-root /path/to/cap-x \
  --results-out benchmarks/capx_cube_stack_complete.json \
  --suites cube_stack \
  --benchmark-mode complete \
  --model NousResearch/Hermes-4.3-36B \
  --server-url http://127.0.0.1:8110/chat/completions
```

Exemple pair `S2` vs `M3`:

```bash
python scripts/benchmark_capx.py \
  --capx-root /path/to/cap-x \
  --results-out benchmarks/capx_cube_stack_s2_vs_m3.json \
  --suites cube_stack \
  --benchmark-mode pair \
  --tiers S2,M3 \
  --model NousResearch/Hermes-4.3-36B \
  --server-url http://127.0.0.1:8110/chat/completions
```

Dry-run pour verifier les commandes sans lancer l'eval:

```bash
python scripts/benchmark_capx.py \
  --capx-root /path/to/cap-x \
  --results-out benchmarks/capx_dry_run.json \
  --suites cube_stack \
  --benchmark-mode complete \
  --dry-run
```

Sur RunPod, tu peux utiliser:

```bash
bash scripts/runpod_capx.sh
```

Exemple pair:

```bash
python scripts/benchmark_feature_modes.py \
  --dataset data/droid_100_hermes_supervised.jsonl \
  --checkpoint-dir benchmarks/pair_checkpoints \
  --results-out benchmarks/pair_results.json \
  --benchmark-mode pair \
  --modes vanilla,full \
  --epochs 20 \
  --device cuda
```

Exemple `layer_sweep`:

```bash
python scripts/benchmark_layer_sweep.py \
  --dataset-id lerobot/droid_100 \
  --split train \
  --robot-profile arm \
  --model-id NousResearch/Hermes-3-Llama-3.1-8B \
  --hf-cache-dir /workspace/.hf \
  --layers -1,-2,-4,-8,-16 \
  --dataset-dir data/layer_sweep \
  --checkpoint-dir benchmarks/layer_sweep_checkpoints \
  --results-out benchmarks/layer_sweep.json \
  --device cuda
```

Sur RunPod:

```bash
RUN_MODE=layer_sweep \
LAYER_SWEEP_LAYERS=-1,-2,-4,-8,-16 \
LAYER_SWEEP_RESULTS_PATH=/workspace/benchmarks/layer_sweep.json \
LAYER_SWEEP_DATASET_DIR=/workspace/data/layer_sweep \
LAYER_SWEEP_CHECKPOINT_DIR=/workspace/benchmarks/layer_sweep_ckpts \
bash scripts/runpod_train.sh
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

Pour eviter de retelecharger/reconvertir si les artefacts existent deja:

- `FORCE_REBUILD_DATASET=0` garde le JSONL existant
- `FORCE_RETRAIN=0` garde le checkpoint existant
- `HF_CACHE_DIR=/workspace/.hf` force un cache Hugging Face persistant
- `LOCAL_FILES_ONLY=1` force l'usage du cache local sans retéléchargement

Exemple:

```bash
HF_CACHE_DIR=/workspace/.hf \
LOCAL_FILES_ONLY=1 \
FORCE_REBUILD_DATASET=0 \
FORCE_RETRAIN=0 \
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

Pour brancher directement le vrai encodeur Hermes pendant la conversion:

```bash
python scripts/convert_hf_dataset.py \
  --dataset-id lerobot/droid_100 \
  --split train \
  --robot-profile arm \
  --source-format droid \
  --encoder-backend hermes_hf \
  --model-id NousResearch/Hermes-4.3-36B \
  --torch-dtype bfloat16 \
  --device-map auto \
  --out data/droid_100_hermes_supervised.jsonl \
  --max-rows 5000 \
  --max-episodes 200
```

Tu peux aussi enchaîner directement conversion + train + eval:

```bash
HF_DATASET_ID=lerobot/droid_100 \
ROBOT_PROFILE=arm \
DATASET_PATH=data/droid_100_supervised.jsonl \
CHECKPOINT_PATH=checkpoints/droid_100_action_ir.pt \
bash scripts/convert_and_train.sh
```

Version avec vrai Hermes HF pendant la conversion:

```bash
HF_DATASET_ID=lerobot/droid_100 \
ROBOT_PROFILE=arm \
ENCODER_BACKEND=hermes_hf \
HERMES_MODEL_ID=NousResearch/Hermes-4.3-36B \
HERMES_TORCH_DTYPE=bfloat16 \
DATASET_PATH=data/droid_100_hermes_supervised.jsonl \
CHECKPOINT_PATH=checkpoints/droid_100_hermes_action_ir.pt \
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

Version RunPod avec vrai Hermes HF pendant la conversion:

```bash
HF_DATASET_ID=lerobot/droid_100 \
ROBOT_PROFILE=arm \
ENCODER_BACKEND=hermes_hf \
HERMES_MODEL_ID=NousResearch/Hermes-4.3-36B \
HERMES_TORCH_DTYPE=bfloat16 \
DATASET_PATH=/workspace/data/droid_100_hermes_supervised.jsonl \
CHECKPOINT_PATH=/workspace/checkpoints/droid_100_hermes_action_ir.pt \
bash scripts/runpod_train.sh
```

## Tests

```bash
cd /Users/robertbadinter/Downloads/latent-relay/separate-repos/hermes-action-transducer
pytest -q
```
