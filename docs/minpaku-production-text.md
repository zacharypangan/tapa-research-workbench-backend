# Minpaku production text worker

The production gateway is disabled until distinct `MODEL_GATEWAY_TOKEN` and
`MODEL_WORKER_TOKEN` values are set. `REPOSITORY_MINPAKU_TEXT_ENABLED=1` switches
text generation to the fixed production gateway and `qwen3:8b`. Embeddings and
vision keep their existing configuration. Never reuse staging tokens or expose
the loopback Ollama listener.

Copy the reviewed `scripts/minpaku_pilot/` files into a separate private
production script directory inside `/home/zachary/workbench-inference`. Place a
user-owned mode-0600 production worker configuration in that project containing
only `backend_base_url`, `worker_id`, and `token`; the URL must be the production
API origin. Preserve its token privately and never put it in shell arguments or
logs. The existing verified runtime and model cache are reused.

Stop the staging allocation before scheduling the production worker. Submit the
production `pilot.sbatch` through Slurm with the production script directory,
project directory and private configuration path as its three arguments. It has
a two-hour limit, stops shortly before that limit, proves GPU execution at
startup, and accepts only bounded text requests. Monitor the job and the private
gateway health before enabling text routing; test assisted review, translation
and graph extraction on a bounded canary. A longer reservation requires the
applicable shared GPU approval and supervision.

For rollback, disable `REPOSITORY_MINPAKU_TEXT_ENABLED` and redeploy the prior
provider settings; then stop the production worker allocation. The code keeps
previous translations and graph checkpoints. Runs pinned to a different text
route require a new graph run; retrying an incomplete translation with a changed
model creates a separate reading copy.
