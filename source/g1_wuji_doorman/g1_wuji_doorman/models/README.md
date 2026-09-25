# HOMIE standing checkpoint

`model_stand.pt` is the frozen HOMIE standing-policy checkpoint used by the
DoorMan project. It was copied from `GR00T-VisualSim2Real/models/model_stand.pt`.

- Upstream copyright: NVIDIA CORPORATION & AFFILIATES
- License: Apache-2.0; see `LICENSE.apache-2.0.txt`
- SHA-256: `ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054`
- Expected policy contract: 6 x 86 observations, 15 actions

Set `G1_HOMIE_CHECKPOINT` to an alternate file to override the bundled model.
