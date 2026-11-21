# Stage 1 GRPO/DAPO core smoke summary

Training subset: 20 unique ECD questions. Each row is one optimizer-step smoke on five selected questions, with five fresh rollouts per question.
Validation used five heldout questions solely to test the pipeline. These accuracies are not benchmark results.
DAPO core used asymmetric clipping, token-level loss, and raw v1.2 accuracy filtering. Soft overlong punishment was disabled, so this is not full DAPO.
Adapter header, tensor count, parameter count, and scope checks are static. Actual PEFT reload is checked separately and recorded when `adapter_reload.json` is available.

| Method | Cap | Attempted / used questions | Raw accuracy (attempted) | Selected reward mean | Answer present | At cap | Mean tokens | Grad norm | Signal | Step s | Holdout5 | Adapter MiB | PEFT reload | SHA256 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| GRPO | 512 | 5 / 5 | 0.560 | 0.560 | 0.840 | 0.160 | 300.6 | 0.079 | yes | 32.0 | 0.600 | 45.0 | yes | ba8be72cec0b | observed_complete |
| DAPO core | 512 | 10 / 5 | 0.540 | 0.480 | 0.880 | 0.120 | 328.2 | 0.098 | yes | 42.8 | 0.600 | 45.0 | yes | fb3786188ef4 | observed_complete |
| GRPO | 1024 | 5 / 5 | 0.600 | 0.600 | 0.920 | 0.080 | 363.1 | 0.067 | yes | 44.1 | 0.600 | 45.0 | yes | 6f26dc6e9ec9 | observed_complete |
| DAPO core | 1024 | 10 / 5 | 0.560 | 0.520 | 0.940 | 0.060 | 365.2 | 0.090 | yes | 61.0 | 0.600 | 45.0 | yes | ae9363810520 | observed_complete |
| GRPO | 2048 | 5 / 5 | 0.600 | 0.600 | 0.960 | 0.040 | 408.6 | 0.068 | yes | 75.5 | 0.600 | 45.0 | yes | 0dae3ac5b5ab | observed_complete |
| DAPO core | 2048 | 10 / 5 | 0.560 | 0.520 | 0.960 | 0.040 | 406.2 | 0.078 | yes | 119.4 | 0.600 | 45.0 | yes | 4579a53459fa | observed_complete |

Attempted visits are logged before generation. For completed runs, attempted visits × 5 gives generated responses. DAPO may attempt more questions than it uses for an optimizer step.
