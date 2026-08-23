# Phase 1 runtime and checkpoint audit

This generated worklist summarizes static audit signals; it does not mark any file
as unused and does not authorize removal. Regenerate it with
`python tools/phase0_inventory.py` after fetching remote refs.

Source fingerprint: `sha256:91c850b486f381a8f9e98227bc7adbd131cbc3e75978f4183c1894062d21fcf7`

## Executable entry points (48)

| File | Signal |
| --- | --- |
| `ciip/eurosat.py` | yes |
| `ciip/evaluation/eurosat_fewshot_1nn.py` | yes |
| `ciip/evaluation/eurosat_fewshot_1nn_multi.py` | yes |
| `ciip/evaluation/export_neuco_embeddings.py` | yes |
| `ciip/evaluation/linearprobe_comparison.py` | yes |
| `ciip/evaluation/neuco_fewshot_benchmark.py` | yes |
| `ciip/evaluation/neuco_fewshot_benchmark_multi.py` | yes |
| `ciip/evaluation/neuco_fewshot_episodic.py` | yes |
| `ciip/evaluation/neuco_layerwise_probes.py` | yes |
| `ciip/evaluation/pangaea-results/run_pangaea_batch_eval.py` | yes |
| `ciip/evaluation/pangaea-results/run_pangaea_batch_eval_fewshot.py` | yes |
| `ciip/evaluation/pangaea-results/visualize_batch_results.py` | yes |
| `ciip/evaluation/plot_downstream.py` | yes |
| `ciip/evaluation/plot_sen12ms_retrieval.py` | yes |
| `ciip/evaluation/run_downstream.py` | yes |
| `ciip/evaluation/sen12ms_hyperbolic_radii.py` | yes |
| `ciip/evaluation/unified_evaluation.py` | yes |
| `ciip/few_shot_comparison.py` | yes |
| `ciip/model_ciip.py` | yes |
| `ciip/open_clip_train/data.py` | yes |
| `ciip/open_clip_train/dataparallel/run_train_val_dataparallel.py` | yes |
| `ciip/open_clip_train/main.py` | yes |
| `ciip/open_clip_train/profiler.py` | yes |
| `ciip/open_clip_train/run_train_val.py` | yes |
| `ciip/open_clip_train/run_train_val_distributed.py` | yes |
| `ciip/open_clip_train/save_embeddings.py` | yes |
| `ciip/open_clip_train/save_ssl4eo_embeddings.py` | yes |
| `ciip/open_clip_train/test_train_one_epoch.py` | yes |
| `ciip/open_clip_train/train.py` | yes |
| `comparison/CROMA-main/extract_croma_embeddings.py` | yes |
| `diagnostics/global_id_table1/merge_global_id_downstream.py` | yes |
| `diagnostics/pangaea_limited_label_results.py` | yes |
| `diagnostics/unified_eval/plot_eurosat_knn_vs_linearprobe.py` | yes |
| `diagnostics/unified_eval/plot_neuco_results.py` | yes |
| `intrinsic-dimension/compute_id.py` | yes |
| `intrinsic-dimension/compute_intrinsic_dimension_ssl4eo_val.py` | yes |
| `tools/phase0_inventory.py` | yes |
| `tools/phase1_config_contracts.py` | yes |
| `tools/phase1_external_audit.py` | yes |
| `tools/phase2_model_contracts.py` | yes |
| `tools/test_phase0_inventory.py` | yes |
| `tools/test_phase1_config_contracts.py` | yes |
| `tools/test_phase1_external_audit.py` | yes |
| `tools/test_phase2_model_contracts.py` | yes |
| `tools/test_phase2_runtime.py` | yes |
| `visualizations/ssl4eo/embedding_collapse_diagnostics.py` | yes |
| `visualizations/ssl4eo/hyperbolic_visualization.py` | yes |
| `visualizations/ssl4eo/initialization_evaluation.py` | yes |

## Argument parser construction (31)

| File | Signal |
| --- | --- |
| `ciip/evaluation/eurosat_fewshot_1nn.py` | yes |
| `ciip/evaluation/eurosat_fewshot_1nn_multi.py` | yes |
| `ciip/evaluation/export_neuco_embeddings.py` | yes |
| `ciip/evaluation/neuco_fewshot_benchmark.py` | yes |
| `ciip/evaluation/neuco_fewshot_benchmark_multi.py` | yes |
| `ciip/evaluation/neuco_fewshot_episodic.py` | yes |
| `ciip/evaluation/neuco_layerwise_probes.py` | yes |
| `ciip/evaluation/pangaea-results/run_pangaea_batch_eval.py` | yes |
| `ciip/evaluation/pangaea-results/run_pangaea_batch_eval_fewshot.py` | yes |
| `ciip/evaluation/pangaea-results/visualize_batch_results.py` | yes |
| `ciip/evaluation/plot_downstream.py` | yes |
| `ciip/evaluation/plot_sen12ms_retrieval.py` | yes |
| `ciip/evaluation/run_downstream.py` | yes |
| `ciip/evaluation/sen12ms_hyperbolic_radii.py` | yes |
| `ciip/evaluation/unified_evaluation.py` | yes |
| `ciip/open_clip_train/data.py` | yes |
| `ciip/open_clip_train/params.py` | yes |
| `ciip/open_clip_train/profiler.py` | yes |
| `ciip/open_clip_train/save_embeddings.py` | yes |
| `ciip/open_clip_train/save_ssl4eo_embeddings.py` | yes |
| `comparison/CROMA-main/extract_croma_embeddings.py` | yes |
| `diagnostics/global_id_table1/merge_global_id_downstream.py` | yes |
| `diagnostics/pangaea_limited_label_results.py` | yes |
| `diagnostics/unified_eval/plot_eurosat_knn_vs_linearprobe.py` | yes |
| `diagnostics/unified_eval/plot_neuco_results.py` | yes |
| `intrinsic-dimension/compute_intrinsic_dimension_ssl4eo_val.py` | yes |
| `tools/phase0_inventory.py` | yes |
| `tools/phase1_config_contracts.py` | yes |
| `tools/phase1_external_audit.py` | yes |
| `tools/phase2_model_contracts.py` | yes |
| `visualizations/ssl4eo/hyperbolic_visualization.py` | yes |

## Dynamic import sites (1)

| File | Signal |
| --- | --- |
| `intrinsic-dimension/compute_intrinsic_dimension_ssl4eo_val.py` | importlib.import_module |

## Checkpoint loading sites (23)

| File | Signal |
| --- | --- |
| `ciip/ciip.py` | torch.load |
| `ciip/eval_utils.py` | model.load_state_dict, torch.load |
| `ciip/evaluation/export_neuco_embeddings.py` | enc.load_state_dict, model.load_state_dict, torch.load |
| `ciip/evaluation/model_utils.py` | backbone.load_state_dict, model.load_state_dict, s1_backbone.load_state_dict, torch.load |
| `ciip/evaluation/sen12ms_hyperbolic_radii.py` | torch.load |
| `ciip/evaluation/unified_evaluation.py` | torch.load |
| `ciip/model.py` | model.load_state_dict |
| `ciip/open_clip_train/dataparallel/factory.py` | model.load_state_dict |
| `ciip/open_clip_train/dataparallel/model_arch.py` | encoder.load_state_dict, target.load_state_dict, torch.load |
| `ciip/open_clip_train/dataparallel/run_train_val_dataparallel.py` | load_state_dict, loss.load_state_dict, optimizer.load_state_dict, scaler.load_state_dict, torch.load |
| `ciip/open_clip_train/file_utils.py` | torch.load |
| `ciip/open_clip_train/main.py` | model.load_state_dict, optimizer.load_state_dict, scaler.load_state_dict |
| `ciip/open_clip_train/run_train_val_distributed.py` | loss.load_state_dict, model.load_state_dict, optimizer.load_state_dict, scaler.load_state_dict |
| `ciip/open_clip_train/save_embeddings.py` | model.load_state_dict, torch.load |
| `ciip/open_clip_train/save_ssl4eo_embeddings.py` | model_s1.load_state_dict, model_s2.load_state_dict |
| `clip/clip.py` | torch.load |
| `clip/model.py` | model.load_state_dict |
| `comparison/CROMA-main/use_croma.py` | self.GAP_FFN_s1.load_state_dict, self.GAP_FFN_s2.load_state_dict, self.cross_encoder.load_state_dict, self.s1_encoder.load_state_dict, self.s2_encoder.load_state_dict, torch.load |
| `intrinsic-dimension/compute_id.py` | model.load_state_dict, torch.load |
| `intrinsic-dimension/compute_intrinsic_dimension_ssl4eo_val.py` | model.load_state_dict, torch.load |
| `tools/test_phase2_runtime.py` | restored.load_state_dict, torch.load |
| `visualizations/ssl4eo/hyperbolic_visualization.py` | torch.load |
| `visualizations/ssl4eo/initialization_evaluation.py` | model.load_state_dict, torch.load |

## Checkpoint/state-dict production sites (20)

| File | Signal |
| --- | --- |
| `ciip/ciip.py` | model.state_dict |
| `ciip/eurosat.py` | model.state_dict, torch.save |
| `ciip/evaluation/model_utils.py` | model.state_dict |
| `ciip/evaluation/sen12ms_hyperbolic_radii.py` | torch.save |
| `ciip/open_clip_train/dataparallel/model_arch.py` | encoder.state_dict, target.state_dict |
| `ciip/open_clip_train/dataparallel/run_train_val_dataparallel.py` | loss.state_dict, optimizer.state_dict, scaler.state_dict, state_dict, torch.save |
| `ciip/open_clip_train/file_utils.py` | torch.save |
| `ciip/open_clip_train/main.py` | loss.state_dict, model.state_dict, optimizer.state_dict, original_model.state_dict, scaler.state_dict, torch.save |
| `ciip/open_clip_train/run_train_val.py` | optimizer.state_dict, original_model.state_dict, scaler.state_dict, torch.save |
| `ciip/open_clip_train/run_train_val_distributed.py` | loss.state_dict, model.state_dict, optimizer.state_dict, original_model.state_dict, scaler.state_dict, torch.save |
| `ciip/open_clip_train/save_embeddings.py` | torch.save |
| `ciip/open_clip_train/save_ssl4eo_embeddings.py` | torch.save |
| `ciip/open_clip_train/train.py` | model.state_dict, optimizer.state_dict, original_model.state_dict, scaler.state_dict, torch.save |
| `clip/clip.py` | model.state_dict |
| `comparison/CROMA-main/extract_croma_embeddings.py` | torch.save |
| `intrinsic-dimension/compute_id.py` | model.state_dict |
| `intrinsic-dimension/compute_intrinsic_dimension_ssl4eo_val.py` | model.state_dict |
| `tools/test_phase2_runtime.py` | restored.state_dict, source.state_dict, torch.save |
| `visualizations/ssl4eo/hyperbolic_visualization.py` | model.state_dict |
| `visualizations/ssl4eo/initialization_evaluation.py` | model.state_dict, torch.save |

## Required follow-up

Before reorganizing any listed file, validate its CLI/config callers and add the
applicable model-construction, forward-pass, and checkpoint round-trip baseline.
Dynamic paths, Hydra composition, notebooks, and external HPC launchers still
require manual review.
