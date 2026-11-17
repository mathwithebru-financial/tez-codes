"""Explicit forward-only all-six B128 check for the V2 gradient adapter.

No input/parameter gradients or SHAP calls. Original objects/state are restored.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import run_shap_stability as runner
from gradient_compatible_adapter import compatible_encoder_input_gradients,ADAPTER_ID

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-diagnostic',action='store_true')
    args=parser.parse_args()
    payload=runner.preflight()
    target=HERE/'v2_batch128_equivalence.json'
    if not args.execute_diagnostic:
        print(json.dumps({'status':'V2_BATCH128_DIAGNOSTIC_PREPARED_NO_MODEL_OR_ARRAY_LOAD','model_count':6,'input_gradients':False}));return 0
    runner.require(not target.exists(),'Preserve earlier diagnostic; no implicit overwrite')
    torch,shap,states,Xtrain,X,saved,environment=runner.environment_and_data(payload)
    from train_p1 import state_hash
    scale=payload['data']['scalers']['y']['scale'];findings=[]
    for arm in ('mse','mae'):
        for seed in runner.MODEL_SEEDS:
            model,before=runner.build_frozen_model(states[(arm,seed)]['state'],torch)
            with compatible_encoder_input_gradients(model,torch) as adapter:
                # Exactly one128-row call, as used by the SHAP runner.
                scaled=runner.scaled_prediction(model,X,torch)
                raw=payload['compare'].inverse_y_float64(scaled,payload['data']['scalers'])
            difference=np.abs(raw-saved[(arm,seed)])/np.asarray(scale,dtype=np.float64)
            maximum=float(np.max(difference))
            runner.require(state_hash(model)==before and all(p.grad is None and not p.requires_grad for p in model.parameters()),'V2 check changed weights or parameter gradients')
            findings.append({'arm':arm,'model_seed':seed,'batch_size':128,'n':128,
                'maximum_absolute_standardized_difference':maximum,
                'per_output_maximum_standardized_difference':np.max(difference,axis=0).tolist(),
                'within_original_tolerance':maximum<=payload['protocol']['prediction_match_max_abs_standardized'],
                'state_sha256_before':before,'state_sha256_after':state_hash(model),'adapter':adapter,
                'original_encoder_and_head_objects_restored':True,'parameter_gradients_none':True})
            del model
    runner.check_sources(payload)
    passed=all(r['within_original_tolerance'] for r in findings)
    result={'status':'PASS_V2_ALL_SIX_BATCH128_SAVED_PREDICTIONS' if passed else 'FAIL_V2_BATCH128_ORIGINAL_TOLERANCE_RETAINED',
        'adapter_id':ADAPTER_ID,'protocol_sha256':payload['protocol_sha256'],'sample_plan_sha256':payload['sample_plan_sha256'],
        'source_sha256':payload['sources'],'environment':environment,'model_count':6,'models':findings,
        'original_tolerance':payload['protocol']['prediction_match_max_abs_standardized'],
        'all_model_states_and_original_objects_unchanged':True,'forward_only':True,
        'input_or_parameter_gradients_computed':False,'SHAP_or_training_or_thesis_edit':False,
        'finite_forward_agreement_not_global_gradient_identity':True}
    runner.writej(target,result)
    print(json.dumps({'status':result['status'],'maximum_standardized_difference':max(r['maximum_absolute_standardized_difference'] for r in findings),'output':str(target)}))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
