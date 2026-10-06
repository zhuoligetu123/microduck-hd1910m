#!/usr/bin/env python3
"""Fail-closed LOCAL qualification. Never certifies powered hardware remotely."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import onnxruntime as ort
from replay_hd1910 import baseline_check, replay_cases, transition_cases, validate_metadata, load_replay_model, MAX_SATURATED_REVERSAL_FRACTION
from mjlab_microduck.actuator.bounded_position import SLEW_CONTRACT

STRESS_CONDITIONS = ((6.5,6),(7.4,3),(8.4,6))


def check_stress(report, digest, profile_digest, voltage, delay):
    errors = check_replay(report,digest,replay_cases(True))
    expected = dict(profile_sha256=profile_digest, voltage_v=voltage,
                    delay_physics_steps=delay, actuator_delay_ms=delay*5,
                    initial_tilt_range_deg=5., seed=42, policy_hz=50, physics_hz=200)
    for key,value in expected.items():
        if report.get(key) != value:
            errors.append('stress '+key+' mismatch')
    return errors


def check_replay(report, digest, expected_cases, continuous=False):
    errors=[]
    if report.get('policy_sha256') != digest:
        errors.append('policy hash mismatch')
    if report.get('action_semantics') != SLEW_CONTRACT:
        errors.append('action contract mismatch')
    step_limit=report.get('max_action_step_rad')
    if not isinstance(step_limit,(int,float)) or not 0<step_limit<=.12:
        return errors+['invalid/missing trained step limit']
    seconds=report.get('seconds_per_case',report.get('seconds',0))
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<20:
        return errors+['insufficient duration']
    rows=report.get('cases',[])
    if [r.get('case') for r in rows] != [name for name,_ in expected_cases]:
        return errors+['missing or reordered cases']
    if continuous and report.get('continuous_transitions') is not True:
        errors.append('state was reset across transitions')
    for row,(name,command) in zip(rows,expected_cases):
        reversal = row.get('saturated_reversal_fraction_max_joint')
        if (not isinstance(reversal,(int,float)) or not math.isfinite(reversal)
                or not 0 <= reversal <= MAX_SATURATED_REVERSAL_FRACTION
                or row.get('motion_quality_check_passed') is not True):
            errors.append(name+': target chatter measurements missing/failed')
        velocity=row.get('mean_body_velocity_after_1s')
        if not isinstance(velocity,list) or len(velocity)!=3:
            errors.append(name+': missing velocity measurements')
            continue
        values=[*velocity,row.get('max_tilt_deg',float('nan')),
                row.get('max_target_jump_rad',float('nan')),row.get('initial_target_jump_rad',float('nan')),
                row.get('rms_vx_error_after_1s',float('nan')),row.get('rms_yaw_error_after_1s',float('nan'))]
        if (len(values)!=8 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
                or row.get('command')!=list(command)):
            errors.append(name+': invalid measurements/command')
            continue
        if row.get('target_limit_violations')!=0:
            errors.append(name+': target bounds failed')
        if row['max_target_jump_rad']>step_limit+1e-6 or row['initial_target_jump_rad']>step_limit+1e-6:
            errors.append(name+': matched slew limit failed')
        if (row.get('completed') is not True or row.get('no_fall') is not True
                or not baseline_check(row,seconds)):
            errors.append(name+': balance/velocity criteria failed')
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    digest=hashlib.sha256(args.policy.read_bytes()).hexdigest()
    session=ort.InferenceSession(str(args.policy),providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map
    errors=[]
    if session.get_inputs()[0].shape!=[1,61] or session.get_outputs()[0].shape!=[1,14]:
        errors.append('policy shape mismatch')
    model,_,motor=load_replay_model(7.4)
    validate_metadata(meta,[model.joint(int(i)).name for i in motor.joint_ids])
    if meta.get('action_semantics')!=SLEW_CONTRACT:
        errors.append('expected trained slew contract')
    inputs={}

    def read(name):
        path=args.reports/name
        try:
            blob=path.read_bytes()
            inputs[name]=hashlib.sha256(blob).hexdigest()
            return json.loads(blob)
        except (OSError,ValueError) as exc:
            errors.append(name+': '+str(exc))
            return {}

    for engine in ('cpu','warp'):
        for seed in (42,7,123):
            name=f'{engine}_seed{seed}.json'
            report=read(name)
            if report.get('seed')!=seed or report.get('profile_sha256')!=meta.get('calibration_sha256'):
                errors.append(name+': seed/profile mismatch')
            errors.extend(name+': '+e for e in check_replay(report,digest,replay_cases(True)))
    errors.extend('transitions: '+e for e in check_replay(read('transitions.json'),digest,transition_cases(),True))
    for name in ('onnx_parity.json','native_parity.json'):
        result=read(name)
        error=result.get('max_abs_parity_error',float('inf'))
        if (result.get('policy_sha256')!=digest or result.get('samples',0)<200
                or not math.isfinite(error) or error>1e-5 or result.get('parity_passed') is not True):
            errors.append(name+': inference parity failed')
        if name == 'onnx_parity.json' and result.get('output_bound_violations') != 0:
            errors.append(name+': output bounds evidence missing/failed')
    nominal_ready = not errors
    for voltage,delay in STRESS_CONDITIONS:
        name = f'stress_v{voltage}_delay{delay}.json'
        report = read(name)
        if report:
            errors.extend(name+': '+e for e in check_stress(
                report,digest,meta['calibration_sha256'],voltage,delay))
        else:
            errors.append(name+': no stress measurements')
    result=dict(policy_sha256=digest,input_report_sha256=inputs,
                nominal_ready=nominal_ready,local_ready=not errors,
                local_blockers=errors,deployment_ready=False,hardware_tested=False,
                physical_blockers=['HD1910 dynamics/load/thermal identification not validated',
                                   'powered timing, fault-stop and supported/grounded gait acceptance missing'],
                calibration_status=meta.get('calibration_status'))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['local_ready'] else 1


if __name__=='__main__':
    raise SystemExit(main())
