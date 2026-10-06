"""Audit supplied BAM parameters without guessing a firmware controller or writing hardware."""
import argparse
import json
import math
from pathlib import Path


def audit(path):
    from bam.actuators import actuators
    data = json.loads(Path(path).read_text())
    required = {'kt','R','error_gain_ratio','armature','max_velocity','q_offset',
                'command_delay','friction_base','friction_viscous','load_friction_base'}
    if set(data) != required | {'model','actuator'} or data['model'] != 'm3' or data['actuator'] != 'hd1910':
        raise ValueError('expected the complete HD1910 BAM m3 parameter record')
    if any(type(data[k]) not in (int,float) or not math.isfinite(data[k]) for k in required):
        raise ValueError('parameters must be finite numbers')
    if any(data[k] < 0 for k in required-{'q_offset'}) or min(data['kt'],data['R'],data['max_velocity']) <= 0:
        raise ValueError('invalid physical parameter sign')
    return dict(
        status='external_candidate_not_hardware_identified',
        installed_bam_has_hd1910=data['actuator'] in actuators,
        q_offset_deg=math.degrees(data['q_offset']),
        max_velocity_rpm=data['max_velocity']*60/(2*math.pi),
        command_delay_ms=data['command_delay']*1000,
        delay_at_5ms_steps=data['command_delay']/.005,
        electrical_estimates=[dict(voltage_v=v,ideal_stall_current_a=v/data['R'],
                                  ideal_stall_torque_nm=v*data['kt']/data['R'],
                                  ideal_no_load_rad_s=v/data['kt']) for v in (7.4,8.4)],
        missing=['HD1910 actuator source and firmware control law',
                 'test voltage, firmware mode, P/D/PWM settings and load',
                 'raw identification traces and held-out residuals'],
        warning='q_offset is a testbench fit, not the 15-joint installation zero; command_delay is not measured full-loop latency',
        hardware_changes=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parameters',type=Path)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    report=audit(args.parameters)
    text=json.dumps(report,indent=2,allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(text+'\n')
    print(text)


if __name__=='__main__': main()
