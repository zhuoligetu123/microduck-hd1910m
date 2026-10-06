"""Run the original PPO pipeline with an explicit HD1910 physical fit.

--smoke-fixture is only a 64-env/5-iteration pipeline check, never a real gait.
Production: --calibration PATH followed by the normal mjlab training options.
"""
import argparse
import os
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--calibration', type=Path)
    mode.add_argument('--smoke-fixture', action='store_true')
    mode.add_argument('--reference-profile', type=Path,
                      help='Uncalibrated reference model; simulation only, never identified hardware')
    parser.add_argument('--bounded-targets', action='store_true',
                        help='New bounded HOME-delta contract; requires reference-profile')
    parser.add_argument('--slew-targets', action='store_true',
                        help='New matched training/ONNX slew contract; requires reference-profile')
    parser.add_argument('--discover-gait', action='store_true',
                        help='Delayed smoothing curriculum; requires --slew-targets')
    parser.add_argument('--refine', action='store_true',
                        help='Opt-in low-speed/yaw or SitStand repair, with unchanged v2 actions')
    parser.add_argument('--posture-balanced', action='store_true',
                        help='With --refine and SitStand: preserve the seated pose prior')
    parser.add_argument('--task', help='Stock MicroDuck task id; preserve its command/action semantics')
    parser.add_argument('--warm-start', action='store_true',
                        help='Restart curricula after loading weights; not ordinary resume')
    parser.add_argument('--head-center', action='store_true', help='Opt-in DC head-bias repair on refined velocity')
    args, rest = parser.parse_known_args()
    if args.head_center and (not args.refine or args.task):
        parser.error('head-center requires refined velocity')
    if args.head_center:
        os.environ['MICRODUCK_HEAD_CENTER'] = '1'
    else:
        os.environ.pop('MICRODUCK_HEAD_CENTER', None)
    if args.warm_start:
        os.environ['MICRODUCK_HD1910_WARM_START'] = '1'
    else:
        os.environ.pop('MICRODUCK_HD1910_WARM_START', None)
    if (args.bounded_targets or args.slew_targets) and not args.reference_profile:
        parser.error('bounded/slew targets require --reference-profile')
    if args.bounded_targets and args.slew_targets:
        parser.error('select only one action contract')
    if args.discover_gait and not args.slew_targets:
        parser.error('--discover-gait requires --slew-targets')
    if args.refine and (not args.slew_targets or args.discover_gait
                       or args.task not in (None, 'Mjlab-SitStand-Flat-MicroDuck')):
        parser.error('--refine requires slew targets and either velocity or SitStand')
    if args.posture_balanced and (not args.refine or args.task != 'Mjlab-SitStand-Flat-MicroDuck'):
        parser.error('--posture-balanced requires --refine and SitStand')
    if args.task and (not args.reference_profile or args.bounded_targets or args.discover_gait):
        parser.error('--task requires reference-profile; bounded pilots use --slew-targets without --discover-gait')
    if args.smoke_fixture:
        if rest:
            parser.error('fixture run is fixed at 64 environments and 5 iterations')
        path = ROOT/'config/hd1910_smoke_fixture.json'
        rest = ['--env.scene.num-envs','64','--agent.max-iterations','5',
                '--agent.logger','tensorboard','--agent.run-name','SYNTHETIC_NOT_DEPLOYABLE',
                '--agent.save-interval','5','--agent.upload-model','False','--enable-nan-guard','True']
    elif args.reference_profile:
        path = args.reference_profile.resolve()
        profile = json.loads(path.read_text())
        if profile.get('model') != 'HD-1910-C001' or profile.get('status') != 'datasheet_based_provisional_NOT_calibrated':
            parser.error('expected an explicitly provisional HD1910 reference profile')
    else:
        path = args.calibration.resolve()
        # Validate before CUDA allocation, and never silently fill null gains.
        from mjlab_microduck.robot.hd1910 import MotorFit
        fit = MotorFit.load(path)
        if any(word in fit.evidence.lower() for word in ('synthetic','fixture','not measured')):
            parser.error('synthetic evidence is allowed only through --smoke-fixture')
    if args.reference_profile:
        os.environ.pop('MICRODUCK_HD1910_FIT', None)
        os.environ['MICRODUCK_HD1910_REFERENCE'] = str(path)
    else:
        os.environ.pop('MICRODUCK_HD1910_REFERENCE', None)
        os.environ['MICRODUCK_HD1910_FIT'] = str(path)
    os.environ.setdefault('OMP_NUM_THREADS','4')
    task = 'Mjlab-Velocity-Flat-MicroDuck-HD1910'+('-Reference' if args.reference_profile else '')
    if args.bounded_targets:
        task += '-Bounded'
    if args.slew_targets:
        task += '-Slew'
    if args.discover_gait:
        task += '-Discovery'
    if args.task:
        os.environ['MICRODUCK_HD1910_SUITE'] = '1'
        import mjlab_microduck.tasks
        from mjlab.tasks.registry import list_tasks, load_rl_cfg
        if args.task not in list_tasks() or 'MicroDuck' not in args.task or '-HD1910' in args.task:
            parser.error('expected a registered stock MicroDuck task id')
        task = args.task + '-HD1910-Reference'
        if args.slew_targets:
            task += '-Slew'
            if task not in list_tasks():
                parser.error('slew task pilots currently support SitStand and GroundPick only')
    if args.refine:
        task += '-Balanced' if args.posture_balanced else '-Refine'
    sys.argv = [sys.argv[0],task,*rest]
    calibration_bytes = path.read_bytes()
    log_root = Path('logs/rsl_rl')/('microduck_hd1910_reference' if args.reference_profile else 'microduck_hd1910_velocity')
    if args.bounded_targets:
        log_root = Path('logs/rsl_rl/microduck_hd1910_bounded')
    if args.slew_targets:
        log_root = Path('logs/rsl_rl/microduck_hd1910_slew')
    if args.discover_gait:
        log_root = Path('logs/rsl_rl/microduck_hd1910_discovery')
    if args.refine:
        log_root = Path('logs/rsl_rl/microduck_hd1910_refine')
    if args.task:
        log_root = Path('logs/rsl_rl') / load_rl_cfg(task).experiment_name
    existing = set(log_root.glob('*'))
    from mjlab.scripts.train import main as train
    train()
    # Keep the exact fit next to each checkpoint, and mark smoke exports so they
    # cannot be mistaken for an identified/validated physical robot policy.
    import onnx
    metadata = dict(hardware_profile='HD1910M-mode4', deployment_ready='false',
                    calibration_status=('external_reference_unvalidated' if args.reference_profile else
                                        'synthetic' if args.smoke_fixture else 'identified_unvalidated'),
                    task_id=task,
                    head_center_recipe='dc_roll3_ramp1_3' if args.head_center else 'baseline',
                    base_task_id=args.task or 'Mjlab-Velocity-Flat-MicroDuck',
                    calibration_sha256=hashlib.sha256(calibration_bytes).hexdigest())
    for directory in set(log_root.glob('*'))-existing:
        if not directory.is_dir():
            continue
        (directory/'motor_calibration.json').write_bytes(calibration_bytes)
        (directory/'hardware_provenance.json').write_text(json.dumps(metadata, indent=2))
        for filename in directory.glob('*.onnx'):
            model = onnx.load(filename)
            properties = {item.key:item.value for item in model.metadata_props}
            onnx.helper.set_model_props(model, {**properties, **metadata})
            onnx.checker.check_model(model)
            onnx.save(model, filename)


if __name__ == '__main__':
    main()
