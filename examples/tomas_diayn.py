from examples.mujoco_all_diayn import parse_args, get_variants, run_experiment

if __name__ == '__main__':
    args = parse_args()
    variant = get_variants(args).variants()[0]
    run_experiment(variant)