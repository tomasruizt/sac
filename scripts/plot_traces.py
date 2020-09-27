from itertools import repeat
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import joblib
import tensorflow as tf
import os
import gc
from sac.policies.hierarchical_policy import FixedOptionPolicy
import sac.envs.point2d_env  # To register with gym


_use_all_skills = -1


def dump_trace(picklefile: str, args):

    filename = '{}_{}_{}_trace.png'.format(os.path.splitext(picklefile)[0],
                                           args.dim_0, args.dim_1)

    with tf.Session(), tf.variable_scope(picklefile):
        data = joblib.load(picklefile)
        policy = data['policy']
        env = data['env']
        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim

        plt.figure(figsize=(6, 6))
        palette = sns.color_palette('hls', num_skills)
        with policy.deterministic(args.deterministic):
            skills = range(num_skills) if args.specific_skill == _use_all_skills else [args.specific_skill]
            for z in skills:
                fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
                for path_index in range(args.n_paths):
                    obs = env.reset()
                    if args.use_qpos:
                        qpos = env.wrapped_env.env.model.data.qpos[:, 0]
                        obs_vec = [qpos]
                    else:
                        obs_vec = [obs]
                    for t in range(args.max_path_length):
                        action, _ = fixed_z_policy.get_action(obs)
                        (obs, _, _, _) = env.step(action)
                        if args.use_qpos:
                            qpos = env.wrapped_env.env.model.data.qpos[:, 0]
                            obs_vec.append(qpos)
                        elif args.use_action:
                            obs_vec.append(action)
                        else:
                            obs_vec.append(obs)

                    obs_vec = np.array(obs_vec)
                    x = obs_vec[:, args.dim_0]
                    y = obs_vec[:, args.dim_1]
                    plt.plot(x, y, c=palette[z])

                    use_plot_lims = np.isfinite(env.observation_space.bounds).all()
                    if use_plot_lims:
                        xlim, ylim = np.asarray(env.observation_space.bounds).T
                        plt.xlim(xlim)
                        plt.ylim(ylim)

        plt.savefig(filename)
        plt.close()


def single_arg_wrapper(params):
    picklefile, args = params
    dump_trace(picklefile=picklefile, args=args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--all-pkl-files', type=bool, default=False)
    parser.add_argument('--max-path-length', '-l', type=int, default=100)
    parser.add_argument('--n_paths', type=int, default=1)
    parser.add_argument('--dim_0', type=int, default=0)
    parser.add_argument('--dim_1', type=int, default=1)
    parser.add_argument('--use_qpos', type=bool, default=False)
    parser.add_argument('--use_action', type=bool, default=False)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.add_argument('--specific-skill', type=int, default=_use_all_skills)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()
    if args.all_pkl_files:
        directory = os.path.dirname(args.file)
        pickle_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pkl")]
    else:
        pickle_files = [args.file]

    with Pool() as p:
        params = zip(pickle_files, repeat(args))
        p.map(single_arg_wrapper, params)
