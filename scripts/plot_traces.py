from itertools import zip_longest, repeat
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse
import numpy as np
import joblib
import tensorflow as tf
import os
from sac.policies.hierarchical_policy import FixedOptionPolicy

_use_all_skills = -1


def dump_trace(picklefile: str, args):

    filename = '{}_{}_{}_trace.png'.format(os.path.splitext(picklefile)[0],
                                           args.dim_0, args.dim_1)

    with tf.Session(), tf.variable_scope("", reuse=tf.AUTO_REUSE):
        data = joblib.load(picklefile)
        policy = data['policy']
        env = data['env']
        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim

        if args.three_dims:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(projection="3d"))
        else:
            fig, axs = plt.subplots(figsize=(12, 12))

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

                    plot_kwargs = dict(c=palette[z], alpha=0.6)
                    if args.three_dims:
                        h = obs_vec[:, 2]
                        axs[0].view_init(30, 30)
                        axs[0].plot(x, y, h, **plot_kwargs)
                        axs[1].plot(x, y, h, **plot_kwargs)
                    else:
                        axs.plot(x, y, **plot_kwargs)

                    use_plot_lims = np.isfinite(env.observation_space.bounds).all()
                    if use_plot_lims:
                        xlim, ylim = np.asarray(env.observation_space.bounds).T
                        plt.xlim(xlim)
                        plt.ylim(ylim)

        plt.tight_layout()
        plt.savefig(filename, dpi=160)
        plt.close()


def tuplify(params):
    p1, p2 = params
    dump_trace(p1, p2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--all-pkl-files', dest="all_pkl_files", action="store_true")
    parser.add_argument('--max-path-length', '-l', type=int, default=100)
    parser.add_argument('--n_paths', type=int, default=1)
    parser.add_argument('--dim_0', type=int, default=0)
    parser.add_argument('--dim_1', type=int, default=1)
    parser.add_argument('--3d', dest="three_dims", action="store_true")
    parser.add_argument('--use_qpos', type=bool, default=False)
    parser.add_argument('--use_action', type=bool, default=False)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.add_argument('--specific-skill', type=int, default=_use_all_skills)
    parser.set_defaults(deterministic=True, three_dim=False, all_pkl_files=False)

    args = parser.parse_args()
    if args.all_pkl_files:
        directory = os.path.dirname(args.file)
        pickle_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pkl")]
    else:
        pickle_files = [args.file]

    with Pool() as p:
        p.map(tuplify, zip(pickle_files, repeat(args)))
