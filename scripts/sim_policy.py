import argparse

import joblib
import tensorflow as tf
from itertools import count

from rllab.sampler.utils import rollout
from sac.policies.hierarchical_policy import FixedOptionPolicy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            env = data['algo'].env
        else:
            policy = data['policy']
            env = data['env']

        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim

        with policy.deterministic(args.deterministic):
            for t in count():
                skill = t % num_skills
                fixed_policy = FixedOptionPolicy(base_policy=policy, num_skills=num_skills, z=skill)
                path = rollout(env, fixed_policy,
                               max_path_length=args.max_path_length,
                               animated=True, speedup=args.speedup)
