from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(4, 20)
players = range(2, 5)
foods = range(1, 10)
coop = [True, False]
partial_obs = [True, False]

for s, p, f, c, po in product(sizes, players, foods, coop, partial_obs):
    register(
        id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v0".format(s, p, f, "-coop" if c else "", "-2s" if po else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "field_size": (s, s),
            "max_food_n": f,
            "sight": 2 if po else s,
            "max_episode_steps": 50,
            "force_coop": c,
        },
    )
