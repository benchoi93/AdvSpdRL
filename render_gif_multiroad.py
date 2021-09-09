import pickle
from util.plotutil import car_moving
from tqdm import tqdm
from itertools import chain
i = 99


env = pickle.load(open(f"simulate_gif/scn_{i}_plot_info.pkl", "rb"))
env.png_list = []
start = car_moving(env, env.vehicle.veh_info[:1], startorfinish=True, combine=True)
mid = [car_moving(env, env.vehicle.veh_info[:i+2], startorfinish=False, combine=True) for i in tqdm(range(env.timestep))]
end = car_moving(env, env.vehicle.veh_info[:env.timestep+1], startorfinish=True, combine=True)

env.png_list = start + list(chain(*mid)) + end
# env.make_gif(f"simulate_gif/scn_{i}_carmoving.gif")
env.png_list[0].save(f"simulate_gif/scn_{i}_carmoving.gif", save_all=True, append_images=env.png_list[1:], optimize=False, duration=30, loop=1)
