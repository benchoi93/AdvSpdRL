from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                # Retrieve training reward
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print("Num timesteps: {}".format(self.num_timesteps))
                        print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose > 0:
                            print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
            except:
                pass

        return True


class GIFCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, env, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(GIFCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.env = env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_result.gif")
            # self.model.save(path)

            ob = self.env.reset()
            episode_over = False
            ob_list = []

            import matplotlib.pyplot as plt
            plt.ioff()

            check_start = 1
            check_finish = 0
            while not episode_over:
                # action = np.array(env.get_random_action())
                action, _ = self.model.predict(ob)
                # action = np.array(min(5, (50/3.6 - ob[1]) / env.dt))
                # print(action)
                ob, reward, episode_over, info = self.env.step(action)
                ob_list.append([self.env.vehicle.position, self.env.vehicle.velocity,
                               self.env.vehicle.acceleration, self.env.timestep, reward])
                # env.render(visible=True)
                self.env.car_moving(ob_list, check_start, check_finish)
                check_start = 0

                # input()

            check_finish = 1
            self.env.car_moving(ob_list, check_start, check_finish)
            self.env.info_graph(ob_list)
            self.env.make_gif(path=path)

            if self.verbose > 1:
                print(f"Saving gif result to {path}")
        return True
