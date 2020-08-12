from genrl import PPO1, A2C


class MarioPPO(PPO1):
    def __init__(self, *args, **kwargs):
        super(MarioPPO, self).__init__(*args, **kwargs)

    def collect_rollouts(self, state):
        for timestep in range(self.rollout_size):
            action, value, old_log_prob = self.select_action(state)

            next_state, reward, done, _ = self.env.step(int(action))

            if self.render:
                self.env.render()

            self.rollout.add(
                state, action, reward, done, value.detach(), old_log_prob.detach(),
            )

            state = next_state

            if done or timestep == self.rollout_size - 1:
                self.rewards.append(self.env.episode_reward)
                state = self.env.reset()

        return value, done
