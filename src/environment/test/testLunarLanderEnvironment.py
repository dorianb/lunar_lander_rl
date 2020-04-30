import unittest
import time

from environment.LunarLanderEnvironment import LunarLanderEnvironment


class LunarLanderEnvironmentTestCase(unittest.TestCase):

    def test_init(self):
        env = LunarLanderEnvironment({"continuous": False})
        self.assertIsNotNone(env)

        env = LunarLanderEnvironment({"continuous": True})
        self.assertIsNotNone(env)

    def test_random_action(self):
        env = LunarLanderEnvironment({"continuous": False})
        env.reset()
        for _ in range(200):
            env.render()
            env.step(env.action_space.sample())
            time.sleep(0.01)
        env.close()

        env = LunarLanderEnvironment({"continuous": True})
        env.reset()
        for _ in range(200):
            env.render()
            env.step(env.action_space.sample())
            time.sleep(0.01)
        env.close()


if __name__ == '__main__':
    unittest.main()