import unittest
import time

from environment.LunarLanderEnvironment import LunarLanderEnvironment


class LunarLanderEnvironmentTestCase(unittest.TestCase):

    def test_init(self):
        env = LunarLanderEnvironment(False, {})
        self.assertIsNotNone(env)

        env = LunarLanderEnvironment(True, {})
        self.assertIsNotNone(env)

    def test_random_action(self):
        env = LunarLanderEnvironment(False, {})
        env.reset()
        for _ in range(200):
            env.render()
            env.step(env.action_space.sample())
            time.sleep(0.01)
        env.close()

        env = LunarLanderEnvironment(True, {})
        env.reset()
        for _ in range(200):
            env.render()
            env.step(env.action_space.sample())
            time.sleep(0.01)
        env.close()

if __name__ == '__main__':
    unittest.main()