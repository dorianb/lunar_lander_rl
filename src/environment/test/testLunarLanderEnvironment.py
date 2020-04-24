import unittest
import time

from environment.LunarLanderEnvironment import LunarLanderEnvironment


class LunarLanderEnvironmentTestCase(unittest.TestCase):

    def test_init(self):
        env = LunarLanderEnvironment({})
        self.assertIsNotNone(env)

    def test_random_action(self):
        env = LunarLanderEnvironment({})
        env.reset()
        for _ in range(200):
            env.render()
            env.step(env.action_space.sample())
            time.sleep(0.01)
        env.close()

    def test_render(self):
        pass

    def test_reset(self):
        pass

    def test_step(self):
        pass

if __name__ == '__main__':
    unittest.main()