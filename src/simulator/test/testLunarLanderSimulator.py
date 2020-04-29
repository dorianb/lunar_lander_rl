import unittest
import time
import numpy as np

from simulator.LunarLanderSimulator import LunarLanderSimulator


class LunarLanderSimulatorTestCase(unittest.TestCase):

    def test_init(self):
        sim = LunarLanderSimulator(False)
        self.assertIsNotNone(sim)

        sim = LunarLanderSimulator(True)
        self.assertIsNotNone(sim)

    def test_random_action(self):
        sim = LunarLanderSimulator(False)
        sim.reset()
        for action in np.random.randint(0, 4, 200):
            sim.render()
            sim.step(action)
            time.sleep(0.01)
        sim.close()

        sim = LunarLanderSimulator(True)
        sim.reset()
        for action in zip(np.random.uniform(0, 1, 200), np.random.uniform(-1, 1, 200)):
            sim.render()
            sim.step(action)
            time.sleep(0.01)
        sim.close()

if __name__ == '__main__':
    unittest.main()