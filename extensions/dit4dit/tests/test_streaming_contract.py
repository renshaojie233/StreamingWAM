"""Contract tests independent of heavyweight model initialization."""

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from DiT4DiT.model.modules.vlm.Cosmos25 import _Cosmos25_Interface
from examples.LIBERO.eval_files.streaming_policy import StreamingLiberoPolicy


class ContractTest(unittest.TestCase):
    def test_video_split_never_uses_future_as_condition(self):
        interface = _Cosmos25_Interface.__new__(_Cosmos25_Interface)
        nn.Module.__init__(interface)
        interface.config = OmegaConf.create({"datasets": {"vla_data": {
            "streaming": True, "video_delta_indices": [-4, -3, -2, -1, 0, 2, 4, 6, 8]}}})
        frames = [np.full((32, 64, 3), i * 20, np.uint8) for i in range(9)]
        training = interface.build_cosmos_inputs([frames], ["pick up cup"])
        inference = interface.build_cosmos_inputs([frames[:5]], ["pick up cup"])
        torch.testing.assert_close(training["videos"], inference["videos"])
        self.assertEqual(tuple(training["videos"].shape), (1, 5, 3, 32, 64))
        self.assertEqual(tuple(training["future_videos"].shape), (1, 4, 3, 32, 64))
        self.assertIsNone(inference["future_videos"])
        with self.assertRaises(ValueError):
            interface.build_cosmos_inputs([frames[:1]], ["bad history"])

    def test_client_records_every_control_frame_and_actual_executed_action(self):
        calls = []
        def predict(examples):
            example = examples[0]
            calls.append(example)
            actions = np.full((1, 4, 8), 0.25, np.float32)
            actions[..., 6] = 0.7
            return {"normalized_actions": actions, "action_buffer": np.zeros((1, 8, 8), np.float32)}
        policy = StreamingLiberoPolicy(predict, {
            "min": [-1.] * 7, "max": [1.] * 7, "mask": [True] * 6 + [False]})
        for step in range(9):
            image = np.full((32, 32, 3), step * 10, np.uint8)
            action = policy.act(image, image, np.zeros(8), "pick up cup")
            self.assertEqual(action.shape, (7,))
            self.assertEqual(action[6], -1.)
        self.assertEqual(len(calls), 3)
        self.assertTrue(np.all(calls[0]["action_prefix"] == 0))
        self.assertTrue(np.all(calls[1]["action_prefix"][:4] == 0))
        self.assertTrue(np.all(calls[1]["action_prefix"][4:, 6] == 1))
        self.assertTrue(np.all(calls[2]["action_prefix"][:, 6] == 1))
        self.assertEqual([round(float(frame.mean()) * 255) for frame in calls[1]["image"]],
                         [0, 10, 20, 30, 40])
        self.assertIsNone(calls[0]["action_buffer"])
        self.assertIsNotNone(calls[1]["action_buffer"])
        policy.reset("pick up cup")
        policy.act(image, image, np.zeros(8), "pick up cup")
        self.assertIsNone(calls[-1]["action_buffer"])
        self.assertTrue(np.all(calls[-1]["action_prefix"] == 0))


if __name__ == "__main__":
    unittest.main()
