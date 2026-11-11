"""Episode-local SDP state. Call observe/act at every 20 Hz environment control step."""

from collections import deque

import numpy as np

from DiT4DiT.dataloader.gr00t_lerobot.streaming_data import encode_libero_state, pack_rgb_views


class StreamingLiberoPolicy:
    def __init__(self, predict, action_stats, k=4):
        if k != 4:
            raise ValueError("This first transfer experiment is fixed to K=4")
        self.predict = predict
        self.stats = action_stats
        self.k = k
        self.reset()

    def reset(self, instruction=None):
        self.instruction = instruction
        self.images = deque(maxlen=5)
        self.executed = deque(maxlen=8)
        self.queue = deque()
        self.buffer = None
        self.control_step = 0

    def _prefix(self):
        prefix = np.zeros((8, 8), dtype=np.float32)
        if self.executed:
            prefix[-len(self.executed):] = np.stack(self.executed)
        return prefix

    def act(self, primary, wrist, state, instruction):
        if self.instruction is None:
            self.instruction = instruction
        if instruction != self.instruction:
            raise ValueError("Reset explicitly at episode boundaries before changing instruction")
        self.images.append(pack_rgb_views(primary, wrist))
        if not self.queue:
            observed = list(self.images)
            history = [observed[0]] * (5 - len(observed)) + observed
            state = encode_libero_state(state)
            example = {
                "image": history, "lang": instruction, "state": state,
                "action_prefix": self._prefix(), "action_buffer": self.buffer,
            }
            result = self.predict([example])
            future = np.asarray(result["normalized_actions"], dtype=np.float32)
            buffer = np.asarray(result["action_buffer"], dtype=np.float32)
            if future.shape != (1, 4, 8) or buffer.shape != (1, 8, 8):
                raise ValueError(f"Unexpected SDP response shapes: {future.shape}, {buffer.shape}")
            if not np.isfinite(future).all() or not np.isfinite(buffer).all():
                raise ValueError("Non-finite SDP output")
            self.buffer = buffer[0].copy()
            self.queue.extend(future[0].copy())

        normalized = np.clip(self.queue.popleft(), -1.0, 1.0)
        normalized[6] = float(normalized[6] >= 0.5)
        normalized[7] = 0.0
        low = np.asarray(self.stats["min"], dtype=np.float32)[:7]
        high = np.asarray(self.stats["max"], dtype=np.float32)[:7]
        mask = np.asarray(self.stats.get("mask", [True] * 6 + [False]), dtype=bool)[:7]
        if mask.shape != (7,) or mask[6]:
            raise ValueError("LIBERO gripper must remain unnormalized, with six continuous min/max dimensions")
        command = np.where(mask, (normalized[:7] + 1) * 0.5 * (high - low) + low,
                           normalized[:7]).astype(np.float32)
        command[6] = 1.0 - 2.0 * float(command[6] > 0.5)
        # Record the clipped/thresholded command, not an unexecuted model prediction.
        self.executed.append(normalized.copy())
        self.control_step += 1
        return command
