"""Transport-only bridge to the unmodified official ModelClient, with K=4."""
from collections import deque
import numpy as np


class OfficialPolicy:
    def __init__(self, predict, statistics, k=4):
        assert k in (1,2,4,6,8)
        from examples.LIBERO.eval_files.model2libero_interface import ModelClient
        owner = self

        class LocalTransport:
            def predict_action(self, request):
                assert len(request['examples']) == 1
                assert len(request['examples'][0]['image']) == 1
                result = predict(request['examples'])
                chunk = np.asarray(result['normalized_actions'])
                assert chunk.shape == (1, 8, 8) and np.isfinite(chunk).all()
                owner.chunks.append(chunk[0].copy())
                return {'data': result}

        # Bypass only the network constructor. step(), view/state transforms,
        # and unnormalize_actions() are the original, unmodified official code.
        self.client = ModelClient.__new__(ModelClient)
        self.client.client = LocalTransport()
        self.client.action_norm_stats = statistics
        self.client.action_chunk_size = k
        self.client.max_state_dim = 16
        self.client.image_size = [224, 224]
        self.client.horizon = 0
        self.client.image_history = deque(maxlen=0)
        self.client.action_ensemble = False
        self.client.use_ddim = True
        self.client.num_ddim_steps = 10  # Legacy request field; actual head uses 4.
        self.reset()

    def reset(self, instruction=None):
        self.client.reset(instruction)
        self.chunks = []
        self.step = 0

    def act(self, primary, wrist, state, instruction):
        out = self.client.step({'image': primary, 'wrist_image': wrist,
                                'state': state, 'lang': instruction}, self.step)
        raw = out['raw_action']
        action = np.concatenate([raw['world_vector'], raw['rotation_delta'],
                                 1 - 2 * (raw['open_gripper'] > 0.5)]).astype(np.float32)
        self.step += 1
        return action
