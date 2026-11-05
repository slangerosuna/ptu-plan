"""REINFORCE signs, baselines, on-policy checks, and held-out isolation."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import finetune_rl as rl

@unittest.skipUnless(HAS_TORCH,"PyTorch is optional; run with the training environment")
class ReinforceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        class Actor:
            def __init__(self,requires_grad=True):
                self.theta=torch.tensor([0.0,0.0],dtype=torch.float64,requires_grad=requires_grad)
            def tensor(self,value):return torch.as_tensor(value,dtype=self.theta.dtype)
            def __call__(self,observations):
                return self.theta.repeat(len(observations)),self.theta.new_zeros(2*len(observations)),list(range(0,2*(len(observations)+1),2))
        self.actor=Actor();self.anchor=Actor(False)
        self.episode=dict(problem_id="p",split="Train",sampled=True,reward=1.0,log_cost=0.0,solved=True,probabilities=[[0.5,0.5]],decisions=[dict(observation={},action=0)])
        self.baseline=dict(problem_id="p",sampled=False,reward=0.0)
    def loss(self):return rl.policy_loss(self.actor,self.anchor,[self.episode],[self.baseline],0.0,0.0,0.0)[0]
    def test_success_advantage_increases_sampled_action_probability(self):
        self.loss().backward()
        torch.testing.assert_close(self.actor.theta.grad,torch.tensor([-0.5,0.5],dtype=torch.float64))
    def test_failure_advantage_discourages_sampled_action(self):
        self.episode["reward"]=-1.0
        self.loss().backward()
        torch.testing.assert_close(self.actor.theta.grad,torch.tensor([0.5,-0.5],dtype=torch.float64))
    def test_equal_self_critical_return_has_zero_policy_gradient(self):
        self.baseline["reward"]=1.0
        self.loss().backward()
        torch.testing.assert_close(self.actor.theta.grad,torch.zeros(2,dtype=torch.float64))
    def test_terminal_return_sums_decisions_instead_of_averaging_trajectory_length(self):
        self.episode["decisions"]*=2;self.episode["probabilities"]*=2
        self.loss().backward()
        torch.testing.assert_close(self.actor.theta.grad,torch.tensor([-1.0,1.0],dtype=torch.float64))
    def test_validation_episodes_cannot_update_actor(self):
        self.episode["split"]="Validation"
        with self.assertRaisesRegex(ValueError,"Train"):self.loss()
    def test_stale_behavior_distribution_is_rejected(self):
        self.episode["probabilities"]=[[0.9,0.1]]
        with self.assertRaisesRegex(ValueError,"on-policy"):self.loss()
    def test_baseline_must_match_same_problem(self):
        self.baseline["problem_id"]="other"
        with self.assertRaisesRegex(ValueError,"baseline"):self.loss()

if __name__=="__main__":unittest.main()
