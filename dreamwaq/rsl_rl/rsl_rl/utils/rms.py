import unittest
import torch

class RunningMeanStd(torch.nn.Module):
    def __init__(self,
                 shape,
                 epsilon=1e-4,
                 device = "cpu"):
        super(RunningMeanStd, self).__init__()

        self.device = device
        self.mean = torch.nn.Parameter(torch.zeros(shape), requires_grad=False).to(self.device)
        self.var = torch.nn.Parameter(torch.zeros(shape), requires_grad=False).to(self.device)
        self.count = epsilon  # Changed from torch.tensor to float

    def update(self, x):
        batch_mean, batch_std, batch_count = x.mean(axis=0).to(self.device), x.std(axis=0).to(self.device), x.shape[0]
        batch_var = torch.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean.to(self.device) - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count   # existed data
        m_b = batch_var * batch_count # new data(added)
        M2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count  # Changed from self.count.data = new_count


class TestRunningMeanStd(unittest.TestCase):
    def setUp(self):
        self.running_mean_std = RunningMeanStd((10,))

    def test_initialization(self):
        self.assertTrue(torch.equal(self.running_mean_std.mean, torch.zeros(10)))
        self.assertTrue(torch.equal(self.running_mean_std.var, torch.zeros(10)))
        self.assertAlmostEqual(self.running_mean_std.count, 1e-4, places=8)

    def test_update(self):
        x = torch.randn(100, 10)
        self.running_mean_std.update(x)

        self.assertTrue(torch.allclose(self.running_mean_std.mean, x.mean(axis=0), atol=1e-5))
        self.assertTrue(torch.allclose(self.running_mean_std.var, torch.square(x.std(axis=0)), atol=1e-5))
        self.assertAlmostEqual(self.running_mean_std.count, 100 + 1e-4, places=8)

    def test_update_from_moments(self):
        batch_mean = torch.randn(10)
        batch_var = torch.randn(10)
        batch_count = 100
        self.running_mean_std.update_from_moments(batch_mean, batch_var, batch_count)

        self.assertTrue(torch.allclose(self.running_mean_std.mean, batch_mean, atol=1e-5))
        self.assertTrue(torch.allclose(self.running_mean_std.var, batch_var, atol=1e-5))
        self.assertAlmostEqual(self.running_mean_std.count, batch_count + 1e-4, places=8)


if __name__ == '__main__':
    unittest.main()