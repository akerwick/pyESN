import torch
import torch.nn as nn
import torch.nn.functional as F


def correct_dimensions(s, targetlength):
    """Checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D tensor
        targetlength: expected length of s

    Returns:
        None if s is None, else tensor of length targetlength
    """
    if s is not None:
        s = torch.tensor(s, dtype=torch.float32)
        if s.dim() == 0:
            s = s.repeat(targetlength)
        elif s.dim() == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


class ESN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, silent=True):
        """
        Args:
            n_inputs: Number of input dimensions
            n_outputs: Number of output dimensions
            n_reservoir: Number of reservoir neurons
            spectral_radius: Spectral radius of the recurrent weight matrix
            sparsity: Proportion of recurrent weights set to zero
            noise: Noise added to each neuron (regularization)
            input_shift: Scalar or tensor of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: Scalar or tensor of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: If True, feed the target back into output units
            teacher_scaling: Factor applied to the target signal
            teacher_shift: Additive term applied to the target signal
            out_activation: Output activation function (applied to the readout)
            inverse_out_activation: Inverse of the output activation function
            random_state: Random seed or None for default initialization
            silent: Suppress messages
        """
        super(ESN, self).__init__()

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state
        self.teacher_forcing = teacher_forcing
        self.silent = silent

        self.set_random_seed()
        self.set_device()

        self.initweights()

    def set_random_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.random_state)

        # If using GPU, also set the seed for CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.silent:
            print(f'running on {self.device}')

    def initweights(self):
        """Initializes the recurrent, input, and feedback weights."""
        W = torch.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[torch.rand_like(W) < self.sparsity] = 0
        radius = torch.max(torch.abs(torch.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)
        self.W_in = torch.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.W_feedb = torch.rand(self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        """Performs one update step."""
        if self.teacher_forcing:
            preactivation = (torch.matmul(self.W, state)
                             + torch.matmul(self.W_in, input_pattern)
                             + torch.matmul(self.W_feedb, output_pattern))
        else:
            preactivation = (torch.matmul(self.W, state)
                             + torch.matmul(self.W_in, input_pattern))
        return torch.tanh(preactivation) + self.noise * (torch.rand(self.n_reservoir) - 0.5)

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = torch.matmul(inputs, torch.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, inspect=False):
        """Trains the ESN using the given inputs and outputs."""
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(1)
        if outputs.dim() < 2:
            outputs = outputs.unsqueeze(1)

        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent:
            print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        states = torch.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])
        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        # we'll disregard the first few states:
        transient = min(int(inputs.shape[1] / 10), 100)
        extended_states = torch.cat((states, inputs_scaled), dim=1)
        #self.W_out = torch.linalg.pinv(extended_states[transient:, :]) @ self.inverse_out_activation(teachers_scaled[transient:, :])
        # Solve for W_out:
        self.W_out = torch.matmul(torch.linalg.pinv(extended_states[transient:, :]),
                            self.inverse_out_activation(teachers_scaled[transient:, :])).T

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]
        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()

        if not self.silent:
            print("training error:")
        # apply learned weights to the collected states:
        #pred_train = self._unscale_teacher(self.out_activation(extended_states @ self.W_out))
        pred_train = self._unscale_teacher(self.out_activation(
            torch.matmul(extended_states, self.W_out.T)))
        if not self.silent:
            print(torch.sqrt(torch.mean((pred_train - outputs)**2)))
        return pred_train

    def predict(self, inputs, continuation=True):
        """Generates predictions using the trained ESN."""
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(1)
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = torch.zeros(self.n_reservoir)
            lastinput = torch.zeros(self.n_inputs)
            lastoutput = torch.zeros(self.n_outputs)

        inputs = torch.cat([lastinput.unsqueeze(0), self._scale_inputs(inputs)])
        states = torch.cat([laststate.unsqueeze(0), torch.zeros((n_samples, self.n_reservoir))])
        outputs = torch.cat([lastoutput.unsqueeze(0), torch.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(self.W_out @ torch.cat([states[n + 1, :], inputs[n + 1, :]]))

        return self._unscale_teacher(self.out_activation(outputs[1:]))
