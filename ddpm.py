import torch as t
import torch.nn as nn

class LinearNoiseSampler(nn.Module):
    def __init__(self,timesteps,beta_begin,beta_end):
        super().__init__()
        self.timesteps = timesteps
        self.beta_begin=beta_begin
        self.beta_end=beta_end

        # register_buffer so tensors move to GPU with .to(device)
        # avoid 0 and 1 at endpoints to prevent division by zero in sampling
        self.register_buffer('lambdas', t.linspace(1e-4,1-1e-4,timesteps))
        self.register_buffer('betas', t.linspace(beta_begin,beta_end, timesteps))
        self.register_buffer('alpha', 1-self.betas)

        self.register_buffer('alpha_cumulative', t.cumprod(self.alpha,dim=0))
        self.register_buffer('alpha_cumulative_sqrt', t.sqrt(self.alpha_cumulative))
        self.register_buffer('alpha_cumulative_1_sqrt', t.sqrt(1-self.alpha_cumulative))
        # clamp to prevent negative values which produce NaN in sqrt during sampling
        self.register_buffer('deltas', t.clamp((1-self.alpha_cumulative) - ((self.lambdas**2)* self.alpha_cumulative), min=1e-8))


    def added_noise(self,x_0,time,noise,x_hat):
        x_0_shape=x_0.shape
        batch_size=x_0_shape[0]

        sqrt_alpha_cumulative=self.alpha_cumulative_sqrt[time].reshape(batch_size,1,1,1)  #broadcast so that all images have their alpha t
        # sqrt_alpha_cumulative_1=self.alpha_cumulative_1_sqrt[time].reshape(batch_size,1,1,1)
        lambda_t = self.lambdas[time].reshape(batch_size,1,1,1)
        delta_t = self.deltas[time].reshape(batch_size,1,1,1)
        x_t = ((1 - lambda_t) * (sqrt_alpha_cumulative * x_0)) + (lambda_t * sqrt_alpha_cumulative * x_hat) + (t.sqrt(delta_t) * noise)
        return x_t


    def sample_previous_timestep(self,x_t,time,noise_pred,x_hat):

        # clamp time-1 to 0 to avoid negative index wrapping to the last element
        time_prev = t.clamp(time - 1, min=0)

        # predict x_0 — loss_coeff simplifies to standard DDPM epsilon
        x_0 = (x_t-(self.alpha_cumulative_1_sqrt[time]*noise_pred))/self.alpha_cumulative_sqrt[time]
        x_0 = t.clamp(x_0,-1,1)

        lambda_t = self.lambdas[time]
        lambda_prev = self.lambdas[time_prev]
        delta_t = self.deltas[time]
        delta_prev = self.deltas[time_prev]
        alpha_t=self.alpha[time]
        sqrt_alpha_t = t.sqrt(self.alpha[time])
        alpha_bar_t = self.alpha_cumulative[time]
        sqrt_alpha_bar_prev = self.alpha_cumulative_sqrt[time_prev]
        a = ((1 - lambda_t) / (1 - lambda_prev))
        delta_t_given_prev = t.clamp(delta_t - ((alpha_t * delta_prev) * (a ** 2)), min=1e-8)
        phi_x = ((delta_prev*a*sqrt_alpha_t)+((1-lambda_prev)*delta_t_given_prev/sqrt_alpha_t))/delta_t
        phi_x_hat = (sqrt_alpha_bar_prev/delta_t)*((lambda_prev*delta_t)-(lambda_t*alpha_t*delta_prev*a))
        phi_noise = ((1-lambda_prev)*delta_t_given_prev*(t.sqrt(1-alpha_bar_t)))/(delta_t*sqrt_alpha_t)
        mean_pred= (phi_x*x_t)+(phi_x_hat*x_hat)-(phi_noise*noise_pred)

        if time==0:
            return mean_pred , x_0,x_hat

        else:
            z=t.randn(x_t.shape).to(x_t.device)
            # return mean_pred + (z * t.sqrt(delta_t)) , x_0,x_hat #reparameterisation for sampling
            return mean_pred + (z * t.sqrt(delta_t_given_prev)) , x_0,x_hat  # Algorithm 2: posterior variance δ_{t|t-1}

    def loss_coeff(self,noise,time,x_0,x_hat):
        x_0_shape=x_0.shape
        batch_size=x_0_shape[0]

        alpha_cum_1_sqrt_inv=(1/self.alpha_cumulative_1_sqrt[time]).reshape(batch_size,1,1,1)
        alpha_cum_sqrt=(self.alpha_cumulative_sqrt[time]).reshape(batch_size,1,1,1)
        lambda_t = self.lambdas[time].reshape(batch_size,1,1,1)
        delta_t_sqrt = t.sqrt(self.deltas[time]).reshape(batch_size,1,1,1)

        coeff=alpha_cum_1_sqrt_inv*((lambda_t*alpha_cum_sqrt*(x_hat-x_0))+(delta_t_sqrt*noise))

        return coeff
