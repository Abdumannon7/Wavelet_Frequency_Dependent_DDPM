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
        # use sqrt(delta_t) not delta_t — noise is scaled by std dev, not variance
        x_t= ((1-lambda_t)*(sqrt_alpha_cumulative*x_0) )+(lambda_t*sqrt_alpha_cumulative*x_hat)+(t.sqrt(delta_t)*noise)
        return x_t
    

    def sample_previous_timestep(self,x_t,time,noise_pred,x_hat):

        sqrt_alpha_cumulative=self.alpha_cumulative_sqrt[time]
        sqrt_alpha=t.sqrt(self.alpha[time])

        # clamp time-1 to 0 to avoid negative index wrapping to the last element
        time_prev = t.clamp(time - 1, min=0)

        # model predicts loss_coeff which simplifies to standard DDPM x_0 recovery
        x_0 = (x_t-(self.alpha_cumulative_1_sqrt[time]*noise_pred))/self.alpha_cumulative_sqrt[time]

        x_0 = t.clamp(x_0,-1,1)

        # use standard DDPM posterior mean — consistent with loss_coeff reducing to standard epsilon
        mean_pred=(1/sqrt_alpha)*(x_t-(((self.betas[time])/(self.alpha_cumulative_1_sqrt[time]))*noise_pred))

        if time==0:
            return mean_pred , x_0,x_hat

        else:
            # standard DDPM posterior variance
            var=((self.betas[time])*(1-self.alpha_cumulative[time_prev]))/(1-self.alpha_cumulative[time])
            sigma = t.sqrt(var)
            return t.randn(x_t.shape).to(x_t.device)*sigma + mean_pred , x_0,x_hat

   


    def loss_coeff(self,noise,time,x_0,x_hat):
        x_0_shape=x_0.shape
        batch_size=x_0_shape[0]

        alpha_cum_1_sqrt_inv=(1/self.alpha_cumulative_1_sqrt[time]).reshape(batch_size,1,1,1)
        alpha_cum_sqrt=(self.alpha_cumulative_sqrt[time]).reshape(batch_size,1,1,1)
        lambda_t = self.lambdas[time].reshape(batch_size,1,1,1)
        delta_t_sqrt = t.sqrt(self.deltas[time]).reshape(batch_size,1,1,1)

        coeff=alpha_cum_1_sqrt_inv*((lambda_t*alpha_cum_sqrt*(x_hat-x_0))+(delta_t_sqrt*noise))

        return coeff








             







        