import torch as t

class LinearNoiseSampler(nn.Module):
    def __init__(self,timesteps,beta_begin,beta_end):
        self.timesteps = timesteps
        self.beta_begin=beta_begin
        self.beta_end=beta_end

        self.betas=t.linspace(timesteps,beta_begin,beta_end)
        self.alpha=1-self.betas

        self.alpha_cumulative = t.cumprod(self.alpha,dim=0)
        self.alpha_cumulative_sqrt =  t.sqrt(self.alpha_cumulative)
        self.alpha_cumulative_1_sqrt= t.sqrt(1-self.alpha_cumulative)


    def added_noise(self,x_0,time,noise):
        x_0_shape=x_0.shape
        batch_size=x_0_shape[0]

        sqrt_alpha_cumulative=self.alpha_cumulative_sqrt[time].reshape(batch_size,1,1,1)  #broadcast so that all images have their alpha t
        sqrt_alpha_cumulative_1=self.alpha_cumulative_1_sqrt[time].reshape(batch_size,1,1,1)
       
        x_t=(sqrt_alpha_cumulative*x_0)+(sqrt_alpha_cumulative_1*noise)
        return x_t
    

    def sample_previous_timestep(self,x_t,time,noise_pred):
        x_0=(x_t-(self.alpha_cumulative_1_sqrt[time]*noise_pred*x_t))/self.alpha_cumulative_sqrt[time]

        x_0 = t.clamp(x_0,-1,1)
        mean_pred=(1/t.sqrt(self.alpha[time]))*(x_t-(((self.betas[time])/(self.alpha_cumulative_1_sqrt[time]))*noise_pred))

        if t==0:
            return mean_pred , x_0
        
        else:
            var=((self.betas[time])*(1-self.alpha_cumulative[time-1]))/(1-self.alpha_cumulative[time])

            sigma = var**0.5
            #reparametrisation trick
            return t.randn(x_t.shape).to(x_t.device)*sigma + mean_pred , x_0





             







        