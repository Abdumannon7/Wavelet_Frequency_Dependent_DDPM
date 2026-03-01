import torch as t

class LinearNoiseSampler(nn.Module):
    def __init__(self,timesteps,beta_begin,beta_end):
        self.timesteps = timesteps
        self.beta_begin=beta_begin
        self.beta_end=beta_end

        self.lambdas = t.linspace(timesteps,0,1)  #to account for conditioned element
        self.betas=t.linspace(timesteps,beta_begin,beta_end)
        self.alpha=1-self.betas

        self.alpha_cumulative = t.cumprod(self.alpha,dim=0)
        self.alpha_cumulative_sqrt =  t.sqrt(self.alpha_cumulative)
        self.alpha_cumulative_1_sqrt= t.sqrt(1-self.alpha_cumulative)
        self.deltas = (1-self.alpha_cumulative) - ((self.lambdas**2)* self.alpha_cumulative)


    def added_noise(self,x_0,time,noise,x_hat):
        x_0_shape=x_0.shape
        batch_size=x_0_shape[0]

        sqrt_alpha_cumulative=self.alpha_cumulative_sqrt[time].reshape(batch_size,1,1,1)  #broadcast so that all images have their alpha t
        # sqrt_alpha_cumulative_1=self.alpha_cumulative_1_sqrt[time].reshape(batch_size,1,1,1)
        lambda_t = self.lambdas[time].reshape(batch_size,1,1,1)
        delta_t = self.deltas[time].reshape(batch_size,1,1,1)
        # x_t=(sqrt_alpha_cumulative*x_0)+(sqrt_alpha_cumulative_1*noise)
        x_t= ((1-lambda_t)*(sqrt_alpha_cumulative*x_0) )+(lambda_t*sqrt_alpha_cumulative*x_hat)+(delta_t*noise)
        return x_t
    

    def sample_previous_timestep(self,x_t,time,noise_pred,x_hat):

        sqrt_alpha_cumulative=self.alpha_cumulative_sqrt[time]
        delta_t = self.deltas[time]
        delta_t_1=self.deltas[time-1]
        lambda_t = self.lambdas[time]
        lambda_t_1 =self.lambdas[time-1]

        sqrt_alpha=t.sqrt(self.alpha[time])
        # x_0=(x_t-(self.alpha_cumulative_1_sqrt[time]*noise_pred))/self.alpha_cumulative_sqrt[time]
        x_0 = (x_t-((lambda_t*sqrt_alpha_cumulative*x_hat)+(delta_t*noise_pred)))/((1-lambda_t)*(sqrt_alpha_cumulative))

        x_0 = t.clamp(x_0,-1,1)

        delta_t_t1 = delta_t-(((1-lambda_t)/(1-lambda_t_1))**2)*self.alpha[time]*delta_t_1
        
        phi_x= ((delta_t_1*(1-lambda_t)*sqrt_alpha)/(delta_t*(1-lambda_t_1))) + (((1-lambda_t_1)*delta_t_t1)/(delta_t*sqrt_alpha))
        phi_x_hat = ((lambda_t_1*delta_t)-((lambda_t*(1-lambda_t)*self.alpha[time]*delta_t_1)/(1-lambda_t_1)))*self.alpha_cumulative_sqrt[time-1]/delta_t
        phi_noise = (1-lambda_t_1*delta_t_t1*self.alpha_cumulative_1_sqrt)/(delta_t*sqrt_alpha)
  
        # mean_pred=(1/t.sqrt(self.alpha[time]))*(x_t-(((self.betas[time])/(self.alpha_cumulative_1_sqrt[time]))*noise_pred))
        mean_pred = (phi_x*x_t)+(phi_x_hat*x_hat)-(phi_noise*noise_pred)
        





  
        mean_pred=0
        if time==0:
            return mean_pred , x_0
        
        else:
            # var=((self.betas[time])*(1-self.alpha_cumulative[time-1]))/(1-self.alpha_cumulative[time])
            sigma = t.sqrt(delta_t)

            # sigma = var**0.5
            #reparametrisation trick
            return t.randn(x_t.shape).to(x_t.device)*sigma + mean_pred , x_0

   


    def loss_coeff(self,noise,time,x_0,x_hat):
        x_0_shape=x_0.shape
        batch_size=x_0_shape[0]

        alpha_cum_1_sqrt_inv=(1/self.alpha_cumulative_1_sqrt[time]).reshape(batch_size,1,1,1)
        alpha_cum_sqrt=(self.alpha_cumulative_sqrt[time]).reshape(batch_size,1,1,1)
        lambda_t = self.lambdas[time].reshape(batch_size,1,1,1)
        delta_t_sqrt = t.sqrt(self.deltas[time]).reshape(batch_size,1,1,1)

        coeff=alpha_cum_1_sqrt_inv*((lambda_t*alpha_cum_sqrt(x_hat-x_0))+(delta_t_sqrt*noise))

        return coeff








             







        