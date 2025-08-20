import torch 
from utils import tcheby, tcheby_der, device
from deepinv.sampling import DiffPIR  as dinv_DiffPIR 
from deepinv.optim.data_fidelity import L2
from diffusers import LCMScheduler, AutoPipelineForText2Image 
import os
from torchvision.utils import save_image


def _noise_pred_cond_y_15(latents, t: int, encoder_hidden_states, guidance_scale,
                          pipe, logdir, y_guidance,
                          forward_model, sigma_y):
    with torch.no_grad():
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # Format timestep correctly
        t_tensor = torch.tensor([t], dtype=torch.float16).to("cuda")

        # Forward pass through UNet
        noise_pred = pipe.unet(
            latent_model_input,
            t_tensor,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # Split the outputs for CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (noise_pred_uncond + 
                      guidance_scale * (noise_pred_text - noise_pred_uncond))
        # Compute z0_pred
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        z0_pred = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_pred)

    # decode
    with torch.no_grad():
        x = pipe.vae.decode(z0_pred / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)

    df = torch.norm(forward_model(x.float()) - y_guidance).item()
    decoder_std, decoder_L = 0.02, 1
    var_x_zt = decoder_std**2 + (1-alpha_t) * decoder_L**2

    # setup for gaussian deblurring
    if t>300:
        delta = 1*df/(1e3)
    else:
        delta = 4*df/(1e4)

    #print(f"delta at step {t}: ", "%.2f" % delta)
    with torch.no_grad():
        prox_x = forward_model.prox_l2(x.float().detach().clone(), y=y_guidance, gamma=delta*var_x_zt/(sigma_y**2))
        # encode
        qz= pipe.vae.encode(prox_x.clip(-1,1).half())
        mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor

        z0_pred_cond_y = mu_z

        noise_pred_cond_y = (torch.sqrt(1/(1-alpha_t))*latents - 
                             torch.sqrt(alpha_t/(1-alpha_t))*z0_pred_cond_y)

    log_image_dict = {'x': x, 'prox': prox_x}

    logdir_iter = os.path.join(logdir, 'iter')
    os.makedirs(logdir_iter, exist_ok=True)

    for k, v in log_image_dict.items():
        save_image(torch.clamp(v * 0.5 + 0.5, 0, 1), 
                   os.path.join(logdir_iter, f'{t:3d}_{k}.png'))

    return x, noise_pred_cond_y
    

class Sampler:
    def __init__(self, gradU, gamma, X_init, proj):
        self.gradU = gradU
        self.gamma = gamma
        self.proj = proj
        self.X = X_init.clone().to(device)

    def __call__(self, *args, **kwargs):
        return self.proj(self.X)

    def update_state(self, X):
        self.X = X.clone().to(device)


class ULA(Sampler):
    def __init__(self, gradU, gamma, X_init, proj):
        super().__init__(gradU, gamma, X_init, proj)

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            self.X = self.proj(self.X - self.gamma * self.gradU(self.X, *args, **kwargs) + 
                               torch.sqrt(2*self.gamma) * torch.randn_like(self.X))
        return self.X


class SKROCK(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, s=3, eta=0.05):
        super().__init__(gradU, gamma, X_init, proj)
        self.s, self.eta = torch.tensor(s), torch.tensor(eta)
        w0 = 1 + self.eta / (self.s ** 2)
        w1 = tcheby(w0, s) / tcheby_der(w0, s)
        self.muj, self.nuj, self.kj = torch.zeros(s+1, device=device), torch.zeros(s+1 , device=device), torch.zeros(s+1, device=device)
        self.muj[1] = w1 / w0
        self.nuj[1] = self.s * w1 / 2.
        self.kj[1] = self.s * w1 / w0
        for j in range(2, s + 1):
            self.muj[j] = 2 * w1 * tcheby(w0, j-1) / tcheby(w0, j)
            self.nuj[j] = 2 * w0 * tcheby(w0, j-1) / tcheby(w0, j)
        self.kj[2:] = 1 - self.nuj[2:]

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            K = self.X
            Z = torch.sqrt(2*self.gamma)*torch.randn_like(self.X)
            Kp = (self.X - self.muj[1]*self.gamma*self.gradU(self.X + self.nuj[1]*Z, *args, **kwargs) +
                  self.kj[1]*Z)
            for j in range(2, self.s + 1):
                Kpp = - self.muj[j]*self.gamma*self.gradU(Kp, *args, **kwargs) + self.nuj[j]*Kp + self.kj[j]*K
                K, Kp = Kp, Kpp
            self.X = self.proj(Kpp.clone())
        return self.X


class Gaussian(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, Q, D):
        super().__init__(None, None, X_init, proj)
        self.Q, self.D = Q, D
        self.d = self.Q.shape[0]
    def __call__(self, mean):
        with torch.no_grad():
            self.X = self.proj((self.Q @ torch.sqrt(torch.diag(self.D))) @ torch.randn(self.d, device=device) + torch.reshape(mean, (-1,)))
        return self.X
        

class GaussianDiag(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, d, sigma):
        super().__init__(None, None, X_init, proj)
        self.sigma, self.d = sigma, d
        self.d = d
    def __call__(self, mean):
        with torch.no_grad():
            self.X = self.proj(torch.randn((self.X.shape[0], 1, self.d), device=device)*self.sigma + mean)
        return self.X


class DiffPIR(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, physics, denoiser, verbose=False, 
                 batch_size=1, max_iter=500, lambda_=7.):
        super().__init__(None, None, X_init, proj)
        self.model = dinv_DiffPIR(data_fidelity=L2(), model=denoiser, device=device, 
                                  verbose=verbose, max_iter=max_iter, lambda_=lambda_)
        self.p = physics
        self.batch_size = batch_size
        
    def __call__(self, y,  *args, **kwargs):
        self.X = self.model(y.repeat(self.batch_size, 1, 1, 1), self.p)
        return self.X

class LATINO(Sampler):
    def __init__(self, gradU, gamma, X_init, proj, physics, model_path, adapter_path, prompt,
                 sigma, inference_steps=8, guidance_scale=1.2,
                 verbose=False, batch_size=1):
        super().__init__(None, None, X_init, proj)
        self.p = physics
        self.batch_size = batch_size
        self.prompt = prompt
        self.sigma_y_norm = 2*sigma # ?? * 4 ?
        
        # load models
        pipe = AutoPipelineForText2Image.from_pretrained(model_path, 
                                                         torch_dtype=torch.float16, 
                                                         variant="fp16")
        pipe.to(device)
    
        # load and fuse lcm lora
        pipe.load_lora_weights(adapter_path)
        pipe.fuse_lora()
        self.pipe = pipe
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.inference_steps = inference_steps
        self._reset_scheduler()
        self.guidance_scale = guidance_scale  # CFG scale
        # prepare text embeddings
        prompt = [prompt]

        tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
        # Encode the prompt to conditioning embeddings
        text_inputs = tokenizer(prompt, padding="max_length", 
                                max_length=tokenizer.model_max_length,
                                return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
        # Create unconditional (empty) prompt embeddings for CFG
        uncond_inputs = tokenizer([""] * len(prompt),  # Empty prompt for unconditional guidance
                                  padding="max_length", max_length=tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")

        # Encode unconditional prompt
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]
        # Concatenate unconditional and conditional embeddings for CFG
        self.text_embeddings_cfg = torch.cat([uncond_embeddings, 
                                              text_embeddings], dim=0)
        #self.AA_t_inv = torch.linalg.inv(self.p.A_A_adjoint(torch.ones_like(X_init)))

    def _reset_scheduler(self):
        self.pipe.scheduler.set_timesteps(self.inference_steps, device=device)

    def __call__(self, y):
        y_norm = 2*y - 1

        # Compute x_init = A^T (AA^T)^-1 y using the transpose operator
        x_init = self.p.A_adjoint(y_norm).clip(-1, 1).half()        
        with torch.no_grad():
            qz = self.pipe.vae.encode(x_init)
        mu_z = qz.latent_dist.mean * self.pipe.vae.config.scaling_factor
        noise = torch.randn_like(mu_z)
        latents = self.pipe.scheduler.add_noise(mu_z, noise=noise, timesteps=torch.tensor([999]))

        for i, timestep in enumerate(self.pipe.scheduler.timesteps):
            #print(f"Step {i + 1}: Timestep {timestep}")
            with torch.no_grad():
                x_0, noise_pred =_noise_pred_cond_y_15(
                    latents=latents,
                    t=timestep,
                    encoder_hidden_states=self.text_embeddings_cfg,
                    guidance_scale=self.guidance_scale,
                    pipe=self.pipe,
                    logdir="test",
                    y_guidance=y_norm,
                    forward_model=self.p,
                    sigma_y=self.sigma_y_norm,
                )
                latents = self.pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
        self._reset_scheduler()  # for next call to self.sample

        with torch.no_grad():
            # Decode latents to image
            decoded_image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample

            self.X = (decoded_image / 2 + 0.5).clamp(0, 1)  # Normalize latents to image space
            return self.X
