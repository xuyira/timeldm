import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            vae_model,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            num_classes=None,
            rescale_timesteps=True,
            eta=0.,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()
        self.vae_model = vae_model
        # we don't want to train any parameters of the vae model
        for param in vae_model.parameters():
            param.requires_grad = False

        self.eta = eta
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.rescale_timesteps = rescale_timesteps
        self.ff_weight = math.sqrt(self.seq_length) / 5

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=0., resid_pdrop=0., mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, num_classes=num_classes, n_embd=d_model, **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default( sampling_timesteps, timesteps)  
        # default num sampling timesteps to number of timesteps at training
           

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, padding_masks=None, y=None):
        trend, season = self.model(x, t, padding_masks=padding_masks, y=y)
        model_output = trend + season
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None, y=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks, y)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True, y=None):
        _, x_start = self.model_predictions(x, t, y=y)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        y = model_kwargs['y'] if model_kwargs is not None else None
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised, y=y)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, model_kwargs=model_kwargs
            )
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        # pred_img = self.q_sample(x_start, batched_times)
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, shape):
        batch, device = shape[0], self.betas.device
        
        img = torch.randn(shape, device=device)
        
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta
            

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, batch_size=16, cond=False, model_kwargs=None, cond_fn=None):
        
        feature_size, seq_length = self.feature_size, self.seq_length
        print('feature_size, seq_length', feature_size, seq_length)
        if cond:
            
            sample_fn = self.fast_sample_cond if self.fast_sampling else self.sample_cond
            
            return sample_fn((batch_size, seq_length, feature_size), model_kwargs=model_kwargs, cond_fn=cond_fn)
        
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        print('sample_fn', sample_fn)
        sample_ldm = sample_fn((batch_size, seq_length, feature_size))
        sample_vae = self.vae_decode(sample_ldm)
        return sample_vae

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None, y=None, use_ff=True):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        # print('target', target)
        if target is None:
            target = x_start
        
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        print('x',x.shape)
        model_out = self.output(x, t, padding_masks, y)
        print('------------')
        print('model_out',model_out.shape)
        print('target', target.shape)
        print('------------')
        train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if use_ff:
            fft1, fft2 = torch.fft.fft(model_out.transpose(1, 2), norm='forward'), torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        print('--------ddpm-----------')
        print('x', x.shape)
        x = self.vae_encode(x)
        print('x', x.shape)
                
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        print('feature_size', feature_size)
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season = self.model(x, t)
        return trend, season

    def fast_sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        y = model_kwargs['y'] if model_kwargs is not None else None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised, y=y)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
    

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        # batch, device, total_timesteps, sampling_timesteps, eta = \
        #     shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        x_self_cond=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                # print(infill_loss, f'start_{i}', logp_term.item(), t[0].item(), sigma.mean().item())
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    
    
    @torch.no_grad()
    def sample_replace(self, shape, target, partial_mask=None, clip_denoised=True):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample_replace(img, t, target, partial_mask, clip_denoised)

        img[partial_mask] = target[partial_mask]
        return img


    @torch.no_grad()
    def fast_sample_replace(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True):
        batch, device, total_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img
    
    def p_sample_replace(self, x, t: int, target, partial_mask=None,
                         clip_denoised=True, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        y = model_kwargs['y'] if model_kwargs is not None else None
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised, y=y)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, model_kwargs=model_kwargs
            )
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        # pred_img = self.q_sample(x_start, batched_times)
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]
        return pred_img, x_start
    
    def sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                         model_kwargs=model_kwargs)
        return img
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x=x, t=self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart
    
    @torch.no_grad()
    def vae_encode(self, x: torch.Tensor):
        """
        Encodes and quantizes an input image

        Args:
            x: the image to encode
        Returns:
            x: encoded and quantized image
        """
        print('--------- get data from vae_encode --------')
        x = self.vae_model.get_ldm_inputs(x)
        # x = self.vae_model.quantize(x)
        x = x.transpose(1, 2) 
        print('x',x.shape)
        return x

    @torch.no_grad()
    def vae_decode(self, x: torch.Tensor):
        """
        Decode latent representation to an image

        Args:
            x: latent representation to decode
        Returns:
            x_hat: decoded latent representation
        """
        print('-------vae_decode---------')
        print('x', x.shape)
        x = x.transpose(1, 2) 
        print('x', x.shape)
        x_hat = self.vae_model.reconstruct_ldm_outputs(x)
        print('x_hat')
        return x_hat


if __name__ == '__main__':
    # args.feat_dim 5
    # args.n_layer_enc 1
    # args.n_layer_dec 2
    # args.seq_length 24
    # args.d_model 64
    # args.sample_steps 500
    # args.time_steps 500
    # args.loss_type l1
    # args.loss_type l1
    # args.n_heads 4
    # args.mlp_times 4
    # class Diffusion_TS(nn.Module):
    # def __init__(
    #         self,
    #         # vae_model,
    #         seq_length,
    #         feature_size,
    #         n_layer_enc=3,
    #         n_layer_dec=6,
    #         d_model=None,
    #         timesteps=1000,
    #         sampling_timesteps=None,
    #         loss_type='l1',
    #         beta_schedule='cosine',
    #         n_heads=4,
    #         mlp_hidden_times=4,
    #         num_classes=None,
    #         rescale_timesteps=True,
    #         eta=0.,
    #         **kwargs
    # ):
    from omegaconf import OmegaConf
    import os
    def load_model_checkpoint(model, filepath, device):
        """ Load model checkpoint """
        if os.path.isfile(filepath):
            ckpt = torch.load(filepath, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            raise ValueError("Checkpoint path '{}' does not exist!".format(filepath))

        return model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    config = OmegaConf.load(r'F:\00_python-code\ST\Diffusion-TS\Models\vae\config.yaml')
    from Models.vae.ae_kl import AutoencoderKL
    # create model and optimizer
    hparams = config.autoencoderkl.params
    vae_model = AutoencoderKL(hparams["embed_dim"], hparams)
    vae_path = r'F:\00_python-code\ST\Diffusion-TS\checkpoint_vae\checkpoint_bst.pt'
    vae_model = load_model_checkpoint(vae_model, vae_path, device)
    vae_model.to(device)
  
    model = Diffusion_TS(vae_model=vae_model,
                         feat_dim=5,
                         n_layer_enc=1,
                         n_layer_dec =2,
                         seq_length =24,
                         d_model= 64,
                         sample_steps =500,
                         time_steps =500,
                         loss_type ='l1',
                         n_heads =4,
                         mlp_times =4
                         ).to(device)
    
    input = torch.randn((1, 8, 12)).to(device)
    
    reconstruction, z_mu, z_sigma = model(input)

