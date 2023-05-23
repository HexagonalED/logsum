import torch
import numpy as np
import matplotlib

def forward(self, img, *args, **kwargs):
    b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
    assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

    #####
    t = torch.randint(0, self.num_timesteps, (4*b,), device=device).long()
    img_ = img.repeat(1,4,1,1)
    img = img.reshape(4*b, c, h, w) 
    img = self.normalize(img)
    #####
    return self.p_losses(img, t, *args, **kwargs)

def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()