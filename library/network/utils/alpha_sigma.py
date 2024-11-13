import torch

def get_logsnr_alpha_sigma(time):
    def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
        b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
        a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
        return -2. * torch.log(torch.tan(a * t + b))

    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma
