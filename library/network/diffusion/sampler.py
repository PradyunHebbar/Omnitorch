import torch
from torch import nn, Tensor
from ..utils import InputType, Source, DistributionInfo, InputType, SourceTuple
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from tqdm import tqdm

class Diffusion_Sampler():
    def __init__(self):
        pass

    def logsnr_schedule_cosine(self, time: Tensor, logsnr_min: float = -20., logsnr_max: float = 20.) -> Tensor:
        logsnr_min = Tensor([logsnr_min]).to(time.device)
        logsnr_max = Tensor([logsnr_max]).to(time.device)
        b = torch.atan(torch.exp(-0.5 * logsnr_max)).to(time.device)
        a = (torch.atan(torch.exp(-0.5 * logsnr_min)) - b).to(time.device)
        return -2.0 * torch.log(torch.tan(a * time.to(torch.float32) + b))

    def get_logsnr_alpha_sigma(self, time: Tensor, shape=None):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = torch.sqrt(torch.sigmoid(logsnr))
        sigma = torch.sqrt(torch.sigmoid(-logsnr))

        if shape is not None:
            logsnr = logsnr.view(shape).to(torch.float32)
            alpha = alpha.view(shape).to(torch.float32)
            sigma = sigma.view(shape).to(torch.float32)

        return logsnr, alpha, sigma

    def add_perturbation(self, sources: Tuple[Source, ...], source_time: Tensor, diffusion_type: InputType, normalizers: nn.ModuleList) -> Tuple[Tuple[Source, ...], Tuple[Source, ...]]:
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(source_time.unsqueeze(-1))
        output_sources = []
        output_vectors = []
        for input_index, name in enumerate(self.input_types):
            source_before_norm, mask = sources[input_index]
            source = normalizers[input_index](source_before_norm, mask)

            if (self.input_types[name] == diffusion_type):
                eps = torch.randn(source.size(), device=source.device, dtype=torch.float32)
                perturbed_x = Source(data=source * alpha + eps * sigma, mask=mask)
                score = Source(data=eps * alpha - source * sigma, mask=mask)
            else:
                perturbed_x = Source(data=source, mask=mask)
                score = Source(data=torch.zeros_like(source), mask=mask)

            output_sources.append(perturbed_x)
            output_vectors.append(score)

        return tuple(output_sources), tuple(output_vectors)

    def add_perturbation_dict(self, source: Dict[str, Tensor], source_time: Tensor, normalizers: nn.ModuleList):
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(source_time.unsqueeze(-1))
        perturbed_source = OrderedDict()
        score = OrderedDict()
        for input_index, name in enumerate(source):
            source_ = source[name]
            source_reshaped = source_.view(source_.shape[0], 1, 1).contiguous()
            source_reshaped = normalizers[input_index](source_reshaped, torch.ones((source_.shape[0], 1), device=source_.device))
            eps = torch.randn(source_reshaped.size(), device=source_reshaped.device, dtype=torch.float32)
            perturbed_source[name] = (alpha * source_reshaped + eps * sigma).view(source_.size())
            score[name] = (eps * alpha - source_reshaped * sigma).view(source_.size())
        return perturbed_source, score

    def prepare_format(self, sources: List[Tuple[Tensor, Tensor]], source_time: Tensor, source_num_vectors: Dict[str, Tensor], source_score: Optional[List[Tuple[Tensor, Tensor]]] = None, source_num_vectors_score: Optional[Dict[str,Tensor]] = None):
        global_x = []
        global_mask = None
        global_score = []

        sequential_x = []
        sequential_mask = None
        sequential_score = []

        for name, source in source_num_vectors.items():
            source = source.view(source.shape[0], 1, 1).contiguous()
            global_x.append(source)
            global_mask = global_mask.logical_and(torch.ones((source.shape[0], 1), device=source.device).bool()) if global_mask is not None else torch.ones((source.shape[0], 1), device=source.device).bool()
            global_score.append(source_num_vectors_score[name].view(source.shape) if source_num_vectors_score is not None else torch.zeros_like(source))

        for input_index, name in enumerate(self.input_types):
            source, mask = sources[input_index]
            if (self.input_types[name] == InputType.Global):
                global_x.append(source)
                global_mask = global_mask.logical_and(mask) if global_mask is not None else mask
                global_score.append(source_score[input_index].data if source_score is not None else torch.zeros_like(source))
            else:
                sequential_x.append(source)
                sequential_mask = sequential_mask.logical_and(mask) if sequential_mask is not None else mask
                sequential_score.append(source_score[input_index].data if source_score is not None else torch.zeros_like(source))

        global_x = torch.cat(global_x, dim=-1)
        sequential_x = torch.cat(sequential_x, dim=-1)

        global_score = torch.cat(global_score, dim=-1)
        sequential_score = torch.cat(sequential_score, dim=-1)

        perturbed_x = dict()
        perturbed_x["Global"] = Source(global_x, global_mask)
        perturbed_x["Sequential"] = Source(sequential_x, sequential_mask)

        perturbed_score = dict()
        perturbed_score["Global"] = Source(global_score, global_mask)
        perturbed_score["Sequential"] = Source(sequential_score, sequential_mask)
        return perturbed_x, perturbed_score

    def decode_tensor(self, input_x: Source, key: str):
        output = DistributionInfo()
        source, mask = input_x
        for name, mapping in self.output_dim_mapping[key].items():
            output[name] = Source(source[..., mapping].contiguous(), mask)
        return output

    def second_order_correction(self, time_step, x, pred_images, pred_noises,
                              alphas, sigmas, w, model, num_steps=100,
                              second_order_alpha=0.5, shape=None):
        with torch.no_grad():
            step_size = 1.0 / num_steps
            t = time_step - second_order_alpha * step_size
            logsnr, alpha_signal_rates, alpha_noise_rates = self.get_logsnr_alpha_sigma(t, shape=shape)
            alpha_noisy_images = alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises
            v, _ = model((alpha_noisy_images, None), t)

            alpha_pred_noises = alpha_noise_rates * alpha_noisy_images + alpha_signal_rates * v
            pred_noises = (1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises

            mean = (x - sigmas * pred_noises) / alphas
            eps = pred_noises

        return mean, eps

    def DDPMSampler(self, model, data_shape, jet=None, w=0.1, num_steps=100, mask=None, device='cpu'):
        with torch.no_grad():
            batch_size = data_shape[0]
            const_shape = (batch_size, 1, 1)
            x = self.prior_sde(data_shape, device=device)

            for time_step in range(num_steps, 0, -1):
                batch_time = torch.ones((batch_size, 1), dtype=torch.int32, device=device) * time_step / num_steps
                batch_next_time = torch.ones((batch_size, 1), dtype=torch.int32, device=device) * (time_step - 1) / num_steps
                logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(batch_time, shape=const_shape)
                logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(batch_next_time, shape=const_shape)

                v, _ = model((x, None), batch_time)

                mean = alpha * x - sigma * v
                eps = v * alpha + x * sigma
                mean, eps = self.second_order_correction(batch_time, x, mean, eps,
                                                       alpha, sigma, w, model,
                                                       num_steps=num_steps,
                                                       shape=const_shape)
                x = alpha_ * mean + sigma_ * eps

        return mean

    def prior_sde(self, shape, device='cpu'):
        return torch.randn(shape, dtype=torch.float32, device=device)

    def create_mask_from_nJet(self, jets_count: Tensor, nMaxjet: int) -> Tensor:
        # jets_count: [B, 1, 1]
        batch_size = jets_count.shape[0]
        jets_count = jets_count.view(batch_size, 1)

        range_tensor = torch.arange(nMaxjet).expand(batch_size, nMaxjet).to(jets_count.device)  # [B, T]
        jets_count_expanded = jets_count.expand_as(range_tensor)  # [B, T]

        mask_matrix = range_tensor < jets_count_expanded

        return mask_matrix

    def generate(self, model, batch_size, normlizers, normalizers_num_vector, num_steps=100, device='cpu'):
        model = model.to(device)

        # Diffuse Global Result
        generated_global_features = Source(
            data=self.DDPMSampler(model.event_generation_decoder,
                                 data_shape=(batch_size, 1, self.num_global),
                                 device=device, num_steps=num_steps),
            mask=torch.ones((batch_size, 1), device=device).bool()
        )
        Generated_Feature_Global = self.decode_tensor(generated_global_features, "Global")

        # Prepare prior distribution for sequential result
        source_num_vector = OrderedDict()
        sources = []

        num_vector_index = 0
        for name, input_feature in self.input_features.items():
            if (self.input_types[name] == InputType.Global):
                data = Generated_Feature_Global[name]
                mask = torch.ones((batch_size, 1), device=device)
                sources.append(Source(data, mask))
            else:
                num_vector = normalizers_num_vector[num_vector_index](Generated_Feature_Global[name].data,
                                                                    Generated_Feature_Global[name].mask)
                num_vector = torch.floor(num_vector + 0.5)
                num_vector = num_vector.view(batch_size, 1)
                source_num_vector[name] = num_vector

                data = self.prior_sde((batch_size, self.nMaxJet, len(input_feature)), device=device)
                mask = self.create_mask_from_nJet(source_num_vector[name], self.nMaxJet)
                data = data * mask.unsqueeze(-1).float()
                sources.append(Source(data, mask))
                num_vector_index += 1

        sources = tuple(sources)

        # Diffuse Sequential Result
        generated_sequential_features = self.DDPMSampler_sequential(model, sources, source_num_vector,
                                                                  batch_size, num_steps=num_steps,
                                                                  device=device)

        Generated_Feature_Sequential = DistributionInfo()

        for input_index, name in enumerate(self.input_types):
            if (self.input_types[name] == InputType.Global):
                continue
            _, mask = sources[input_index]
            data, _ = generated_sequential_features[input_index]
            data = data * mask.unsqueeze(-1).float()
            Generated_Feature_Sequential[name] = Source(data, mask)

        return self.denormalizer(Generated_Feature_Global, Generated_Feature_Sequential,
                               normlizers, normalizers_num_vector)

    def denormalizer(self, Generated_Feature_Global, Generated_Feature_Sequential,
                    normlizers, normalizers_num_vector):
        Generated_Feature_before_norm = OrderedDict()

        num_vector_index = 0
        for name, input_type in self.input_types.items():
            if (input_type == InputType.Sequential):
                source, mask = Generated_Feature_Global[name]
                source_before_norm = normalizers_num_vector[num_vector_index].denormalize(source, mask)
                source_int = torch.floor(source_before_norm + 0.5)
                Generated_Feature_before_norm["num_" + name] = Source(source_int, mask)
                num_vector_index += 1

        for input_index, name in enumerate(self.input_features):
            input_feature = self.input_features[name]
            if (self.input_types[name] == InputType.Global):
                source, mask = Generated_Feature_Global[name]
                source_before_norm = normlizers[input_index].denormalize(source, mask)
            else:
                source, mask = Generated_Feature_Sequential[name]
                source_before_norm = normlizers[input_index].denormalize(source, mask)

            for dim_ in range(len(input_feature)):
                if not (self.input_features[name][dim_].normalize):  # TODO: normally it means that the input is integer
                    source_before_norm[..., dim_] = torch.floor(source_before_norm[..., dim_] + 0.5)
            Generated_Feature_before_norm[name] = Source(source_before_norm, mask)

        return Generated_Feature_before_norm

    def second_order_correction_sequential(self, time_step, x, source_num_vector,
                                         pred_images, pred_noises, alphas, sigmas, w,
                                         model, num_steps=100, second_order_alpha=0.5,
                                         shape=None):
        with torch.no_grad():
            step_size = 1.0 / num_steps
            t = time_step - second_order_alpha * step_size
            logsnr, alpha_signal_rates, alpha_noise_rates = self.get_logsnr_alpha_sigma(t, shape=shape)
            alpha_noisy_images = alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises
            v, _ = model.predict_sequential_vector(alpha_noisy_images, t, source_num_vector)
            v = SourceTuple(v)

            alpha_pred_noises = alpha_noise_rates * alpha_noisy_images + alpha_signal_rates * v
            pred_noises = (1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises

            mean = (x - sigmas * pred_noises) / alphas
            eps = pred_noises

        return mean, eps

    def DDPMSampler_sequential(self, model, sources, source_num_vector, batch_size,
                              w=0.1, num_steps=100, mask=None, device='cpu'):
        x = SourceTuple(sources)
        with torch.no_grad():
            const_shape = (batch_size, 1, 1)
            for time_step in tqdm(range(num_steps, 0, -1), desc="Generating Sequential Result", mininterval=30):  # Update every 30 seconds
                batch_time = torch.ones((batch_size, 1), dtype=torch.int32, device=device) * time_step / num_steps
                batch_next_time = torch.ones((batch_size, 1), dtype=torch.int32, device=device) * (time_step - 1) / num_steps

                logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(batch_time, shape=const_shape)
                logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(batch_next_time, shape=const_shape)

                v, _ = model.predict_sequential_vector(x, batch_time, source_num_vector)  # SourceTuple
                v = SourceTuple(v)

                mean = alpha * x - sigma * v
                eps = v * alpha + x * sigma
                mean, eps = self.second_order_correction_sequential(batch_time,
                                                                  x, source_num_vector, mean, eps,
                                                                  alpha, sigma, w,
                                                                  model, num_steps=num_steps,
                                                                  shape=const_shape)
                x = alpha_ * mean + sigma_ * eps
                output = []
                for x_ in x:
                    data, mask = x_
                    output.append(Source(data * mask.unsqueeze(-1).float(), mask))
                x = SourceTuple(tuple(output))

        return mean