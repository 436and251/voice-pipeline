from dataclasses import dataclass


@dataclass(frozen=True)
class S2BatchResult:
    optimizer_steps: int
    discriminator_total: object
    generator_total: object


def run_v2proplus_batch(
    *, discriminator_loss, generator_loss, feature_loss, mel_loss, kl_ssl, kl_loss,
    optim_d, optim_g, scaler, clip_fn, combine_generator,
):
    d_total = discriminator_loss[0] if isinstance(discriminator_loss, tuple) else discriminator_loss
    g_adv = generator_loss[0] if isinstance(generator_loss, tuple) else generator_loss

    optim_d.zero_grad()
    scaler.scale(d_total).backward()
    scaler.unscale_(optim_d)
    clip_fn("d")
    scaler.step(optim_d)

    g_total = combine_generator(g_adv, feature_loss, mel_loss, kl_ssl, kl_loss)
    optim_g.zero_grad()
    scaler.scale(g_total).backward()
    scaler.unscale_(optim_g)
    clip_fn("g")
    scaler.step(optim_g)
    scaler.update()
    return S2BatchResult(1, d_total, g_total)
