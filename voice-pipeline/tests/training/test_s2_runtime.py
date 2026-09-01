from voice_pipeline.training.s2_runtime import run_v2proplus_batch


class FakeScaler:
    def __init__(self, events): self.events = events
    def scale(self, loss): self.events.append(("scale", loss)); return self
    def backward(self): self.events.append("backward")
    def unscale_(self, optim): self.events.append(("unscale", optim.name))
    def step(self, optim): self.events.append(("step", optim.name))
    def update(self): self.events.append("update")


class FakeOptim:
    def __init__(self, name, events): self.name, self.events = name, events
    def zero_grad(self): self.events.append(("zero", self.name))


def test_v2proplus_batch_updates_d_before_g_and_counts_one_step():
    events = []
    scaler = FakeScaler(events)
    d = FakeOptim("d", events)
    g = FakeOptim("g", events)
    clips = []
    result = run_v2proplus_batch(
        discriminator_loss=("d_total", [1], [2]),
        generator_loss=("g_adv", [3]),
        feature_loss="fm",
        mel_loss="mel",
        kl_ssl="kl_ssl",
        kl_loss="kl",
        optim_d=d,
        optim_g=g,
        scaler=scaler,
        clip_fn=lambda which: clips.append(which),
        combine_generator=lambda adv, fm, mel, kl_ssl, kl: "g_total",
    )
    assert result.optimizer_steps == 1
    assert result.discriminator_total == "d_total"
    assert result.generator_total == "g_total"
    assert clips == ["d", "g"]
    assert events == [
        ("zero", "d"), ("scale", "d_total"), "backward", ("unscale", "d"), ("step", "d"),
        ("zero", "g"), ("scale", "g_total"), "backward", ("unscale", "g"), ("step", "g"), "update",
    ]
