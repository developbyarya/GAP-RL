import inspect

def dummy(
    observation_space, action_space, lr_schedule,
    orig_observation_space, stack_keys, net_arch=None, activation_fn=None,
    use_sde=False, log_std_init=-3, use_expln=False, clip_mean=2.0,
    features_extractor_class=None, features_extractor_kwargs=None,
    normalize_images=True, optimizer_class=None, optimizer_kwargs=None,
    n_critics=2, share_features_extractor=False, extra_pred_dim=7
):
    print("log_std_init =", log_std_init)

kwargs = dict(
    log_std_init=-1.5,
    net_arch=[256, 256],
    features_extractor_class="CombinedExtractor",
    features_extractor_kwargs=None,
    normalize_images=False,
    share_features_extractor=False,
    extra_pred_dim=9,
    orig_observation_space="orig",
    stack_keys=["a", "b"]
)

dummy("obs", "act", "lr", **kwargs)
