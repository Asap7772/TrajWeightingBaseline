import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (128, 128)

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 3.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.
    
    # Additional args
    config.exp_a_clip = 100

    return config
