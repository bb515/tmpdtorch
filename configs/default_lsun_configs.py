import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()

  # data
  config.data = data = ml_collections.ConfigDict()

  return config