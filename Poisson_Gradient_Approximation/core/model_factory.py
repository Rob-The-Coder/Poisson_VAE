from vae.encoders import Encoder_36M, Encoder_53M, Encoder_60M, Encoder_GRT_53M
from vae.decoders import Decoder_36M, Decoder_53M, Decoder_60M, Decoder_GRT_53M

class ModelFactory:
  _REGISTRY = {
    ("PGA", "36M"): (Encoder_36M, Decoder_36M),
    ("PGA", "53M"): (Encoder_53M, Decoder_53M),
    ("PGA", "60M"): (Encoder_60M, Decoder_60M),
    ("GRT", "53M"): (Encoder_GRT_53M, Decoder_GRT_53M),
  }

  @classmethod
  def create(cls, sampling, model_type, height, width, latent_dim):
    if (sampling, model_type) not in cls._REGISTRY:
      supported = ", ".join(*cls._REGISTRY)
      raise ValueError(
        f"Unsupported model configuration {sampling}-{model_type}. Supported: {supported}"
      )
    encoder, decoder = cls._REGISTRY[(sampling, model_type)]
    return encoder(height, width, latent_dim), decoder(height, width, latent_dim)
