import torch

from typing import Optional
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

from tqdm.auto import tqdm

from vae import VAE
from utils import ELBO_Loss, Model_Args

class VAE_Trainer():
  def __init__(
      self,
      vae: VAE,
      loss_function: Optional[ELBO_Loss] = None,
      train_loader: Optional[torch.utils.data.DataLoader] = None,
      optimizer: Optional[torch.optim.Optimizer] = None,
      create_optimizer: tuple[str, float] = ("AdamW", 1e-4),
      gradient_clipping: bool = True,
      LAMBDA=100,
      RESCALE=1e-2,
      trained_epochs: int = 0,
  ):
    self.__device = "cuda" if torch.cuda.is_available() else "cpu"
    self.vae = vae

    if loss_function is None:
      self.loss_function = ELBO_Loss()
    else:
      self.loss_function = loss_function

    self.train_loader = train_loader

    # Instantiating optimizer
    if optimizer is not None:
      self.optimizer = optimizer
      self.lr = optimizer.param_groups[0]['lr']
    else:
      self.optimizer = VAE_Trainer.instantiate_optimizer(self.vae, create_optimizer)
      self.lr = create_optimizer[1]

    self.gradient_clipping = gradient_clipping
    self.LAMBDA = LAMBDA
    self.RESCALE = RESCALE
    self.trained_epochs = trained_epochs

  @staticmethod
  def __restore_trainer(model_args: Model_Args):

    # Loading model from filesystem
    data = torch.load(model_args.project_dir + "checkpoints/" + model_args.checkpoint_filename, weights_only=False)

    vae = VAE.from_pretrained(data=data["vae"])

    # recovering optimizer
    optimizer_name = data["optimizer_name"]
    lr = data["LR"]
    optimizer = VAE_Trainer.instantiate_optimizer(vae, (optimizer_name, lr))
    optimizer.load_state_dict(data["optimizer_state"])

    gradient_clipping = data["gradient_clipping"]
    LAMBDA = data["LAMBDA"]
    RESCALE = data["RESCALE"]
    trained_epochs = data["trained_epochs"]

    return VAE_Trainer(
      vae=vae,
      optimizer=optimizer,
      gradient_clipping=gradient_clipping,
      LAMBDA=LAMBDA,
      RESCALE=RESCALE,
      trained_epochs=trained_epochs
    )

  @staticmethod
  def instantiate_optimizer(vae: VAE, optimizer_tuple: tuple[str, float]):
    optimizer_name = optimizer_tuple[0]
    lr = optimizer_tuple[1]

    OPTIMIZERS_MAP = {
      "AdamW": torch.optim.AdamW(params=vae.parameters(), lr=lr),
      "Adam": torch.optim.Adam(params=vae.parameters(), lr=lr),
      "SGD": torch.optim.SGD(params=vae.parameters(), lr=lr),
    }
    if optimizer_name not in OPTIMIZERS_MAP:
      supported = ", ".join(OPTIMIZERS_MAP.keys())
      raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. Supported: {supported}"
      )
    return OPTIMIZERS_MAP[optimizer_name]

  @staticmethod
  def from_checkpoint(model_args: Model_Args, train_loader: Optional[torch.utils.data.DataLoader] = None,
                      loss_function: Optional[ELBO_Loss] = None):
    if (train_loader is None and loss_function is not None) or (train_loader is not None and loss_function is None):
      raise ValueError("If train loader is passed, also loss function must be passed. And vice-versa!")

    trainer = VAE_Trainer.__restore_trainer(model_args)
    if train_loader is not None and loss_function is not None:
      trainer.set_train_loader(train_loader)
      trainer.set_loss_function(loss_function)

    return trainer

  def set_train_loader(self, train_loader: Optional[torch.utils.data.DataLoader] = None):
    if train_loader is None:
      raise ValueError("Error! Trying to set a None train loader.")

    self.train_loader = train_loader

  def set_loss_function(self, loss_function: Optional[ELBO_Loss] = None):
    if loss_function is None:
      raise ValueError("Error! Trying to set a None loss function.")

    self.loss_function = loss_function

  def create_checkpoint(self, model_args: Model_Args):
    data = {
      "vae": self.vae.save_model(model_args),
      "loss_function": self.loss_function,
      "train_loader": self.train_loader,
      "optimizer_state": self.optimizer.state_dict(),
      "optimizer_name": self.optimizer.__class__.__name__,
      "LR": self.lr,
      "gradient_clipping": self.gradient_clipping,
      "LAMBDA": self.LAMBDA,
      "RESCALE": self.RESCALE,
      "trained_epochs": self.trained_epochs
    }

    # Saving model on filesystem
    torch.save(data, model_args.project_dir + "checkpoints/" + model_args.checkpoint_filename)

  def explain_checkpoint(self):
    print("The current model has been trained for " + str(self.trained_epochs) + " epochs on " + str(
      self.optimizer.__class__.__name__) + " optimizer, using the following hyperparameters:")
    print("LR: " + str(self.lr) + "\nLAMBDA: " + str(self.LAMBDA) + "\nRESCALE: " + str(
      self.RESCALE) + "\nGRADIENT_CLIPPING: " + str(self.gradient_clipping))

  def train(self, model_args: Model_Args, EPOCHS: int = 200, epochs_to_create_checkpoint: int = 0, optimize: bool = False):
    torch.backends.cudnn.benchmark = True

    if self.train_loader is None:
      raise ValueError("Error! Trying to train without specifying the train loader.")

    if self.loss_function is None:
      raise ValueError("Error! Trying to train without specifying the loss function.")

    self.vae.to(self.__device)

    # Checking if optimization is enabled
    scaler = None
    if optimize:
      tmp_img, _ = next(iter(self.train_loader))
      tmp_img = tmp_img.to(self.__device)
      
      tmp_latents = self.vae.encode(tmp_img).to(self.__device)

      # Tracing
      self.vae.train()
      self.vae.encoder = torch.jit.trace(self.vae.encoder, tmp_img)
      self.vae.decoder = torch.jit.trace(self.vae.decoder, tmp_latents)

      scaler = GradScaler(enabled=True)

    metrics = {
      "avg_epoch_loss": "0",
      "batch_kl_divergence": "0",
      "batch_reconstruction_error": "0"
    }

    tot_loss = 0
    num_batch = 0
    initial = self.trained_epochs
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[bold cyan]AVG_LOSS: {task.fields[avg_loss]}"),
        TextColumn("[bold magenta]KL: {task.fields[kl]}"),
        TextColumn("[bold yellow]REC: {task.fields[rec]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True
    ) as progress:
      epochs_task = progress.add_task("[green]Total Progress", total=EPOCHS, avg_loss="---", kl="---", rec="---")
      batch_task = progress.add_task("[cyan]Current epoch", total=len(self.train_loader), avg_loss="---", kl="---", rec="---")

      for epoch in range(EPOCHS):
        tot_loss = 0
        num_batch = 0
        curr_epoch = epoch + initial + 1

        # Second bar reset for current epoch
        progress.reset(batch_task, description=f"[cyan]Epoch {curr_epoch} Batches")

        # Epoch training loop
        for x, label in self.train_loader:
          x = x.to(self.__device)

          self.optimizer.zero_grad()

          # Executing optimized version or normal depending on self.optimize
          if optimize:
            with autocast(self.__device, enabled=True, dtype=torch.float16):
              lam, y = self.vae(x)

              kl_div, rec_error = self.loss_function.compute_loss(x, y, lam, self.LAMBDA, self.RESCALE)

            with autocast(self.__device, enabled=False):
              loss = kl_div + rec_error

            tot_loss += loss.item()

            scaler.scale(loss).backward()

            if self.gradient_clipping:
              scaler.unscale_(self.optimizer)
              torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)

            scaler.step(self.optimizer)
            scaler.update()
          else:
            lam, y = self.vae(x)

            kl_div, rec_error = self.loss_function.compute_loss(x, y, lam, self.LAMBDA, self.RESCALE)
            loss = kl_div + rec_error

            tot_loss += loss.item()
            loss.backward()

            if self.gradient_clipping:
              torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)

            self.optimizer.step()

          if (num_batch + 1) % 1 == 0:
            metrics["batch_kl_divergence"] = f"{kl_div.item():.4f}"
            metrics["batch_reconstruction_error"] = f"{rec_error.item():.4f}"

            progress.update(epochs_task, advanced=0, kl=metrics["batch_kl_divergence"], rec=metrics["batch_reconstruction_error"])
            progress.update(batch_task, advance=1)

          num_batch += 1

        metrics["avg_epoch_loss"] = f"{tot_loss / num_batch:.4f}"
        progress.update(epochs_task, advance=1, avg_loss=metrics["avg_epoch_loss"])

        if epochs_to_create_checkpoint!=0 and ((epoch + 1) % epochs_to_create_checkpoint)==0:
          self.trained_epochs += epochs_to_create_checkpoint
          self.create_checkpoint(model_args)
