import torch
from torchvision import datasets, transforms
from models.generator import Generator
from models.discriminator import Discriminator
from torchvision.utils import save_image
import wandb
from tqdm.auto import tqdm
import os
import argparse

#TODO:
# - logging: loss, hyper params(done)
# - save checkpoint (done)
# - reorganize code (done)
# - resume training (done)
# - change to train on cifar 10
# - calculate FID
# - change architecture 
# - test distrubuted traning: vast AI

parser = argparse.ArgumentParser(description="Train a GAN on MNIST")
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for Adam optimizer')
parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch si0001ze for dataloader')
parser.add_argument('--step_log', type=int, default=5000, help="Control how many step each log")
parser.add_argument('--generator_train_times', type=int, default=2, help='Number of times to train generator per discriminator step')
parser.add_argument('--checkpoint_path', type=str, help="Checkpoint path for resume traning")
parser.add_argument('--_wandb_id', type=str, help="WandB ID runs for resume traning")
args = parser.parse_args()

#wandb logging
config = {
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "loss_function": "BCELoss",
    "optimizer": "Adam",
    "adam_betas": (0.5, 0.999),
    "generator_train_times" : args.generator_train_times
}

if args._wandb_id:
    wandb.init(project="gan-mnist", id=args._wandb_id, resume="must")
else:
    wandb.init(project="gan-mnist", config=config)

def main():
    #init generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    generator.to("cuda")
    discriminator.to("cuda")

    #NOTE: loss function is just binary classification loss
    #init loss function
    loss_function = torch.nn.BCELoss()

    #NOTE: optimizer in paper is Minibatch stochastic gradient descent
    #init optimizer for generator and discriminator
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=wandb.config.adam_betas)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=wandb.config.adam_betas)
    # optimizer_generator = torch.optim.SGD(generator.parameters(), lr=0.0002)
    # optimizer_discriminator = torch.optim.SGD(discriminator.parameters(), lr=0.0002)

    #init dataloder for mnist dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root="../../data/mnist",
                                train=True,
                                download=True,
                                transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size, shuffle=True)

    #TODO: logging to check quality output
    wandb.save("models/*.py")
    wandb.watch((generator, discriminator), log="all", log_freq=200, log_graph=True)
    #create image log dir
    image_log_dir = os.path.join(wandb.run.dir, "image_log")
    os.makedirs(image_log_dir, exist_ok=True)
    #create model log dir
    checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    #resume tranning
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cuda")

        #load models weights
        generator.load_state_dict(checkpoint["G_state_dict"])
        discriminator.load_state_dict(checkpoint["D_state_dict"])

        #load optimizer states
        optimizer_generator.load_state_dict(checkpoint["optimizerG_state_dict"])
        optimizer_discriminator.load_state_dict(checkpoint["optimizerD_state_dict"])

        #load epoch
        start_epoch = checkpoint["epoch"]

        #load step
        sample_done = checkpoint["step"]

        #set generator and discriminator to traning mode
        generator.train()
        discriminator.train()
    else:
        start_epoch = 0
        sample_done = 0

    #train
    for epoch in tqdm(range(start_epoch, args.epochs)):
        for i, (images, labels) in enumerate(dataloader):
            # breakpoint()
            #move images to cuda
            images = images.to("cuda")
            #setup label
            real_label = torch.full((images.shape[0], 1), 1.0, device="cuda")
            fake_label = torch.zeros(images.shape[0], 1, device="cuda")

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_discriminator.zero_grad()

            # init sample and pass into generator
            z = torch.randn(images.shape[0], 100, device="cuda")
            gen_images = generator(z)
            # breakpoint()

            #setup loss for discriminator
            real_loss = loss_function(discriminator(images), real_label)
            fake_loss = loss_function(discriminator(gen_images.detach()), fake_label)
            loss_d = (real_loss + fake_loss) / 2
            
            #optimize discriminator
            loss_d.backward()
            optimizer_discriminator.step()

            # -----------------
            #  Train Generator
            # -----------------
            for _ in range(args.generator_train_times):
                optimizer_generator.zero_grad()

                z = torch.randn(images.shape[0], 100, device="cuda")
                gen_images = generator(z)
                
                #generator loss and optimize
                loss_g = loss_function(discriminator(gen_images), real_label)
                loss_g.backward()
                optimizer_generator.step()

            #logging
            wandb.log({
                "epoch": epoch,
                "loss_g": loss_g.item(),
                "loss_d": loss_d.item(),    
            }, step=sample_done)
            sample_done += 1

            if (sample_done) % 5000 == 0:
                save_image(gen_images.data[:50], os.path.join(image_log_dir, f"{sample_done}.png"), nrow=10, normalize=True)
                checkpoints = {
                    "epoch" : epoch,
                    "step": sample_done,
                    "G_state_dict": generator.state_dict(),
                    "D_state_dict": discriminator.state_dict(),
                    "optimizerG_state_dict": optimizer_generator.state_dict(),
                    "optimizerD_state_dict": optimizer_discriminator.state_dict()
                }
                torch.save(checkpoints, os.path.join(checkpoint_dir,f"checkpoints_{sample_done}.pth"))

if __name__ == "__main__":
    main()

