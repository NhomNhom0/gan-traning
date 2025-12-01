import torch
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
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
# - change to train on cifar 10 (done)
# - calculate FID (done) change to 10k image to calculate
# - change architecture (done), maybe setup convolution to make generator better
# - class condition(done)
# - test distrubuted traning: vast AI
# - mix precision: learning rate scheduler 

parser = argparse.ArgumentParser(description="Train a GAN on MNIST")
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for Adam optimizer')
parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch si0001ze for dataloader')
parser.add_argument('--laten_dim', type=int, default=600, help='Image laten_dim')
parser.add_argument('--label_emb_size', type=int, default=100, help='Labels embedding size')
parser.add_argument('--generator_train_times', type=int, default=2, help='Number of times to train generator per discriminator step')
parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path for resume traning')
parser.add_argument('--wandb_path', type=str, help='WandB ID runs for resume traning')
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

if args.wandb_path:
    wandb.init(project="gan-cifar10", id=args.wandb_path, resume="must")
else:
    wandb.init(project="gan-cifar10", config=config)

def main():
    #init generator and discriminator
    generator = Generator(laten_dim=args.laten_dim, hidden_size_1=1024, hidden_size_2=2048, hidden_size_3=4096, label_emb_size=args.label_emb_size)
    discriminator = Discriminator(hidden_size_1=512, hidden_size_2=256, hidden_size_3=128, label_emb_size=args.label_emb_size)
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

    #init dataloder for cifar dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    cifar = datasets.CIFAR10(root="../../data/cifar10",
                                train=True,
                                download=True,
                                transform=transform)
    dataloader = torch.utils.data.DataLoader(cifar, batch_size=args.batch_size, shuffle=True)

    #TODO: logging to check quality output
    wandb.save("models/*.py")
    wandb.watch((generator, discriminator), log="all", log_freq=200, log_graph=True)
    #create image log dir
    image_log_dir = os.path.join(wandb.run.dir, "image_log")
    os.makedirs(image_log_dir, exist_ok=True)
    #create model log dir
    checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    #init FID
    fid = FrechetInceptionDistance(feature=2048).to('cuda')

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
            #move images, labels to cuda
            labels = labels.to("cuda")
            images = images.to("cuda")
            #setup label
            real_label = torch.full((images.shape[0], 1), 1.0, device="cuda")
            fake_label = torch.zeros(images.shape[0], 1, device="cuda")

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_discriminator.zero_grad()

            # init sample and pass into generator
            z = torch.randn(images.shape[0], args.laten_dim, device="cuda")
            gen_images = generator(z, labels) #TODO: add labels to generator(done)
            # breakpoint()

            #setup loss for discriminator
            #TODO: add labels to discriminator(done)
            real_loss = loss_function(discriminator(images, labels), real_label)
            fake_loss = loss_function(discriminator(gen_images.detach(), labels), fake_label)
            loss_d = (real_loss + fake_loss) / 2
            
            #optimize discriminator
            loss_d.backward()
            optimizer_discriminator.step()

            # -----------------
            #  Train Generator
            # -----------------
            for _ in range(args.generator_train_times):
                optimizer_generator.zero_grad()

                # samples image and labels
                z = torch.randn(images.shape[0], args.laten_dim, device="cuda")
                sample_labels = torch.randint(0, 10, (images.shape[0],), device="cuda")
                gen_images = generator(z, sample_labels)
                
                #generator loss and optimize
                loss_g = loss_function(discriminator(gen_images, sample_labels), real_label)
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
                generator.eval()
                with torch.no_grad():
                    z = torch.randn(images.shape[0], args.laten_dim, device="cuda")
                    sample_labels = torch.randint(0, 10, (images.shape[0],), device="cuda")
                    gen_images_for_fid = generator(z, sample_labels)

                #calculate FID
                fid.update(((images+1) / 2 * 255).to(torch.uint8), real=True)
                fid.update(((gen_images_for_fid+1) / 2 * 255).to(torch.uint8), real=False)
                fid_score = fid.compute()
                wandb.log({"fid_score": fid_score.item()}, step=sample_done)
                fid.reset()

                save_image(gen_images_for_fid.data[:50], os.path.join(image_log_dir, f"{sample_done}.png"), nrow=10, normalize=True)
                checkpoints = {
                    "epoch" : epoch,
                    "step": sample_done,
                    "G_state_dict": generator.state_dict(),
                    "D_state_dict": discriminator.state_dict(),
                    "optimizerG_state_dict": optimizer_generator.state_dict(),
                    "optimizerD_state_dict": optimizer_discriminator.state_dict()
                }
                torch.save(checkpoints, os.path.join(checkpoint_dir,f"checkpoints_{sample_done}.pth"))
                generator.train()

if __name__ == "__main__":
    main()

