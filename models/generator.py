import torch

class Generator(torch.nn.Module):
    def __init__(self, laten_dim=100, hidden_size_1=256, hidden_size_2=512, 
                hidden_size_3=1024, negative_slope=0.1, label_emb_size=100):
        super(Generator, self).__init__()
        #embedding labels
        self.embedding_labels = torch.nn.Embedding(10, label_emb_size)
        # activation function
        self.tanh = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        # mlp
        self.mlp_0 = torch.nn.Linear(laten_dim+label_emb_size, hidden_size_1)
        self.mlp_1 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.mlp_2 = torch.nn.Linear(hidden_size_2, hidden_size_3)
        self.mlp_3 = torch.nn.Linear(hidden_size_3, 3072) #32*32*3

    def forward(self, z, labels):
        labels_emb = self.embedding_labels(labels)
        input = torch.cat((z, labels_emb), dim=1)
        #layer 1
        z = self.mlp_0(input)
        z = self.leaky_relu(z)
        #layer 2
        z = self.mlp_1(z)
        z = self.leaky_relu(z)
        #layer 3
        z = self.mlp_2(z)
        z = self.leaky_relu(z)
        #layer 4
        z = self.mlp_3(z)
        z = self.tanh(z)
        #FIXME: This return image, need to transform to image not just sigmoid (done)
        image = z.view(z.shape[0], 3, 32, 32)
        
        return image