import torch

class Discriminator(torch.nn.Module):
    def __init__(self, hidden_size_1=512, hidden_size_2=256, hidden_size_3=128, 
                 dropout_rate=0.3, negative_slope=0.1, label_emb_size=100):
        super(Discriminator, self).__init__()
        #embedding labels
        self.embedding_labels = torch.nn.Embedding(10, label_emb_size)
        # mlp
        self.mlp_0 = torch.nn.Linear(3072+label_emb_size, hidden_size_1)
        self.mlp_1 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.mlp_2 = torch.nn.Linear(hidden_size_2, hidden_size_3)
        self.mlp_3 = torch.nn.Linear(hidden_size_3, 1)
        # activation function and dropout
        self.sigmoid = torch.nn.Sigmoid()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self, image, labels):
        labels_emb = self.embedding_labels(labels)
        flat_image = image.view(image.shape[0], -1)
        input = torch.cat((flat_image, labels_emb), dim=1)
        #layer 0
        x = self.mlp_0(input)
        x = self.leaky_relu(x)
        #layer 1
        x = self.mlp_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        #layer 2
        x = self.mlp_2(x)
        x = self.leaky_relu(x)
        #layer 3
        x = self.mlp_3(x)
        x = self.sigmoid(x)

        return x