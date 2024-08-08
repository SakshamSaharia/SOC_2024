from torch import nn  # Import the nn module from PyTorch, which provides classes for building neural networks

# Define a class ConvLayer which inherits from nn.Module
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvLayer, self).__init__()  # Initialize the parent class nn.Module
        # Create a sequential container with ReLU activation and a Conv2d layer
        self.module = nn.Sequential(
            nn.ReLU(inplace=True),  # Apply ReLU activation function in-place to save memory
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
            # Apply 2D convolution with specified kernel size, input/output channels, and padding to maintain spatial dimensions
        )

    # Define the forward pass of the ConvLayer
    def forward(self, x):
        return self.module(x)  # Pass the input x through the sequential container and return the result


# Define a class ResidualUnit which inherits from nn.Module
class ResidualUnit(nn.Module):
    def __init__(self, num_features):
        super(ResidualUnit, self).__init__()  # Initialize the parent class nn.Module
        # Create a sequential container with two ConvLayer instances
        self.module = nn.Sequential(
            ConvLayer(num_features, num_features),  # First ConvLayer with num_features input/output channels
            ConvLayer(num_features, num_features)   # Second ConvLayer with the same number of input/output channels
        )

    # Define the forward pass of the ResidualUnit
    def forward(self, h0, x):
        return h0 + self.module(x)  # Add the input h0 to the output of the module (residual connection)


# Define a class RecursiveBlock which inherits from nn.Module
class RecursiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, U):
        super(RecursiveBlock, self).__init__()  # Initialize the parent class nn.Module
        self.U = U  # Number of recursive iterations
        self.h0 = ConvLayer(in_channels, out_channels)  # Initial ConvLayer to transform input
        self.ru = ResidualUnit(out_channels)  # Residual unit with out_channels as the number of features

    # Define the forward pass of the RecursiveBlock
    def forward(self, x):
        h0 = self.h0(x)  # Pass input x through the initial ConvLayer
        x = h0  # Initialize x with h0 for the recursive process
        for i in range(self.U):  # Apply the ResidualUnit U times
            x = self.ru(h0, x)  # Pass h0 and x through the ResidualUnit and update x
        return x  # Return the final output after U recursive iterations


# Define a class DRRN (Deep Recursive Residual Network) which inherits from nn.Module
class DRRN(nn.Module):
    def __init__(self, B, U, num_channels=1, num_features=128):
        super(DRRN, self).__init__()  # Initialize the parent class nn.Module
        # Create a sequence of B RecursiveBlock instances
        self.rbs = nn.Sequential(
            *[RecursiveBlock(num_channels if i == 0 else num_features, num_features, U) for i in range(B)]
        )
        self.rec = ConvLayer(num_features, num_channels)  # Final ConvLayer to reduce features to original channels
        self._initialize_weights()  # Initialize weights of the network

    # Method to initialize weights of convolutional layers
    def _initialize_weights(self):
        for m in self.modules():  # Iterate over all modules in the network
            if isinstance(m, nn.Conv2d):  # Check if the module is a Conv2d layer
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Initialize weights with Kaiming normalization suited for ReLU
                if m.bias is not None:  # If the bias exists, initialize it to zero
                    nn.init.constant_(m.bias, 0)

    # Define the forward pass of the DRRN
    def forward(self, x):
        residual = x  # Save the original input as residual for later addition
        x = self.rbs(x)  # Pass the input through the sequence of RecursiveBlocks
        x = self.rec(x)  # Pass the output through the final ConvLayer
        x += residual  # Add the residual (original input) to the output (residual connection)
        return x  # Return the final output of the network
