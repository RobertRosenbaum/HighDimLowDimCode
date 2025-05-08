import numpy as np
import torch
import torch.nn.functional as F

# Some of these functions were originally written for numpy instead of torch, 
# so the current versions generate outputs as numpy arrays first, then convert to torch tensors
# This should be fixed later.


# Create a smooth Gaussian process by convolving
# white noise with a Gaussian kernel.
# Noise will have variance=1
def MakeSmoothGaussianProcess(taux, Nt, dt, N=1, device='cpu', method='fft', chunk_size=None):
    """
    Generate colored Gaussian noise with Gaussian autocovariance function
    R(t) = exp(-t^2/(2*taux^2)) 
    
    Args:
        taux: correlation time (width parameter of autocorrelation function)
        Nt: number of time points
        dt: time step
        N: number of processes
        device: computation device
        method: 'conv' for direct convolution, 'fft' for FFT-based
        chunk_size: for FFT method, process this many samples at once
    
    Returns:
        torch.Tensor: Array of shape (N, Nt) containing the colored noise
    """
    if method not in ['conv', 'fft']:
        raise ValueError("Method must be either 'conv' or 'fft'")
    
    if method == 'conv':
        # For conv method, we need kernel width = taux/sqrt(2) to get
        # autocorrelation width = taux
        kernel_width = taux/np.sqrt(2)
        
        # Make kernel
        taus = torch.arange(-4 * kernel_width, 4 * kernel_width + dt/2, dt).to(device)
        
        # Ensure odd length for symmetric kernel
        if len(taus) % 2 == 0:
            taus = torch.cat([taus, taus[-1:] + dt])
        
        # Normalize kernel for unit variance
        K = torch.exp(-taus**2 / (2 * kernel_width**2))
        K = K / (dt * K.sum())
        
        # Generate noise
        if N == 1:
            white_noise = (1/np.sqrt(dt)) * torch.randn(Nt).to(device)
            K = K[None,None,:]  # Add batch and channel dims
            white_noise = white_noise[None,None,:]  # Add batch and channel dims
            X = torch.squeeze(F.conv1d(white_noise, K, padding='same')*dt)
        else:
            K = K[None,None,:]
            white_noise = (1/np.sqrt(dt)) * torch.randn(N, 1, Nt).to(device)
            X = torch.squeeze(F.conv1d(white_noise, K, padding='same')*dt)
        
        # Normalize for unit variance
        X *= np.sqrt(np.sqrt(2*np.pi) * taux)
        
    else:  # FFT method
        # Process in chunks if requested
        if chunk_size is None:
            chunk_size = N
            
        # Calculate frequencies
        freqs = torch.fft.rfftfreq(Nt, dt).to(device)
        
        # Power spectrum for Gaussian autocorrelation with width taux
        # This is sqrt of FT of exp(-t²/(2τ²))
        kernel_fft = torch.exp(-2 * (np.pi * freqs * taux)**2 / 2)
        
        # Initialize output array
        X = torch.zeros(N, Nt, device=device)
        
        # Process in chunks
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_size_actual = end_idx - i
            
            # Generate white noise
            white_noise = (1/np.sqrt(dt)) * torch.randn(chunk_size_actual, Nt, device=device)
            
            # FFT of noise
            noise_fft = torch.fft.rfft(white_noise, dim=-1)
            
            # Multiply by sqrt of power spectrum
            colored_fft = noise_fft * kernel_fft[None, :]
            
            # Inverse FFT
            X[i:end_idx] = torch.fft.irfft(colored_fft, n=Nt, dim=-1)
        
        # Normalize for unit variance
        X *= np.sqrt(np.sqrt(2*np.pi) * taux)
    
    return X



def TorchPCA(x, scaled=False):
    """
    Compute PCA of a multivariate time series using SVD in PyTorch.
    
    Args:
        x (torch.Tensor): Input tensor of shape (Nt, N) where:
            Nt is the number of time steps
            N is the dimension of the time series
    
    Returns:
        principal_vectors (torch.Tensor): Matrix of principal component vectors of shape (N, N)
        explained_variance (torch.Tensor): Vector of explained variances in descending order
    """
    # Center the data
    x = x - x.mean(dim=0, keepdim=True)
    
    # Scale the data
    if scaled:
        x = x / x.std(dim=0, keepdim=True, unbiased=True)  # Add after centering
        stdx = 1
    else:
        # We still want to scale by the overall std for numerical stability 
        # We will scale S back, so all is well
        stdx = x.std()

    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    #S=S*stdx
    # Convert singular values to explained variance
    # Factor of 1/(Nt-1) for unbiased estimation
    S = (S ** 2) / (x.shape[0] - 1)        
    # Principal vectors are the right singular vectors
    V = Vh.T
    
    
    
    return V, S

# Conver torch tensor to numpy array for plotting
def ToNP(x):
  return x.detach().cpu().numpy()



def GetOrthonormalVectors(N, n, v0=None):
    if v0 is not None:
        # Normalize v0 if provided
        v0 = v0.view(-1) / torch.norm(v0)
        vectors = [v0]
        start = 1
    else:
        vectors = []
        start = 0

    for i in range(start, n):
        # Generate a random vector from a normal distribution
        v = torch.randn(N)

        # Gram-Schmidt process
        for u in vectors:
            v -= torch.dot(u, v) * u

        # Normalize the vector
        v = v / torch.norm(v)

        vectors.append(v)

    return torch.stack(vectors)


###
#####


# Function to generate blockwise ER connection matrix
# NsPre = tuple of ints containing number of pre neurons in each block
# Jm = matrix connection weights in each block
# P = matrix of connection probs in each block
# NsPost = number of post neurons in each block
# If NsPost == None, connectivity is assumed recurrent (so NsPre=NsPost)
def GetBlockErdosRenyi(NsPre,Jm,P,NsPost=None):

  # Convert tensors to numpy arrays.
  # Get rid of this after changing to PyTorch version
  if torch.is_tensor(NsPre):
      NsPre=NsPre.numpy()
  if torch.is_tensor(Jm):
      Jm=Jm.numpy()
  if torch.is_tensor(P):
      P=P.numpy()
  if torch.is_tensor(NsPost):
      NsPost=NsPost.numpy()

  if NsPost==None:
    NsPost=NsPre

  # # If Jm is a 1D array, reshape it to column vector
  # if len(Jm.shape)==1:
  #   Jm = np.array([Jm]).T
  # if len(P.shape)==1:
  #   P = np.array([P]).T

  Npre = int(np.sum(NsPre))
  Npost = int(np.sum(NsPost))
  cNsPre = np.cumsum(np.insert(NsPre,0,0)).astype(int)
  cNsPost = np.cumsum(np.insert(NsPost,0,0)).astype(int)
  J = np.zeros((Npost,Npre), dtype = np.float32)

  for j1,N1 in enumerate(NsPost):
    for j2,N2 in enumerate(NsPre):
      J[cNsPost[j1]:cNsPost[j1+1],cNsPre[j2]:cNsPre[j2+1]]=Jm[j1,j2]*(np.random.binomial(1, P[j1,j2], size=(N1, N2)))
  J = torch.tensor(J)
  return J




def DrawRecNet(ax):
    import networkx as nx

    n_pre=30
    n_post=50
    feedforward_sparsity=0.95
    recurrent_sparsity=0.95
    #figsize=(5, 3)

    np.random.seed(0)


    """
    Create a visualization of a neural network with:
    - Feedforward connections from pre to post neurons with specified sparsity
    - Recurrent connections between post neurons with specified sparsity

    Parameters:
    -----------
    n_pre : int
        Number of pre-synaptic neurons
    n_post : int
        Number of post-synaptic neurons
    feedforward_sparsity : float
        Fraction of feedforward connections to remove (0.0 = fully connected, 1.0 = no connections)
    recurrent_sparsity : float
        Fraction of recurrent connections to remove (0.0 = fully connected, 1.0 = no connections)
    figsize : tuple
        Figure size in inches
    """
    # Validate sparsity parameters
    if not 0 <= feedforward_sparsity < 1:
        raise ValueError("Feedforward sparsity must be between 0 and 1")
    if not 0 <= recurrent_sparsity < 1:
        raise ValueError("Recurrent sparsity must be between 0 and 1")

    # Create directed graph
    G = nx.DiGraph()

    # Create pre-synaptic neurons (randomly distributed)
    pre_neurons = [(i, {
        'pos': (-.25+np.random.random(), np.random.random()),
        'layer': 'pre'
    }) for i in range(n_pre)]

    # Create post-synaptic neurons (randomly distributed)
    post_neurons = [(i + n_pre, {
        'pos': (1.5 + 2.5 * np.random.random(), np.random.random()),
        'layer': 'post'
    }) for i in range(n_post)]

    # Add nodes to graph
    G.add_nodes_from(pre_neurons)
    G.add_nodes_from(post_neurons)

    # Create all possible feedforward connections (pre -> post)
    all_feedforward_edges = [(i, j + n_pre) for i in range(n_pre) for j in range(n_post)]

    # Randomly select feedforward edges based on sparsity
    n_feedforward = int(len(all_feedforward_edges) * (1 - feedforward_sparsity))
    selected_feedforward = np.random.choice(len(all_feedforward_edges), 
                                            size=n_feedforward, 
                                            replace=False)
    feedforward_edges = [all_feedforward_edges[i] for i in selected_feedforward]

    # Create all possible recurrent connections (post -> post), excluding self-connections
    all_recurrent_edges = [(i + n_pre, j + n_pre) 
                            for i in range(n_post) 
                            for j in range(n_post) 
                            if i != j]  # Exclude self-connections

    # Randomly select recurrent edges based on sparsity
    n_recurrent = int(len(all_recurrent_edges) * (1 - recurrent_sparsity))
    selected_recurrent = np.random.choice(len(all_recurrent_edges), 
                                        size=n_recurrent, 
                                        replace=False)
    recurrent_edges = [all_recurrent_edges[i] for i in selected_recurrent]

    # Add the selected edges to the graph
    G.add_edges_from(feedforward_edges)
    G.add_edges_from(recurrent_edges)

    # Create positions dictionary for drawing
    pos = nx.get_node_attributes(G, 'pos')

    # # Set up the plot
    # ax.figure(figsize=figsize)


    # Draw the feedforward edges
    nx.draw_networkx_edges(G, pos, 
                            edgelist=feedforward_edges, 
                            edge_color='green', 
                            alpha=0.25, 
                            arrows=False,
                            ax=ax)

    # Draw the recurrent edges
    nx.draw_networkx_edges(G, pos, 
                            edgelist=recurrent_edges, 
                            edge_color='red', 
                            alpha=0.25, 
                            arrows=False,
                            ax=ax) 

    # Draw pre-synaptic neurons
    pre_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == 'pre']
    nx.draw_networkx_nodes(G, pos, nodelist=pre_nodes, 
                            node_color='green', node_size=5,
                            ax=ax)

    # Draw post-synaptic neurons
    post_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == 'post']
    nx.draw_networkx_nodes(G, pos, nodelist=post_nodes, 
                            node_color='blue', node_size=5,
                            ax=ax)

    ax.text(1.55,1,"z",color='b',size=12)
    #plt.text(1.18,.85,"W",color=[.3,.3,1],size=32)
    ax.text(2.25,1.01,"W",color=[.85,.1,.1],size=12)
    ax.text(.05,1,"x",color='g',size=12)
    ax.axis('off')




def DrawFFwdNet(ax):
    import networkx as nx

    n_pre=30
    n_post=50
    feedforward_sparsity=0.95
    recurrent_sparsity=0.95
    #figsize=(5, 3)

    np.random.seed(0)


    """
    Create a visualization of a neural network with:
    - Feedforward connections from pre to post neurons with specified sparsity
    - Recurrent connections between post neurons with specified sparsity

    Parameters:
    -----------
    n_pre : int
        Number of pre-synaptic neurons
    n_post : int
        Number of post-synaptic neurons
    feedforward_sparsity : float
        Fraction of feedforward connections to remove (0.0 = fully connected, 1.0 = no connections)
    recurrent_sparsity : float
        Fraction of recurrent connections to remove (0.0 = fully connected, 1.0 = no connections)
    figsize : tuple
        Figure size in inches
    """
    # Validate sparsity parameters
    if not 0 <= feedforward_sparsity < 1:
        raise ValueError("Feedforward sparsity must be between 0 and 1")
    if not 0 <= recurrent_sparsity < 1:
        raise ValueError("Recurrent sparsity must be between 0 and 1")

    # Create directed graph
    G = nx.DiGraph()

    # Create pre-synaptic neurons (randomly distributed)
    pre_neurons = [(i, {
        'pos': (-.25+np.random.random(), np.random.random()),
        'layer': 'pre'
    }) for i in range(n_pre)]

    # Create post-synaptic neurons (randomly distributed)
    post_neurons = [(i + n_pre, {
        'pos': (1.5 + 2.5 * np.random.random(), np.random.random()),
        'layer': 'post'
    }) for i in range(n_post)]

    # Add nodes to graph
    G.add_nodes_from(pre_neurons)
    G.add_nodes_from(post_neurons)

    # Create all possible feedforward connections (pre -> post)
    all_feedforward_edges = [(i, j + n_pre) for i in range(n_pre) for j in range(n_post)]

    # Randomly select feedforward edges based on sparsity
    n_feedforward = int(len(all_feedforward_edges) * (1 - feedforward_sparsity))
    selected_feedforward = np.random.choice(len(all_feedforward_edges), 
                                            size=n_feedforward, 
                                            replace=False)
    feedforward_edges = [all_feedforward_edges[i] for i in selected_feedforward]

    # # Create all possible recurrent connections (post -> post), excluding self-connections
    # all_recurrent_edges = [(i + n_pre, j + n_pre) 
    #                         for i in range(n_post) 
    #                         for j in range(n_post) 
    #                         if i != j]  # Exclude self-connections

    # # Randomly select recurrent edges based on sparsity
    # n_recurrent = int(len(all_recurrent_edges) * (1 - recurrent_sparsity))
    # selected_recurrent = np.random.choice(len(all_recurrent_edges), 
    #                                     size=n_recurrent, 
    #                                     replace=False)
    # recurrent_edges = [all_recurrent_edges[i] for i in selected_recurrent]

    # Add the selected edges to the graph
    G.add_edges_from(feedforward_edges)
    #G.add_edges_from(recurrent_edges)

    # Create positions dictionary for drawing
    pos = nx.get_node_attributes(G, 'pos')

    # # Set up the plot
    # ax.figure(figsize=figsize)


    # Draw the feedforward edges
    nx.draw_networkx_edges(G, pos, 
                            edgelist=feedforward_edges, 
                            edge_color='red', 
                            alpha=0.25, 
                            arrows=False,
                            ax=ax)

    # # Draw the recurrent edges
    # nx.draw_networkx_edges(G, pos, 
    #                         edgelist=recurrent_edges, 
    #                         edge_color='red', 
    #                         alpha=0.25, 
    #                         arrows=False,
    #                         ax=ax) 

    # Draw pre-synaptic neurons
    pre_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == 'pre']
    nx.draw_networkx_nodes(G, pos, nodelist=pre_nodes, 
                            node_color='green', node_size=5,
                            ax=ax)

    # Draw post-synaptic neurons
    post_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == 'post']
    nx.draw_networkx_nodes(G, pos, nodelist=post_nodes, 
                            node_color='blue', node_size=5,
                            ax=ax)

    ax.text(1.85,1,"z",color='b',size=13)
    #plt.text(1.18,.85,"W",color=[.3,.3,1],size=32)
    ax.text(.85,.95,"W",color='r',size=14)
    ax.text(.05,1,"x",color='g',size=14)
    ax.axis('off')

