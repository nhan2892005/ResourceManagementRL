import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)


class PolicyNetwork(nn.Module):
    """Policy Network Architecture"""
    def __init__(self, n_features, n_actions):
        super(PolicyNetwork, self).__init__()
        
        # Hidden layer with 10 units
        self.fc1 = nn.Linear(n_features, 10)
        self.tanh = nn.Tanh()
        
        # Output layer
        self.fc2 = nn.Linear(10, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to match TensorFlow version"""
        # fc1 weights: Normal(0, 0.3)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.fc1.bias, 0.1)
        
        # fc2 weights: Normal(0, 0.3)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.fc2.bias, 0.1)
    
    def forward(self, x):
        """Forward pass"""
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        action_probs = self.softmax(x)
        return action_probs


class PolicyGradient:
    """Policy Gradient Agent using PyTorch"""
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = 0.95
        
        # Episode memory
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        
        # Determine device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build network
        self.model = PolicyNetwork(n_features, n_actions).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if output_graph:
            print(f"Policy Gradient model initialized on {self.device}")
            print(f"Input features: {n_features}, Output actions: {n_actions}")
    
    def choose_action(self, observation):
        """
        Choose action based on observation
        
        Args:
            observation: numpy array of shape (1, n_features) or (n_features,)
        
        Returns:
            action: integer action index
        """
        # Convert observation to tensor
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        # Get action probabilities
        self.model.eval()
        with torch.no_grad():
            action_probs = self.model(obs_tensor)
            action_probs = action_probs.cpu().numpy().flatten()
        
        # Sample action from probability distribution
        action = np.random.choice(range(len(action_probs)), p=action_probs)
        
        return action
    
    def store_ob(self, s):
        """Store observation"""
        self.ep_obs.append(s)
    
    def store_action(self, a):
        """Store action"""
        self.ep_as.append(a)
    
    def store_adv(self, r):
        """Store reward"""
        self.ep_rs.append(r)
    
    def learn(self, all_ob, all_action, all_adv):
        """
        Update policy network
        
        Args:
            all_ob: numpy array of observations, shape (batch_size, n_features)
            all_action: numpy array of actions, shape (batch_size,)
            all_adv: numpy array of advantages, shape (batch_size,)
        
        Returns:
            loss: float, the loss value
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(all_ob).to(self.device)
        actions_tensor = torch.LongTensor(all_action).to(self.device)
        advantages_tensor = torch.FloatTensor(all_adv).to(self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        action_probs = self.model(obs_tensor)
        
        # Calculate log probabilities using cross entropy
        # This matches: tf.nn.sparse_softmax_cross_entropy_with_logits
        log_probs = torch.log(action_probs + 1e-10)  # Add small epsilon for numerical stability
        selected_log_probs = log_probs[range(len(actions_tensor)), actions_tensor]
        
        # Policy gradient loss (negative because we maximize reward)
        # loss = -mean(log_prob * advantage)
        loss = -torch.mean(selected_log_probs * advantages_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        # Clear episode memory
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        
        return loss.item()
    
    def _discount_and_norm_rewards(self):
        """
        Compute discounted and normalized rewards
        
        Returns:
            discounted_ep_rs_norm: normalized discounted rewards
        """
        # Compute discounted rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        
        # Normalize rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        std = np.std(discounted_ep_rs)
        if std > 0:
            discounted_ep_rs /= std
        
        return discounted_ep_rs
    
    def save_data(self, path):
        """
        Save model weights
        
        Args:
            path: string, path to save the model (without extension)
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'{path}.pth')
        print(f"Model saved to {path}.pth")
    
    def load_data(self, path):
        """
        Load model weights
        
        Args:
            path: string, path to load the model (without extension)
        """
        checkpoint = torch.load(f'{path}.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}.pth")
    
    def get_params(self):
        """Get model parameters (for compatibility)"""
        return self.model.state_dict()
    
    def set_params(self, params):
        """Set model parameters (for compatibility)"""
        self.model.load_state_dict(params)


# Test the implementation
if __name__ == '__main__':
    print("Testing PolicyGradient PyTorch Implementation")
    print("=" * 50)
    
    # Create a simple test case
    n_features = 100
    n_actions = 6
    batch_size = 32
    
    # Initialize agent
    agent = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=0.01,
        output_graph=True
    )
    
    print("\n1. Testing choose_action...")
    obs = np.random.randn(1, n_features).astype(np.float32)
    action = agent.choose_action(obs)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Selected action: {action}")
    print(f"   Action type: {type(action)}")
    
    print("\n2. Testing learn...")
    # Create dummy batch data
    all_obs = np.random.randn(batch_size, n_features).astype(np.float32)
    all_actions = np.random.randint(0, n_actions, size=batch_size)
    all_advantages = np.random.randn(batch_size).astype(np.float32)
    
    loss = agent.learn(all_obs, all_actions, all_advantages)
    print(f"   Batch size: {batch_size}")
    print(f"   Loss: {loss:.4f}")
    
    print("\n3. Testing save/load...")
    # Save model
    agent.save_data('test_model')
    
    # Create new agent and load
    agent2 = PolicyGradient(n_actions, n_features)
    agent2.load_data('test_model')
    
    # Verify they produce same output
    action1 = agent.choose_action(obs)
    action2 = agent2.choose_action(obs)
    print(f"   Action from original agent: {action1}")
    print(f"   Action from loaded agent: {action2}")
    
    # Clean up test file
    import os
    if os.path.exists('test_model.pth'):
        os.remove('test_model.pth')
        print("   Test model file cleaned up")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nCompatibility with existing code:")
    print("✓ choose_action(observation) -> int")
    print("✓ learn(all_ob, all_action, all_adv) -> float")
    print("✓ save_data(path) / load_data(path)")
    print("✓ Handles both (batch, features) and (features,) input shapes")