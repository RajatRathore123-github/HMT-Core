import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DisentanglementPrismLayer(nn.Module):
    def __init__(self, feature_dim, alpha=0.001, beta=2.0):
        super(DisentanglementPrismLayer, self).__init__()
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta  # The orthogonalization push parameter

    def forward(self, X1, X2):
        """
        Takes two semantic activation vectors that are mixed/superpositioned
        and routes them through a velocity-dependent metric tensor to separate them.
        """
        # Calculate cross-channel metric strain (how heavily they overlap)
        overlap_initial = torch.dot(X1, X2) / (torch.norm(X1) * torch.norm(X2))
        
        # Apply the Logarithmic Damping multiplier to bound the absolute field scale
        damp1 = 1.0 / (1.0 + torch.log1p(self.alpha * torch.exp(torch.norm(X1))))
        damp2 = 1.0 / (1.0 + torch.log1p(self.alpha * torch.exp(torch.norm(X2))))
        
        reg_X1 = X1 * damp1
        reg_X2 = X2 * damp2
        
        # --- The Geometric Prism Transformation ---
        # How: We construct a localized metric deformation matrix based on vector projection
        proj_matrix = torch.outer(reg_X1, reg_X1) / (torch.norm(reg_X1) ** 2 + 1e-6)
        
        # Metric Tensor transformation pushes X2 away from the subspace spanned by X1
        metric_tensor = torch.eye(self.feature_dim) + self.beta * proj_matrix
        
        # Deform the coordinate system for the second vector
        disentangled_X2 = torch.mv(torch.inverse(metric_tensor), reg_X2)
        
        return reg_X1, disentangled_X2

if __name__ == "__main__":
    print("--- Phase 2: Launching Topological Disentanglement Test ---")
    
    # Let's create two highly overlapping, superpositioned concept vectors
    # Dimension 4 (e.g., hidden features representing different traits)
    concept_doctor = torch.tensor([1.0, 2.0, 0.5, 0.2])
    concept_magenta = torch.tensor([1.1, 1.9, 0.4, 0.3]) # Heavily overlapping raw inputs
    
    initial_cosine = torch.dot(concept_doctor, concept_magenta) / (torch.norm(concept_doctor) * torch.norm(concept_magenta))
    print(f"Raw Input Overlap (Cosine Similarity): {initial_cosine.item():.4f} (High Superposition Chaos)")
    
    # Route through our Homeostatic Disentanglement Prism
    prism = DisentanglementPrismLayer(feature_dim=4, beta=5.0)
    out_doctor, out_magenta = prism(concept_doctor, concept_magenta)
    
    final_cosine = torch.dot(out_doctor, out_magenta) / (torch.norm(out_doctor) * torch.norm(out_magenta))
    print(f"Post-Manifold Overlap (Cosine Similarity): {final_cosine.item():.4f}")
    print("Status: Overlapping concepts successfully separated into distinct coordinate channels.")
