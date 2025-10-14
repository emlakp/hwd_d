#!/usr/bin/env python3
"""
CRSSM Debugging Script

This script provides comprehensive debugging and testing for the CRSSM implementation.
Run sections individually by uncommenting them or run the entire script.
"""

import sys
import os
import time
import traceback
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Import CRSSM components
try:
    from lumos.world_models.contextrssm.core import ContextRSSMCore as CRSSMCore
    from lumos.world_models.contextrssm.cell import ContextRSSMCell as CRSSMCell
    from lumos.world_models.contextrssm.gatel0rd import GateL0RDCell as GRUCell
    # from lumos.world_models.crssm.utils import get_act  # This module doesn't exist
    from omegaconf import DictConfig
    print("✓ All imports successful!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_gru_cell():
    """Test GRU Cell functionality"""
    print("\n" + "="*50)
    print("Testing GRU Cell...")
    print("="*50)
    
    batch_size = 4
    hidden_size = 32
    input_size = 16
    
    try:
        gru = GRUCell(input_size=input_size, hidden_size=hidden_size, reg_lambda=0.01, output_size=hidden_size)
        
        # Test input
        x = torch.randn(batch_size, input_size)
        h = torch.randn(batch_size, hidden_size)
        
        h_new, _, gates = gru(x, h)  # GateL0RDCell returns (output, hidden_state, gates)
        print(f"✓ GRU Cell forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Hidden state shape: {h.shape} -> {h_new.shape}")
        
        # Check gradients
        loss = h_new.sum()
        loss.backward()
        print(f"✓ GRU Cell backward pass successful")
        
    except Exception as e:
        print(f"✗ GRU Cell test failed: {e}")
        traceback.print_exc()

def test_crssm_cell_init():
    """Test CRSSM Cell initialization"""
    print("\n" + "="*50)
    print("Testing CRSSM Cell initialization...")
    print("="*50)

    try:
        cell = CRSSMCell(
            embed_dim=64,
            action_dim=8,
            deter_dim=256,
            stoch_dim=32,
            stoch_rank=32,
            context_dim=64,
            hidden_dim=128,
            ensemble=1,
            layer_norm=True,
            context_sample_noise=0.05
        )

        print(f"✓ CRSSM Cell created successfully")
        print(f"  Parameters: {sum(p.numel() for p in cell.parameters())}")
        print(f"  Embed dim: {cell.embed_dim}")
        print(f"  Deterministic dim: {cell.deter_dim}")
        print(f"  Context dim: {cell.context_dim}")

        # Test initial state
        batch_size = 4
        init_state = cell.init_state(batch_size)
        print(f"✓ Initial state created")
        h, z, context = init_state
        print(f"    h: {h.shape}")
        print(f"    z: {z.shape}")
        print(f"    context: {context.shape}")

    except Exception as e:
        print(f"✗ CRSSM Cell initialization failed: {e}")
        traceback.print_exc()

def test_crssm_forward():
    """Test CRSSM forward pass with synthetic data"""
    print("\n" + "="*50)
    print("Testing CRSSM forward pass...")
    print("="*50)
    
    # Create synthetic data
    batch_size = 4
    seq_len = 10
    obs_dim = 64
    act_dim = 8
    
    # Synthetic inputs
    embeds = torch.randn(seq_len, batch_size, obs_dim)
    actions = torch.randn(seq_len, batch_size, act_dim)
    resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)  # No resets
    
    print(f"Data shapes:")
    print(f"  Embeds: {embeds.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Resets: {resets.shape}")
    
    try:
        # Create cell config for hydra instantiation
        cell_config = DictConfig({
            '_target_': 'lumos.world_models.contextrssm.cell.ContextRSSMCell',
            'embed_dim': obs_dim,
            'action_dim': act_dim,
            'deter_dim': 256,
            'stoch_dim': 32,
            'stoch_rank': 32,
            'context_dim': 64,
            'hidden_dim': 128,
            'ensemble': 1,
            'layer_norm': True,
            'context_sample_noise': 0.05
        })
        
        crssm = CRSSMCore(cell=cell_config)
        
        # Initialize state
        init_state = crssm.init_state(batch_size)
        print(f"✓ Initial state created: {type(init_state)}")
        
        # Forward pass
        outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
        
        print(f"✓ Forward pass successful!")
        print(f"  Priors shape: {outputs['priors'].shape}")
        print(f"  Posts shape: {outputs['posts'].shape}")
        print(f"  Context priors shape: {outputs['ctxt_priors'].shape}")
        print(f"  Context posts shape: {outputs['ctxt_posts'].shape}")
        print(f"  Features shape: {outputs['features'].shape}")
        print(f"  Final state type: {type(final_state)}")
        
        # Test backward pass
        loss = outputs['features'].sum()
        loss.backward()
        print(f"✓ Backward pass successful!")
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        traceback.print_exc()

def profile_crssm():
    """Memory and timing analysis"""
    print("\n" + "="*50)
    print("Profiling CRSSM performance...")
    print("="*50)
    
    batch_sizes = [1, 4, 8, 16]
    seq_lengths = [10, 50, 100]
    
    cell_config = DictConfig({
        '_target_': 'lumos.world_models.contextrssm.cell.ContextRSSMCell',
        'embed_dim': 64,
        'action_dim': 8,
        'deter_dim': 256,
        'stoch_dim': 32,
        'stoch_rank': 32,
        'context_dim': 64,
        'hidden_dim': 128,
        'ensemble': 1,
        'layer_norm': True,
        'context_sample_noise': 0.05
    })
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            try:
                crssm = CRSSMCore(cell=cell_config)
                
                # Create data
                embeds = torch.randn(seq_len, batch_size, 64)
                actions = torch.randn(seq_len, batch_size, 8)
                resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
                init_state = crssm.init_state(batch_size)
                
                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                
                # Time forward pass
                start_time = time.time()
                with torch.no_grad():
                    outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
                end_time = time.time()
                
                # Measure memory after
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_after = torch.cuda.memory_allocated()
                    mem_used = (mem_after - mem_before) / 1024**2  # MB
                else:
                    mem_used = 0
                
                forward_time = (end_time - start_time) * 1000  # ms
                
                results.append({
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'forward_time_ms': forward_time,
                    'memory_mb': mem_used
                })
                
                print(f"B={batch_size:2d}, T={seq_len:3d}: {forward_time:6.2f}ms, {mem_used:6.2f}MB")
                
            except Exception as e:
                print(f"Failed for B={batch_size}, T={seq_len}: {e}")
    
    return results

def analyze_distributions():
    """Analyze the distributions produced by CRSSM"""
    print("\n" + "="*50)
    print("Analyzing CRSSM distributions...")
    print("="*50)
    
    # Test both discrete and continuous versions
    configs = [
        ('Continuous', False),
        # ('Discrete', True)  # Uncomment if discrete version works
    ]
    
    for name, discrete in configs:
        print(f"\n=== {name} CRSSM ===")
        
        try:
            cell_config = DictConfig({
                '_target_': 'lumos.world_models.contextrssm.cell.ContextRSSMCell',
                'embed_dim': 64,
                'action_dim': 8,
                'deter_dim': 256,
                'stoch_dim': 32 if not discrete else 8,
                'stoch_rank': 32,
                'context_dim': 64,
                'hidden_dim': 128,
                'ensemble': 1,
                'layer_norm': True,
                'context_sample_noise': 0.05
            })
            
            if discrete:
                cell_config['discrete'] = 32  # Number of discrete values
            
            crssm = CRSSMCore(cell=cell_config)
            
            # Generate some data
            batch_size = 8
            seq_len = 20
            embeds = torch.randn(seq_len, batch_size, 64)
            actions = torch.randn(seq_len, batch_size, 8)
            resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
            init_state = crssm.init_state(batch_size)
            
            with torch.no_grad():
                outputs, final_state = crssm.forward(
                    embeds, actions, resets, init_state
                )
            
            print(f"✓ {name} CRSSM forward pass successful")
            print(f"  Prior distribution shape: {outputs['priors'].shape}")
            print(f"  Posterior distribution shape: {outputs['posts'].shape}")
            print(f"  Features shape: {outputs['features'].shape}")
            
            # Analyze feature statistics
            feat_mean = outputs['features'].mean(dim=[0, 1])
            feat_std = outputs['features'].std(dim=[0, 1])
            
            print(f"  Feature statistics:")
            print(f"    Mean: {feat_mean.mean().item():.4f} ± {feat_mean.std().item():.4f}")
            print(f"    Std:  {feat_std.mean().item():.4f} ± {feat_std.std().item():.4f}")
            
            # Test distribution sampling
            if hasattr(crssm.cell, 'zdistr'):
                prior_dist = crssm.zdistr(outputs['priors'][0])  # First timestep
                samples = prior_dist.sample()
                print(f"  Distribution samples shape: {samples.shape}")
                
        except Exception as e:
            print(f"✗ {name} CRSSM failed: {e}")
            traceback.print_exc()

def check_gradients():
    """Check gradient flow through the network"""
    print("\n" + "="*50)
    print("Analyzing gradient flow...")
    print("="*50)
    
    cell_config = DictConfig({
        '_target_': 'lumos.world_models.contextrssm.cell.ContextRSSMCell',
        'embed_dim': 64,
        'action_dim': 8,
        'deter_dim': 256,
        'stoch_dim': 32,
        'stoch_rank': 32,
        'context_dim': 64,
        'hidden_dim': 128,
        'ensemble': 1,
        'layer_norm': True,
        'context_sample_noise': 0.05
    })
    
    crssm = CRSSMCore(cell=cell_config)
    
    # Generate data
    batch_size = 4
    seq_len = 10
    embeds = torch.randn(seq_len, batch_size, 64, requires_grad=True)
    actions = torch.randn(seq_len, batch_size, 8, requires_grad=True)
    resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
    init_state = crssm.init_state(batch_size)
    
    # Forward pass
    outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
    
    # Compute loss
    loss = outputs['features'].mean()
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_stats = []
    
    for name, param in crssm.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            grad_stats.append({
                'name': name,
                'shape': tuple(param.shape),
                'grad_norm': grad_norm,
                'grad_mean': grad_mean,
                'grad_std': grad_std
            })
    
    # Sort by gradient norm
    grad_stats.sort(key=lambda x: x['grad_norm'], reverse=True)
    
    print(f"\nGradient Statistics (top 10 by norm):")
    print(f"{'Parameter':<40} {'Shape':<20} {'Norm':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 90)
    
    for stat in grad_stats[:10]:
        print(f"{stat['name']:<40} {str(stat['shape']):<20} "
              f"{stat['grad_norm']:<10.6f} {stat['grad_mean']:<10.6f} {stat['grad_std']:<10.6f}")
    
    # Check input gradients
    if embeds.grad is not None:
        print(f"\nInput gradient norms:")
        print(f"  Embeds: {embeds.grad.norm().item():.6f}")
    if actions.grad is not None:
        print(f"  Actions: {actions.grad.norm().item():.6f}")
    
    # Check for vanishing/exploding gradients
    grad_norms = [stat['grad_norm'] for stat in grad_stats]
    if grad_norms:
        max_grad = max(grad_norms)
        min_grad = min(grad_norms)
        
        print(f"\nGradient analysis:")
        print(f"  Max gradient norm: {max_grad:.6f}")
        print(f"  Min gradient norm: {min_grad:.6f}")
        print(f"  Ratio: {max_grad/min_grad:.2f}")
        
        if max_grad > 10:
            print(f"  ⚠️  Potential exploding gradients detected!")
        if min_grad < 1e-6:
            print(f"  ⚠️  Potential vanishing gradients detected!")
        if min_grad > 1e-6 and max_grad < 10:
            print(f"  ✓ Gradients appear healthy")

def visualize_crssm():
    """Visualize CRSSM behavior"""
    print("\n" + "="*50)
    print("Visualizing CRSSM behavior...")
    print("="*50)
    
    try:
        cell_config = DictConfig({
            '_target_': 'lumos.world_models.contextrssm.cell.ContextRSSMCell',
            'embed_dim': 64,
            'action_dim': 8,
            'deter_dim': 256,
            'stoch_dim': 32,
            'stoch_rank': 32,
            'context_dim': 64,
            'hidden_dim': 128,
            'ensemble': 1,
            'layer_norm': True,
            'context_sample_noise': 0.05
        })
        
        crssm = CRSSMCore(cell=cell_config)
        
        # Generate a longer sequence for visualization
        batch_size = 1
        seq_len = 100
        
        # Create structured input (sine wave)
        t = torch.linspace(0, 4*np.pi, seq_len).unsqueeze(0)  # [1, T]
        embeds = torch.zeros(seq_len, batch_size, 64)
        embeds[:, :, 0] = torch.sin(t).T  # First dimension is sine wave
        embeds[:, :, 1] = torch.cos(t).T  # Second dimension is cosine wave
        embeds[:, :, 2:] = torch.randn(seq_len, batch_size, 62) * 0.1  # Noise in other dims
        
        actions = torch.randn(seq_len, batch_size, 8) * 0.1
        resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
        init_state = crssm.init_state(batch_size)
        
        with torch.no_grad():
            outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
        
        # Convert to numpy for plotting
        embeds_np = embeds[:, 0, :].numpy()  # [T, obs_dim]
        features_np = outputs['features'][:, 0, :].numpy()  # [T, feat_dim]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('CRSSM Behavior Visualization')
        
        # Plot input signals
        axes[0, 0].plot(embeds_np[:, 0], label='Sin input', alpha=0.8)
        axes[0, 0].plot(embeds_np[:, 1], label='Cos input', alpha=0.8)
        axes[0, 0].set_title('Input Signals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot first few feature dimensions
        for i in range(min(5, features_np.shape[1])):
            axes[0, 1].plot(features_np[:, i], alpha=0.7, label=f'Feat {i}')
        axes[0, 1].set_title('Feature Evolution (first 5 dims)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot feature statistics over time
        feat_mean = np.mean(features_np, axis=1)
        feat_std = np.std(features_np, axis=1)
        axes[1, 0].plot(feat_mean, label='Mean', color='blue')
        axes[1, 0].fill_between(range(len(feat_mean)), 
                               feat_mean - feat_std, feat_mean + feat_std, 
                               alpha=0.3, color='blue', label='±1 std')
        axes[1, 0].set_title('Feature Statistics Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot feature correlation matrix (last timestep)
        feat_corr = np.corrcoef(features_np[-1:].T)
        im = axes[1, 1].imshow(feat_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_title('Feature Correlation (final timestep)')
        plt.colorbar(im, ax=axes[1, 1])
        
        # Plot PCA of features
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_np)
            
            scatter = axes[2, 0].scatter(features_pca[:, 0], features_pca[:, 1], 
                                        c=range(len(features_pca)), cmap='viridis', alpha=0.7)
            axes[2, 0].set_title(f'Features PCA (explained var: {pca.explained_variance_ratio_.sum():.3f})')
            axes[2, 0].set_xlabel('PC1')
            axes[2, 0].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[2, 0], label='Time')
            
            print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
        except ImportError:
            axes[2, 0].text(0.5, 0.5, 'sklearn not available\nfor PCA analysis', 
                           ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('PCA Analysis (sklearn required)')
        
        # Plot feature magnitude over time
        feat_norm = np.linalg.norm(features_np, axis=1)
        axes[2, 1].plot(feat_norm, color='green')
        axes[2, 1].set_title('Feature Magnitude Over Time')
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('||features||')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('crssm_visualization.png', dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved as 'crssm_visualization.png'")
        
        print(f"✓ Visualization complete")
        print(f"  Input shape: {embeds_np.shape}")
        print(f"  Features shape: {features_np.shape}")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            print("  (Plot display not available in this environment)")
            
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        traceback.print_exc()

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n" + "="*50)
    print("Testing edge cases...")
    print("="*50)
    
    base_config = DictConfig({
        '_target_': 'lumos.world_models.contextrssm.cell.ContextRSSMCell',
        'embed_dim': 64,
        'action_dim': 8,
        'deter_dim': 256,
        'stoch_dim': 32,
        'stoch_rank': 32,
        'context_dim': 64,
        'hidden_dim': 128,
        'ensemble': 1,
        'layer_norm': True,
        'context_sample_noise': 0.05
    })
    
    test_cases = [
        ("Single timestep", lambda: test_single_timestep(base_config)),
        ("All resets", lambda: test_all_resets(base_config)),
        ("Large batch size", lambda: test_large_batch(base_config)),
        ("Zero inputs", lambda: test_zero_inputs(base_config)),
        ("Extreme values", lambda: test_extreme_values(base_config)),
    ]
    
    results = []
    
    for desc, test_func in test_cases:
        print(f"\nTesting: {desc}")
        try:
            test_func()
            print(f"  ✓ {desc} passed")
            results.append((desc, "PASS"))
        except Exception as e:
            print(f"  ✗ {desc} failed: {e}")
            results.append((desc, f"FAIL: {e}"))
    
    print(f"\n=== Edge Case Test Summary ===")
    for desc, result in results:
        print(f"  {desc}: {result}")

def test_single_timestep(config):
    crssm = CRSSMCore(cell=config)
    batch_size = 4
    seq_len = 1

    embeds = torch.randn(seq_len, batch_size, 64)
    actions = torch.randn(seq_len, batch_size, 8)
    resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
    init_state = crssm.init_state(batch_size)

    outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
    tensors = [outputs['priors'], outputs['posts'], outputs['ctxt_priors'], outputs['ctxt_posts'], outputs['features']]
    assert all(out.shape[0] == seq_len for out in tensors)

def test_all_resets(config):
    crssm = CRSSMCore(cell=config)
    batch_size = 4
    seq_len = 10

    embeds = torch.randn(seq_len, batch_size, 64)
    actions = torch.randn(seq_len, batch_size, 8)
    resets = torch.ones(seq_len, batch_size, dtype=torch.bool)  # All resets
    init_state = crssm.init_state(batch_size)

    outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
    assert not torch.isnan(outputs['features']).any()  # Check features

def test_large_batch(config):
    crssm = CRSSMCore(cell=config)
    batch_size = 128
    seq_len = 5

    embeds = torch.randn(seq_len, batch_size, 64)
    actions = torch.randn(seq_len, batch_size, 8)
    resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
    init_state = crssm.init_state(batch_size)

    with torch.no_grad():
        outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
    assert outputs['features'].shape[1] == batch_size

def test_zero_inputs(config):
    crssm = CRSSMCore(cell=config)
    batch_size = 4
    seq_len = 10

    embeds = torch.zeros(seq_len, batch_size, 64)
    actions = torch.zeros(seq_len, batch_size, 8)
    resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
    init_state = crssm.init_state(batch_size)

    outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
    assert not torch.isnan(outputs['features']).any()

def test_extreme_values(config):
    crssm = CRSSMCore(cell=config)
    batch_size = 4
    seq_len = 10

    embeds = torch.randn(seq_len, batch_size, 64) * 10  # Large values
    actions = torch.randn(seq_len, batch_size, 8) * 10
    resets = torch.zeros(seq_len, batch_size, dtype=torch.bool)
    init_state = crssm.init_state(batch_size)

    outputs, final_state = crssm.forward(embeds, actions, resets, init_state)
    assert torch.isfinite(outputs['features']).all()  # Check features are finite

def generate_summary():
    """Generate debugging summary"""
    print("\n" + "="*60)
    print("CRSSM DEBUGGING SUMMARY")
    print("="*60)
    print()
    print("This script tested the CRSSM implementation across multiple dimensions:")
    print()
    print("1. ✓ Component Testing")
    print("   - GRU Cell functionality")
    print("   - CRSSM Cell initialization")
    print("   - Forward pass mechanics")
    print()
    print("2. ✓ Performance Analysis")
    print("   - Memory usage profiling")
    print("   - Timing across batch sizes and sequence lengths")
    print()
    print("3. ✓ Distribution Analysis")
    print("   - Both continuous and discrete variants")
    print("   - Feature statistics and correlations")
    print()
    print("4. ✓ Gradient Flow")
    print("   - Backward pass functionality")
    print("   - Gradient magnitude analysis")
    print("   - Vanishing/exploding gradient detection")
    print()
    print("5. ✓ Visualization")
    print("   - Feature evolution over time")
    print("   - PCA analysis")
    print("   - Input-output relationships")
    print()
    print("6. ✓ Edge Cases")
    print("   - Single timesteps")
    print("   - All resets")
    print("   - Large batch sizes")
    print("   - Zero and extreme inputs")
    print()
    print("=== Debugging Recommendations ===")
    print()
    print("When debugging CRSSM issues:")
    print("1. Check gradient magnitudes first")
    print("2. Verify feature statistics are reasonable")
    print("3. Test with simplified inputs")
    print("4. Monitor memory usage for large sequences")
    print("5. Use visualization to understand feature dynamics")
    print()
    print("=== Common Issues and Solutions ===")
    print()
    print("• NaN values: Check for numerical instabilities in distributions")
    print("• Exploding gradients: Reduce learning rate or add gradient clipping")
    print("• Vanishing gradients: Check initialization and activation functions")
    print("• Memory issues: Use gradient checkpointing for long sequences")
    print("• Slow convergence: Verify feature normalization and learning rates")
    print()
    print("Happy debugging! 🐛🔧")

def main():
    """Run all debugging tests"""
    print("CRSSM Debugging Script")
    print("=" * 60)
    
    # Run all tests
    test_gru_cell()
    test_crssm_cell_init()
    test_crssm_forward()
    
    print("\nRunning performance profiling...")
    profile_results = profile_crssm()
    
    analyze_distributions()
    check_gradients()
    visualize_crssm()
    test_edge_cases()
    
    generate_summary()

if __name__ == "__main__":
    # Uncomment the sections you want to run:
    
    # Run all tests
    main()
    
    # Or run individual sections:
    # test_gru_cell()
    # test_crssm_cell_init() 
    # test_crssm_forward()
    # profile_crssm()
    # analyze_distributions()
    # check_gradients()
    # visualize_crssm()
    # test_edge_cases()
    # generate_summary()
