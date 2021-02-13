"""This is a module for performing high dimensional rotations."""

import torch
import numpy as np

def rotation_matrix(theta: torch.Tensor, n_1 : torch.Tensor, n_2 : torch.Tensor) -> torch.Tensor:
    """
    This method returns a rotation matrix which rotates any vector 
    in the 2 dimensional plane spanned by 
    @n1 and @n2 an angle @theta. The vectors @n1 and @n2 have to be orthogonal.
    Inspired by 
    https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    :param @n1: first vector spanning 2-d rotation plane, needs to be orthogonal to @n2
    :param @n2: second vector spanning 2-d rotation plane, needs to be orthogonal to @n1
    :param @theta: rotation angle
    :returns : rotation matrix
    """
    dim = len(n_1)
    assert len(n_1) == len(n_2)
    assert np.abs(n_1.dot(n_2) < 1e-4)
    return (np.eye(dim) +
        (np.outer(n_2, n_1) - np.outer(n_1, n_2)) * np.sin(theta) +
        (np.outer(n_1, n_1) + np.outer(n_2, n_2)) * (np.cos(theta) - 1)
    )

def rotate_vector(
    theta: torch.Tensor, 
    n_1 : torch.Tensor, 
    n_2 : torch.Tensor, 
    vec : torch.Tensor) -> torch.Tensor:
    """
    This method returns a rotated vector which rotates @vec in the 2 dimensional plane spanned by 
    @n1 and @n2 by an angle @theta. The vectors @n1 and @n2 have to be orthogonal.
    Inspired by 
    https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    :param @n1: first vector spanning 2-d rotation plane, needs to be orthogonal to @n2
    :param @n2: second vector spanning 2-d rotation plane, needs to be orthogonal to @n1
    :param @theta: rotation angle
    :param @vec: vector to be rotated
    :returns : rotation matrix
    """
    assert len(n_1) == len(n_2)
    assert len(n_1) == len(vec)
    assert np.abs(n_1.dot(n_2) < 1e-4)
    
    n1_dot_vec = n_1.dot(vec)
    n2_dot_vec = n_2.dot(vec)
    
    return (vec +
        (n_2 * n1_dot_vec - n_1 * n2_dot_vec) * np.sin(theta) +
        (n_1 * n1_dot_vec + n_2 * n2_dot_vec) * (np.cos(theta) - 1)
    )
