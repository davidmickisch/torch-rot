import numpy as np
from scipy.spatial.transform import Rotation as R

from torch_rot.rotations import rotation_matrix, rotate_vector

def create_plane_vectors(dim):
    plane = np.random.random(size=(dim, 2)) - 0.5
    q, _ = np.linalg.qr(plane)
    n1, n2 = q.T
    return n1, n2

def is_small(arr: np.array) -> bool:
    return np.abs(arr.flatten()).sum() < 1e-4

def test_rotation_matrix_in_3d():
    n1, n2 = create_plane_vectors(dim=3)
    rot_axis = np.cross(n1, n2)
    rot_angle = 2*np.pi*(np.random.random() - 0.5)
    rot_vec = rot_angle * rot_axis
    
    true_rot_matrix = R.from_rotvec(rot_vec).as_matrix()
    
    rot_matrix = rotation_matrix(theta = rot_angle,
                          n1 = n1,
                          n2 = n2)
    
    assert is_small(true_rot_matrix - rot_matrix)
    
    
def test_rotate_random_vector_in_3d():
    n1, n2 = create_plane_vectors(dim=3)
    rot_axis = np.cross(n1, n2)
    rot_angle = 2*np.pi*(np.random.random() - 0.5)
    rot_vec = rot_angle * rot_axis
    
    true_rot_matrix = R.from_rotvec(rot_vec).as_matrix()
    
    random_vec = np.random.random(size=3) - 0.5
    
    true_rotated_random_vec = true_rot_matrix.dot(random_vec)
    rotated_random_vec = rotate_vector(
        theta=rot_angle,
        n1=n1,
        n2=n2,
        vec=random_vec
    )
    
    assert is_small(true_rotated_random_vec - rotated_random_vec)
    
def test_rotate_rotation_axis_in_30d():
    dim=30
    arr = np.random.random(size=(dim, 3)) - 0.5
    q, _ = np.linalg.qr(arr)
    n1, n2, axis = q.T
    
    rot_angle = 2*np.pi*(np.random.random() - 0.5)
    
    rotated_axis = rotate_vector(
        theta=rot_angle,
        n1=n1,
        n2=n2,
        vec=axis
    )
    
    assert is_small(rotated_axis - axis)