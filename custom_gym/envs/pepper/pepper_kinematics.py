import math
import numpy as np
import scipy as sp
from scipy import linalg



L1 = 0.14974
L2 = 0.015
L3 = 0.1812
L4 = 0
L5 = 0.150
L6 = 0.0695
L7 = 0.0303

p = np.array([0,0,0,1])
v0 = np.array([[1],[0],[0],[0]])
v1 = np.array([[0],[1],[0],[0]])
v2 = np.array([[0],[0],[1],[0]])

right_arm_tags = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
right_arm_initial_pose = [1.0, -0.2, 1.57-0.2, 1.0, -1.57]
right_arm_work_pose = [0.8, -0.2, 1.57-0.2, 0.9, -1.57]

_inverse_case = [1.0, -1.0, -1.0, -1.0, -1.0]

left_arm_tags = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
left_arm_initial_pose = [p[0] * p[1] for p in zip(right_arm_initial_pose, _inverse_case)]
left_arm_work_pose = [p[0] * p[1] for p in zip(right_arm_work_pose, _inverse_case)]

def transX(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[1, 0, 0, x], [0, c, -s, y], [0, s, c, z], [0, 0, 0, 1]])

def transY(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[c, 0, -s, x], [0, 1, 0, y], [s, 0, c, z], [0, 0, 0, 1]])

def transZ(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def rotation_matrix_to_rpy(RM):
    sx = RM[0,0]
    sy = RM[1,0]
    sz = RM[2,0]
    nx = RM[0,1]
    ny = RM[1,1]
    nz = RM[2,1]
    ax = RM[0,2]
    ay = RM[1,2]
    az = RM[2,2]

    #roll = np.arctan2( nz , az )
    #pitch  = np.arctan2( -sz , np.sqrt( nz*nz + az*az ) )
    #yaw = np.arctan2( sy , sx )

    roll = math.atan( nz / az )
    pitch  = math.atan( -sz / math.sqrt( nz*nz + az*az ) )
    yaw = math.atan( sy / sx )
    
    return np.array([roll, pitch, yaw])

def calc_fk_and_jacob(angles, jacob=True, right=True):
    _L1_ = -L1 if right else L1
    _L2_ = -L2 if right else L2

    T1 = transY(-angles[0], 0, _L1_, 0)
    T2 = transZ(angles[1], 0, 0, 0)
    Td = transY(9.0/180.0*math.pi, L3, _L2_, 0)
    T3 = transX(angles[2], 0, 0, 0)
    T4 = transZ(angles[3], 0, 0, 0)
    T5 = transX(angles[4], L5, 0, 0)
    T6 = transZ(0, L6, 0, -L7)
    
    T1Abs = T1
    T2Abs = T1Abs.dot(T2)
    TdAbs = T2Abs.dot(Td)
    T3Abs = TdAbs.dot(T3)
    T4Abs = T3Abs.dot(T4)
    T5Abs = T4Abs.dot(T5)
    T6Abs = T5Abs.dot(T6)

    pos = T6Abs.dot(p)
    ori = T6Abs[0:3,0:3]

    if not jacob:
        return pos, ori

    OfstT1 = L1 * T1Abs.dot(v1)
    OfstTd = TdAbs.dot(np.array([[L3], [L2], [0], [0]]))
    OfstT5 = L5 * T5Abs.dot(v0)
    OfstT6 = T5Abs.dot(np.array([[L6], [0], [-L7], [0]]))

    vec6 = OfstT6
    vec5 = vec6 + OfstT5
    vec4 = vec5
    vec3 = vec4
    vecd = vec3 + OfstTd
    vec2 = vecd
    vec1 = vec2 + OfstT1
    
    j1 = T1Abs.dot(v1)
    j2 = T2Abs.dot(v2)
    jd = TdAbs.dot(v1)
    j3 = T3Abs.dot(v0)
    j4 = T4Abs.dot(v2)
    j5 = T5Abs.dot(v0)
    
    J1 = cross(j1, vec1)
    J2 = cross(j2, vec2)
    J3 = cross(j3, vec3)
    J4 = cross(j4, vec4)
    J5 = cross(j5, vec5)
    
    J = np.c_[J1, J2, J3, J4, J5]
    return pos, ori, J

def cross(j, v):
    t0 = j[1][0] * v[2][0] - j[2][0] * v[1][0]
    t1 = j[2][0] * v[0][0] - j[0][0] * v[2][0]
    t2 = j[0][0] * v[1][0] - j[1][0] * v[0][0]
    return np.array([[t0], [t1], [t2]])
        
def calc_inv_pos(angles, target_pos, target_ori, epsilon, right=True):
    p  = np.array([0,0,0,1])
    angs = np.array([a for a in angles])
    sum_old = 100000
    while True:
        pos, ori, j = calc_fk_and_jacob(angs, jacob=True, right=right)
        J = _calc_invJ(j)
        delta_pos = np.matrix((target_pos-pos[:3])[0:3]).transpose()
        v = (J * delta_pos).transpose()
        angs = np.squeeze(np.asarray(v)) + angs[:5]
        
        sum = 0
        for d in delta_pos:
            sum = sum + math.fabs(d)
        #sum = np.sum(delta_pos)
        if sum < epsilon:
            break
        if sum > sum_old:
            print ('# set_position error : Distance can not converged.')
            return None
        sum_old = sum
    return angs

def _calc_invJ(J, epsilon = 0.01):
    u, sigma, v = np.linalg.svd(J, full_matrices=0)
    sigma_ = [1/s if s > epsilon else 0 for s in sigma]
    rank = np.shape(J)[0]
    return np.matrix(v.transpose()) * np.matrix(linalg.diagsvd(sigma_, rank, rank)) * np.matrix(u.transpose())
  
def right_arm_get_position(angles):
    """
    Just calculate the position when joints on the pepper's right arm is in given positions

    Args:
      angles : Angles of right arm joints (list of 5 double values. unit is radian)
    
    Returns:
      A tuple of two arrays (position, orientation). orientation is presented as Matrix. Unit = meter.
      
      (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    return calc_fk_and_jacob(angles, jacob=False, right=True)

def left_arm_get_position(angles):
    """
    Just calculate the position when joints on the pepper's left arm is in given positions

    Args:
      angles : Angles of left arm joints (list of 5 double values. unit is radian)
    
    Returns:
      A tuple of two arrays (position, orientation). orientation is presented as Matrix. Unit = meter.
      
      (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    return calc_fk_and_jacob(angles, jacob=False, right=False)

def right_arm_set_position(angles, target_pos, target_ori, epsilon=0.0001):
    """
    Just calculate the joint angles when the Pepper's right hand position is in the given position
    
    Args:
      angles : Use the initial position of calculation. Unit = radian
      target_pos : List. [Px, Py, Pz]. Unit is meter.
      target_ori : np.array([[R00,R01,R02],[R10,R11,R12],[R20,R21,R22]])
      epsilon    : The threshold. If the distance between calculation result and target_position is lower than epsilon, this returns value.
    
    Returns:
      A list of joint angles (Unit is radian). If calculation fails, return None.
    """
    return calc_inv_pos(angles, target_pos, target_ori, epsilon, right=True)

def left_arm_set_position(angles, target_pos, target_ori, epsilon = 0.0001):
    return calc_inv_pos(angles, target_pos, target_ori, epsilon, right=False)


