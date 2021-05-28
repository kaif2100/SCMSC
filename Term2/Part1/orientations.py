import numpy as np
import constants
import quaternion as qt

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(w0 = 0.005)
def InertialOrientation(t):
    """
    Input:
    t - time
    
    Return:
    np.quaternion A(t), np.ndarray w(t) and np.ndarray e(t)
    where A(t), w(t), e(t) - desired orientation of sc with respect to inertial frame in inertial frame(!)
    """
    # just randomly created
    A = np.quaternion(np.sin(InertialOrientation.w0 * t / 2.), 0, 0, np.cos(InertialOrientation.w0 * t / 2.))
    Ad = InertialOrientation.w0 / 2. * np.quaternion(np.cos(InertialOrientation.w0 * t / 2.), 0, 0, -np.sin(InertialOrientation.w0 * t / 2.))
    Add = InertialOrientation.w0 ** 2  / 4. * np.quaternion(-np.sin(InertialOrientation.w0 * t / 2.), 0, 0, -np.cos(InertialOrientation.w0 * t / 2.))
    Aconj = A.conj()
    w = 2 * (Ad * Aconj).vec
    e = 2 * (Add * Aconj + Ad * Ad.conj()).vec
    return A, w, e


def OrbitalQuaternion(r, V):
    """
    Input:
    r, V - radius vector and velocity of sc mass center in inertial frame
    
    Returns:
    np.quaternion that determines orbital frame orientation with respect to inertial frame in inertial frame
    
    P.S. to convert a quternion M_i from inertial basis to bounded basis M_e one needs to have
    a quaternion determines orientation of bounded basis relative to intertial basis
    then M_e = A.conj() * M_i * A; A can be in either inertial and bounded basis
    """    
    o1 = r / np.linalg.norm(r)
    o3 = np.cross(r, V)
    o3 /= np.linalg.norm(o3)
    o2 = np.cross(o3, o1)
    
    xi = np.array([1, 0, 0])
    yi = np.array([0, 1, 0])
    zi = np.array([0, 0, 1])
    
    buf1 = [np.dot(o1, xi), np.dot(o1, yi), np.dot(o1, zi)]
    buf2 = [np.dot(o2, xi), np.dot(o2, yi), np.dot(o2, zi)]
    buf3 = [np.dot(o3, xi), np.dot(o3, yi), np.dot(o3, zi)]
    B = np.vstack((buf1, buf2, buf3)).T
    A = qt.from_rotation_matrix(B)

    return A

C = np.quaternion(0, 0, 0, 1)
C /= C.abs()
@static_vars(C = C)
def OrbitalOrientation(r, V, t = None):
    """
    Input:
    r, V - radius vector and velocity of sc mass center in inertial frame
    t - time
    
    Orientation of sc regarding to the orbital system is supposed to be constant with quaternion C in orbital frame
    
    Returns:
    np.quaternion A - defines desired orientation of sc regarding to the inertial frame in inertial frame(!)
    w - desired angular velocity of sc in inertial frame(!)
    e - desired angular acceleration of sc in inertial frame (zero)
    """
    B = OrbitalQuaternion(r, V) # in inertial
    A = B * OrbitalOrientation.C # in inertial
    
    wo = np.cross(r, V) / (np.linalg.norm(r)**2) # angular speed of orbital frame
    Bd = 0.5 * np.quaternion(*wo) * B
    Ad = Bd * OrbitalOrientation.C
    w = 2 * (Ad * A.conj()).vec
    
    Bdd = 0.5 * np.quaternion(*wo) * Bd
    Add = Bdd * OrbitalOrientation.C
    e = 2 * (Add * A.conj() + Ad * Ad.conj()).vec
    return A, w, e


T = 24 * 60 * 60
wE = 2 * np.pi / T # Earths angular speed
t0 = 0 # time when the object had y = 0 in ECI
lattitude = 45 / 180 * np.pi # lattitude of the object
h = 100 # object's height above the sea level
@static_vars(wE = wE,t0 = t0, fi = lattitude, p = constants.RE + h, z = (constants.RE + h) * np.sin(lattitude))
def _getTarget(t):
    a = wE * (t - _getTarget.t0)
    x = _getTarget.p * np.cos(_getTarget.fi) * np.cos(a)
    y = _getTarget.p * np.cos(_getTarget.fi) * np.sin(a)
    r = np.array([x, y, _getTarget.z])
    we = np.array([0, 0, wE])
    V = np.cross(we, r)
    e = np.cross(we, V)
    return r, V, e

def _cosMatrix(o1, o2, o3):
    B = np.vstack((o1, o2, o3)).T
    return B

def _f(r, mu = constants.muE):
    return -mu * r / np.linalg.norm(r)**3

w_prev = None
t_prev = None
@static_vars(w_prev = w_prev, t_prev = t_prev)
def DZZOrientation(r, V, t):
    """
    Input:
    r, V - radius vector and velocity of sc mass center in inertial frame
    t - time
    
    This function defines oirentation with following parameters:
    e1 is always towards definite point on Earth, e3 is close to orbital momentum as possible
    
    Returns:
    np.quaternion A - defines desired orientation of sc regarding to the inertial frame in inertial frame(!)
    w - desired angular velocity of sc in inertial frame(!)
    e - desired angular acceleration of sc in inertial frame (zero)
    """
    rt, Vt, et = _getTarget(t)
    p = rt - r # vector from sc to a target on Earth in inertial
    c = np.cross(r, V)
    
    cn = np.linalg.norm(c)
    pn = np.linalg.norm(p)
    
    e1 = p / pn
    buf = c / cn
    e2 = np.cross(buf, e1)
    
    buffer2 = np.linalg.norm(e2)
    if np.allclose(e2, 0):
        warnings.warn("[p, c] is almost zero!")
    
    e2 /= buffer2
    e3 = np.cross(e1, e2)
    
    B = _cosMatrix(e1, e2, e3) # cosine Matrix (e1, e2, e3)
    A = qt.from_rotation_matrix(B) # this function takes B = (e1, e2, e3)
    
    pd = Vt - V
    e1d = (pd - e1 * np.dot(pd, e1)) / pn

    cd = np.cross(r, _f(r))
    bufd = (cd - buf * np.dot(cd, buf)) / cn
    bufe1d = np.cross(bufd, e1) + np.cross(buf, e1d)
    e2d = (bufe1d - e2 * np.dot(bufe1d, e2)) / buffer2

    e3d = np.cross(e1d, e2) + np.cross(e1, e2d)

    Bd = _cosMatrix(e1d, e2d, e3d) # ri = B re
    wix = Bd @ B.T
    wi = np.array([wix[2, 1], wix[0, 2], wix[1, 0]])    
    
    ei = np.array([0., 0., 0.])
    if DZZOrientation.t_prev and DZZOrientation.t_prev != t:
        dt = t - DZZOrientation.t_prev
        dw = wi - DZZOrientation.w_prev
        ei = dw / dt
    DZZOrientation.t_prev = t
    DZZOrientation.w_prev = wi.copy()    
    
    return A, wi, ei