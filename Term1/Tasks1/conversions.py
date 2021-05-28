import numpy as np
import constants

def Kepler2Cartesian(i, W, w, e, p, v, mu = constants.muE):
    # ALL ANGLES MUST BE IN RADIANS
    # i - inclination; W - longitude of the ascending node; w - argument of periapsis;
    # p - focal parameter; v - true anomaly; e - eccentricity
    # mu - standart gravitational parameter. Earths by default
    # Returns numpy.ndarray with shape (6,) that contains [x, y, z, Vx, Vy, Vz] in SI
    # Calculation are carried out according to Okhotsimsky "Basics of space flight mechanics" 1990, p. 101-102
    # To do: assess accuracy
    result = np.empty(6)
    r = p / (1 + e * np.cos(v))
    Vr = np.sqrt(mu / p) * e * np.sin(v)
    Vn = np.sqrt(mu / p) * (1 + e * np.cos(v))
    u = w + v
    rx = np.cos(u) * np.cos(W) - np.cos(i) * np.sin(u) * np.sin(W)
    ry = np.cos(u) * np.sin(W) + np.cos(i) * np.sin(u) * np.cos(W)
    rz = np.sin(i) * np.sin(u)
    nx = -np.sin(u) * np.cos(W) - np.cos(i) * np.cos(u) * np.sin(W)
    ny = -np.sin(u) * np.sin(W) + np.cos(i) * np.cos(u) * np.cos(W)
    nz = np.sin(i) * np.cos(u)
    
    result[0] = r * rx
    result[1] = r * ry
    result[2] = r * rz
    result[3] = Vr * rx + Vn * nx
    result[4] = Vr * ry + Vn * ny
    result[5] = Vr * rz + Vn * nz
    
    return result

def Cartesian2Kepler(x, y, z, Vx, Vy, Vz, t = 0, mu = constants.muE):
    # Everything have to be in SI
    # x, y, z, Vx, Vy, Vz - components of phase vector in INERTIONAL frame
    # t - time of measurment
    # mu - standart gravitational parameter. Earths by default
    # Returns np.ndarray of [i, W, w, e, p, epoch, v]. All angles in RADIANS
    # In case of zero inclination W assumed to be zero.
    # In case of zero eccentricity w assumed to be zero.
    # Calculation are carried out according to Okhotsimsky "Basics of space flight mechanics" 1990, p. 122-124
    # To do: assess accuracy
    R = np.array([x, y, z])
    r = R / np.linalg.norm(R)
    V = np.array([Vx, Vy, Vz])
    
    # Area integral
    c = np.cross(R, V)
    C = np.linalg.norm(c)
    c /= C
    
    # Energy integral
    h = np.linalg.norm(V)**2 - 2 * mu / np.linalg.norm(R)
    
    # Eccentricity
    e = np.sqrt(1 + h * C**2 / mu**2)
        
    # Inclination
    i = np.arccos(c[2])
    
    # Longitude of the ascending node W
    # If inclination is almost zero or pi we will assume W = 0
    if abs(np.sin(i)) > constants.max_to_zero:
        cosW = -c[1]/np.sin(i)
        
        if cosW > 1 and cosW <= 1 + constants.max_to_zero:
            cosW = 1
        elif cosW > 1:
            cosW = 1
            print("Error encountered while running: cos(W) is bigger then 1. It have been set zero.")
        
        sinW = c[0]/np.sin(i)
        buf = np.arccos(cosW)
        if sinW >= 0:
            W = buf
        else:
            W = np.pi * 2 - buf
    else:
        W = 0
    
    # Focal parameter
    p = C**2 / mu
    
    # True anomaly v
    if e > constants.max_to_zero:
        Vr = np.dot(V, r)
        Vn = np.linalg.norm(V - Vr * r)
        sinv = Vr / e * np.sqrt(p / mu)
        cosv = (Vn * np.sqrt(p / mu) - 1) / e
        
        if cosv > 1 and cosv <= 1 + constants.max_to_zero:
            cosv = 1
        elif cosv > 1:
            cosv = 1
            print("Error encountered while running: cos(v) is bigger then 1. It have been set zero.")
        
        if sinv >=0:
            v = np.arccos(cosv)
        else:
            v = 2 * np.pi - np.arccos(cosv)
    else:
        v = 0
        
    # Periapsis argument
    rW = np.array([np.cos(W), np.sin(W), 0])
    cosu = np.dot(rW, r)
    
    if cosu > 1 and cosu <= 1 + constants.max_to_zero:
        cosu = 1
    elif cosu > 1:
        cosu = 1
        print("Error encountered while running: cos(u) is bigger then 1. It have been set zero.")

    sinu = np.sign(r[2]) * np.linalg.norm(np.cross(rW, r))
    if sinu >=0:
        u = np.arccos(cosu)
    else:
        u = 2 * np.pi - np.arccos(cosu)
    w = u - v
    
    # If e = 0 we will assume w = 0 and v = u
    if e <= constants.max_to_zero:
        v = u
        w = 0
        
    # Epoch
    if e < 1 - constants.max_to_zero:
        E = 2 * np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(v / 2))
        a = p / (1 - e**2)
        epoch = t - np.sqrt(a**3 / mu) * (E - e * np.sin(E))
    elif e < 1 + constants.max_to_zero and e >= 1 - constants.max_to_zero:
        epoch = t - np.sqrt(p**3 / mu) * (np.tan(v/2) + np.tan(v/2)**3 / 3) / 2
    else:
        H = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(v / 2))
        a = p / (e**2 - 1)
        epoch = t - np.sqrt(a**3 / mu) * (e * np.sinh(H) - H)

    return np.array([i, W, w, e, p, epoch, v])