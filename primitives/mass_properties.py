import numpy as np

def mass_properties(cone,rho=1):
    """ 
    Calculates the mass properties of a cone.
    
    Args:
        cone (Cone): The cone object.
        rho (float): The density of the cone material.

    Returns:
        tuple: (mass, inertia matrix)
    """

    # Radius of the base
    r = np.tan(cone.beta) * cone.H

    # Volume
    V = (1/3) * (np.pi * (r**2) * cone.H)

    # Mass
    m = V * rho

    # Inertia
    Iyy = m * ((3 / 20) * r**2 + (3 / 80) * cone.H**2)
    Izz = Iyy
    Ixx = 0.3 * m * r**2

    inertia = np.diag([Ixx,Iyy,Izz])

    return m, inertia