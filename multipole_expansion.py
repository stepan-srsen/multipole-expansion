import numpy as np
from scipy.special import sph_harm

def xyz2spherical(xyz, elevation=False):
    '''Transforms xyz coords to spherical coords (radius, inclination, azimuth).'''
    
    xyz = np.asarray(xyz)
    ndim = len(xyz.shape)
    xyz = np.atleast_2d(xyz)
    
    scoords = np.empty_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    scoords[:,0] = np.sqrt(xy + xyz[:,2]**2) # radius
    if elevation:
        scoords[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # elevation angle defined from XY-plane up
    else:
        scoords[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # inclination angle defined from Z-axis down
    scoords[:,2] = np.arctan2(xyz[:,1], xyz[:,0]) # azimuth angle
    if ndim==1:
        scoords = np.squeeze(scoords)
    return scoords

def spherical2xyz(scoords):
    '''Transforms spherical coords (radius, inclination, azimuth) to xyz coords.'''
    
    scoords = np.asarray(scoords)
    ndim = len(scoords.shape)
    scoords = np.atleast_2d(scoords)
    xyz = np.empty_like(scoords)
    xyz[:,0] = scoords[:,0]*np.sin(scoords[:,1])*np.cos(scoords[:,2])
    xyz[:,1] = scoords[:,0]*np.sin(scoords[:,1])*np.sin(scoords[:,2])
    xyz[:,2] = scoords[:,0]*np.cos(scoords[:,1])
    if ndim==1:
        xyz = np.squeeze(xyz)
    return xyz

def el_pot(xyz, charges, coords, au=False, power=-1):
    '''Returns electric potential (a.u.) given by point charges "charges" at coords "coords" for point "xyz".'''
    
    # "au" turns on calculation in atomic units
    ke = 8.9875517923e9 # Cooulomb constant
    coords2 = coords-xyz
    rs = np.sqrt(np.sum(coords2**2, axis=1))
    field = np.sum(charges*(rs**power))
    if not au:
        field *= ke
    return field

class SphericalMultipoleExpansion():
    '''Performs a spherical multipole expansion for given point charges.'''
    
    # Inspired by Zangwill 2013 Modern Electrodynamics

    def __init__(self, charges, coords, l_max, interior=False, origin=None, au=False, verbose=0):
        
        # add some checks on dimensions etc.
        self.charges = charges
        self.coords = coords
        self.l_max = l_max
        self.interior = interior
        if origin is None:
            origin = np.sum(coords.T*np.abs(charges), axis=1)/np.sum(np.abs(charges)) # charge center
        self.origin = origin
        self.au = au # atomic units switch
        self.verbose = verbose
        self.calc_moments()
    
    def calc_coeff(self, l, m):
        '''Calculates the complex moment for specific l and m.'''
        
        Q = 0.0
        for i, xyz in enumerate(self.coords):
            q = self.charges[i]
            r, theta, phi = xyz2spherical(xyz-self.origin)
            if r < self.r_min:
                self.r_min = r
            if r > self.r_max:
                self.r_max = r
            # phi and theta are defined the opposite way in scipy
            # sph. harmonics in scipy include the Condon-Shortley phase (-1)^m, TODO: check vs Zangwill - seems OK
            Y = sph_harm(m, l, phi, theta)
            if self.interior:
                Q += q / r ** (l + 1) * np.conj(Y)
            else:
                Q += q * r ** l * np.conj(Y)
        Q *= 4*np.pi/(2*l+1) # normalization: 4*np.pi/(2*l+1) or sqrt(4*np.pi/(2*l+1)) or 1 -> changes __call__ function
        return Q
        
    def calc_moments(self):
        '''Calculates both complex and real multipoles and the power spectrum.'''
        
        self.r_min, self.r_max = np.inf, 0.0
        moments = {} # use arrays instead?
        rmoments = {}
        pspectrum = []
        for l in range(0, self.l_max + 1):
            pspectrum.append(0.0)
            for m in range(0, l + 1):
                moments[(l, m)] = self.calc_coeff(l, m)
                pspectrum[-1] += moments[(l, m)]*np.conj(moments[(l, m)])
                if m > 0:
                    moments[(l, -m)] = (-1)**m * np.conj(moments[(l, m)]) # use symmetry to calc m=-x from m=x
                    pspectrum[-1] += moments[(l, -m)]*np.conj(moments[(l, -m)])
                    rmoments[(l, m)] = np.sqrt(2) * (-1)**m * moments[(l, m)].real
                    rmoments[(l, -m)] = np.sqrt(2) * (-1)**m * moments[(l, m)].imag
                elif m == 0:
                    rmoments[(l, m)] = moments[(l, m)].real
        self.moments = moments
        self.rmoments = rmoments
        self.pspectrum = np.real(pspectrum)
        # return moments

    def __call__(self, xyz, l_max=None):
        '''Evaluate the multipole expansion at xyz coordinates possibly limiting max. ang. momentum.'''

        # now there is potential and rpotential for validation but only one is needed
        if l_max is None:
            l_max = self.l_max
        elif l_max > self.l_max:
            raise ValueError('Requested l_max({}) is larger than the precalculated l_max({}).'.format(l_max, self.l_max))
            
        xyzs = np.atleast_2d(xyz)
        potentials = []
        rpotentials = []
        for xyz in xyzs:        
            r, theta, phi = xyz2spherical(xyz-self.origin)
            if self.interior and r > self.r_min:
                print('WARNING: assumptions for interior expansion violated: r={} > r_min={}!'.format(r, self.r_min))
            elif (not self.interior) and r < self.r_max:
                print('WARNING: assumptions for exterior expansion violated: r={} < r_max={}!'.format(r, self.r_max))
            potential = 0.0
            rpotential = 0.0
            
            for l in range(l_max + 1):
                if self.interior:
                    rscale = r**(l)
                else:
                    rscale = 1 / r**(l+1)
                for m in range(0, l + 1):
                    Y_lm = sph_harm(m, l, phi, theta)
                    q_lm = self.moments[(l, m)]
                    r_q_lm = self.rmoments[(l, m)]
                    potential += q_lm * Y_lm * rscale
                    if self.verbose:
                        print('l =', l, ', m =', m, 'complex contrib.', q_lm * Y_lm * rscale, '=', q_lm, '*', Y_lm, '*', rscale)
                    if m==0:
                        r_Y_lm = Y_lm.real
                        rpotential += r_q_lm * r_Y_lm * rscale
                        if self.verbose:
                            print('l =', l, ', m =', m, 'real contrib.', r_q_lm * r_Y_lm * rscale, '=', r_q_lm, '*', r_Y_lm, '*', rscale)
                    else:
                        Y_lm2 = (-1)**m * np.conj(Y_lm)
                        q_lm2 = self.moments[(l, -m)]
                        potential += q_lm2 * Y_lm2 * rscale
                        if self.verbose:
                            print('l =', l, ', m =', -m, 'complex contrib.', q_lm2 * Y_lm2 * rscale, '=', q_lm2, '*', Y_lm2, '*', rscale)
                        r_Y_lm = np.sqrt(2) * (-1)**m * Y_lm.real
                        r_Y_lm2 = np.sqrt(2) * (-1)**m * Y_lm.imag
                        r_q_lm2 = self.rmoments[(l, -m)]
                        rpotential += r_q_lm * r_Y_lm * rscale
                        rpotential += r_q_lm2 * r_Y_lm2 * rscale
                        if self.verbose:
                            print('l =', l, ', m =', m, 'real contrib.', r_q_lm * r_Y_lm * rscale, '=',  r_q_lm, '*', r_Y_lm, '*', rscale)
                            print('l =', l, ', m =', -m, 'real contrib.', r_q_lm2 * r_Y_lm2 * rscale, '=', r_q_lm2, '*', r_Y_lm2, '*', rscale)
            potentials.append(potential) # .real
            rpotentials.append(rpotential)
        potentials = np.real(np.squeeze(potentials))
        rpotentials = np.real(np.squeeze(rpotentials))
        if not self.au:
            ke = 8.9875517923e9
            potentials *= ke
            rpotentials *= ke
        return potentials
