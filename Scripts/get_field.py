"""
load_field function 

- just loads in Quijote field with pylians gadget reader 
- then adds optional RSD and interpolates with CIC. (this is the periodic version with no mask)

Using _pylians version is slightly faster as it cythonised which is quicker than NUMBA in this case.(i mean it just better writtern bruh)
"""
import sys
sys.path.append('/home/addis/Pylians3/library/build/lib.linux-x86_64-3.8')
import numpy as np
import Pk_library as PKL
from numba import jit
import readgadget
import MAS_library as MASL
import redshift_space_library as RSL

def load_field_pylians(path,file,N_side,rsd = 'no',obs_pos=[0,0,0],verbose=True):
    # input files   # so we enter the Quijote folder and access which initialization
    snapshot = path + file  #10000/snapdir_004/snap_004
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    Nall     = header.nall         #Total number of particles
    Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    Omega_m  = header.omega_m      #value of Omega_m
    Omega_l  = header.omega_l      #value of Omega_l
    h        = header.hubble       #value of h
    redshift = header.redshift     #redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)*(1/(1+redshift))#Value of (conformal) H(z) in km/s/(Mpc/h)

    # read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0

    #this is for adding a artificial RSD
    #make periodic grid again
    def make_periodic(x,L):
        tmp = np.where((x>L)|(x<0.0))
        x[tmp] = (x[tmp]+L)%L
        return x
   
    @jit(nopython=True)# CHECK speed up...
    def add_RSD(pos,vel,obs_pos):
        #this is for adding a artificial RSD
        #make periodic grid again
        def make_periodic(x,L):
            tmp = np.where((x>L)|(x<0.0))
            x[tmp] = (x[tmp]+L)%L
            return x
        
        pos1 = np.zeros_like(pos) #these will be the the position of particles from observer
        #need to normalise
        pos1[0] = pos[0] - obs_pos[0];pos1[1] = pos[1] - obs_pos[1];pos1[2] = pos[2] - obs_pos[2]
        conf_norm = np.sqrt(pos1[0]**2 + pos1[1]**2 + pos1[2]**2) # make a unit vector - normalise
        #avoid zero errors:
        conf_norm = np.where(conf_norm==0,1,conf_norm) # where conf_norm is 0, - so is x,y,z!!!
        x_hat= pos1[0]/conf_norm;y_hat= pos1[1]/conf_norm;z_hat= pos1[2]/conf_norm

        v_dot_x = vel[0]*x_hat + vel[1]*y_hat + vel[2]*z_hat  #get v . x hat
        z_rsd = make_periodic(pos[2] + (v_dot_x*z_hat/(Hubble)),BoxSize)#*h    *a is included, this conformal
        y_rsd = make_periodic(pos[1] + (v_dot_x*y_hat/(Hubble)),BoxSize)#*h
        x_rsd = make_periodic(pos[0] + (v_dot_x*x_hat/(Hubble)),BoxSize)#*h
        return x_rsd,y_rsd,z_rsd
    
    if rsd != 'no':#if RSD are added or not!!!
        x,y,z = pos.T      # get positions in Mpc/h
        vx,vy,vz = vel.T   # get velocities in km/s
        x,y,z = add_RSD(pos.T,vel.T,np.array(obs_pos))
        pos = np.array([x,y,z],dtype=np.float32).T
    
    # density field parameters
    grid    = N_side    #the 3D field will have grid x grid x grid voxels
    BoxSize = 1000.0 #Mpc/h ; size of box
    MAS     = 'CIC'  #mass-assigment scheme
    #verbose = True   #print information on progress

    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    return delta


def load_field_pylians_PP(path,file,N_side,rsd = 'no',obs_pos=[0,0,0],verbose=True):
    # input files   # so we enter the Quijote folder and access which initialization
    snapshot = path + file  #10000/snapdir_004/snap_004
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    Nall     = header.nall         #Total number of particles
    Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    Omega_m  = header.omega_m      #value of Omega_m
    Omega_l  = header.omega_l      #value of Omega_l
    h        = header.hubble       #value of h
    redshift = header.redshift     #redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#*(1/(1+redshift))#Value of  H(z) in km/s/(Mpc/h)

    # read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
    
    axis=2
    RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)
    
    # density field parameters
    grid    = N_side    #the 3D field will have grid x grid x grid voxels
    BoxSize = 1000.0 #Mpc/h ; size of box
    MAS     = 'CIC'  #mass-assigment scheme
    #verbose = True   #print information on progress

    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    return delta

def load_field_pylians_pp1(path,file,N_side,rsd = 'no',verbose=True): #plane-parallel limit version
    # input files   # so we enter the Quijote folder and access which initialization
    snapshot = path + file  #10000/snapdir_004/snap_004
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    Nall     = header.nall         #Total number of particles
    Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    Omega_m  = header.omega_m      #value of Omega_m
    Omega_l  = header.omega_l      #value of Omega_l
    h        = header.hubble       #value of h
    redshift = header.redshift     #redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

    # read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0

    x,y,z = pos.T   # get positions in Mpc/h
    vx,vy,vz = vel.T   # get velocities in km/s

    #this is for adding a artificial RSD
    #make periodic grid again
    def make_periodic(x,L):
        tmp = np.where((x>L)|(x<0.0)) #or
        x[tmp] = (x[tmp]+L)%L
        return x

    def add_RSD(x,vx): #adds RSD in one direction in PP limit
        return make_periodic(x + (vx/H))

    if rsd == 'z':
        z = make_periodic(z + (vz/(Hubble)),BoxSize)
    if rsd == 'y':
        y = make_periodic(y + (vy/(Hubble)),BoxSize)
    if rsd == 'x':
        x = make_periodic(x + (vx/(Hubble)),BoxSize)
    
    pos = np.array([x,y,z]).T
    
    # density field parameters
    grid    = N_side    #the 3D field will have grid x grid x grid voxels
    BoxSize = 1000.0 #Mpc/h ; size of box
    MAS     = 'CIC'  #mass-assigment scheme
    #verbose = True   #print information on progress

    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    return delta

def just_load(path,file):    #useful for adding RSD manually    #can then just cic
    # input files   # so we enter the Quijote folder and access which initialization
    snapshot = path + file  #10000/snapdir_004/snap_004
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    Nall     = header.nall         #Total number of particles
    Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    Omega_m  = header.omega_m      #value of Omega_m
    Omega_l  = header.omega_l      #value of Omega_l
    h        = header.hubble       #value of h
    redshift = header.redshift     #redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

    # read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0

    return pos.T,vel.T      # get positions in Mpc/h,  # get velocities in km/s
    
    
    
def load_field(path,file,N_side,rsd = 'no'):
    # input files   # so we enter the Quijote folder and access which initialization
    snapshot = path + file  #10000/snapdir_004/snap_004
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    Nall     = header.nall         #Total number of particles
    Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    Omega_m  = header.omega_m      #value of Omega_m
    Omega_l  = header.omega_l      #value of Omega_l
    h        = header.hubble       #value of h
    redshift = header.redshift     #redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

    # read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0

    x,y,z = pos.T   # get positions in Mpc/h
    vx,vy,vz =vel.T   # get velocities in km/s

    #this is for adding a artificial RSD
    #make periodic grid again
    def make_periodic(x,L):
        tmp = np.where((x>L)|(x<0.0))
        x[tmp] = (x[tmp]+L)%L
        return x

    def add_RSD(x,vx): #adds RSD in one direction in PP limit
        return make_periodic(x + (vx/H))

    if rsd == 'z':
        z = make_periodic(z + (vz/(Hubble*h)),BoxSize)
    if rsd == 'y':
        y_rsd = make_periodic(y + (vy/(Hubble*h)),BoxSize)
    if rsd == 'x':
        x_rsd = make_periodic(x + (vx/(Hubble*h)),BoxSize)

    #periodic version 
    #lets write a cloud-in-cell interpolation scheme!!!
    # see http://background.uchicago.edu/~whu/courses/Ast321_11/pm.pdf - formalism for tx,dx etc
    #this has periodic bouncdary conditions for the box
    # this box is periodic - hmm - need a not periodic version
    @jit(nopython=True)
    def cic(x,y,z,weights,Nside,L):

        cell_density = np.zeros((Nside,Nside,Nside)) # create empty cell grid
        grid_spacing = L/Nside
        x_int = np.where(x/grid_spacing==Nside,0,x/grid_spacing)#;  del(x)
        y_int = np.where(y/grid_spacing==Nside,0,y/grid_spacing)#;  del(y)
        z_int = np.where(z/grid_spacing==Nside,0,z/grid_spacing)#;  del(z) #this gives x,y,z in scale with the integer bins
        #print(np.int64(x_int))
        #print(np.array(np.floor(x_int),dtype=np.int64)==np.int64(x_int))

        #get coordinates of cell point is paritally in 
        x_floor = x_int.astype(np.int64)#np.floor(x_int).astype(int)
        y_floor = y_int.astype(np.int64)#np.floor(y_int).astype(int)
        z_floor = z_int.astype(np.int64)#np.floor(z_int).astype(int) #this is all x,y,z index of delta 
        #print(x_int)
        #print(x_floor)
        #print(y_floor.max())
        """
        |   . |     |               boxes with particle (.)
           |.    |     |     x_floor (|) is the center of the box used to calculate tx,dx etc
        t     d    
        """
        dx, dy, dz = x_int-x_floor, y_int-y_floor, z_int-z_floor #; del(x_int,y_int,z_int) #portion of shape function in adjacent cell (i+1)
        tx, ty, tz = 1 - dx, 1 - dy, 1 - dz     #portion in current cell (i)

        for i in range(len(x)):
            cell_density[x_floor[i],y_floor[i],z_floor[i]] += weights[i]*tx[i]*ty[i]*tz[i] #linear interpolation of t and d for 3D
            cell_density[(x_floor[i]+1)%Nside,y_floor[i],z_floor[i]] += weights[i]*dx[i]*ty[i]*tz[i] #(x_floor[i]+1)%Nside implements periodic boundary condtions for the box
            cell_density[x_floor[i],(y_floor[i]+1)%Nside,z_floor[i]] += weights[i]*tx[i]*dy[i]*tz[i] 
            cell_density[x_floor[i],y_floor[i],(z_floor[i]+1)%Nside] += weights[i]*tx[i]*ty[i]*dz[i] 
            cell_density[(x_floor[i]+1)%Nside,(y_floor[i]+1)%Nside,z_floor[i]] += weights[i]*dx[i]*dy[i]*tz[i] 
            cell_density[(x_floor[i]+1)%Nside,y_floor[i],(z_floor[i]+1)%Nside] += weights[i]*dx[i]*ty[i]*dz[i] 
            cell_density[x_floor[i],(y_floor[i]+1)%Nside,(z_floor[i]+1)%Nside] += weights[i]*tx[i]*dy[i]*dz[i] 
            cell_density[(x_floor[i]+1)%Nside,(y_floor[i]+1)%Nside,(z_floor[i]+1)%Nside] += weights[i]*dx[i]*dy[i]*dz[i] 

        return cell_density

    # normal density field w/o RSD
    #delta = cic(x,y,z,np.ones_like(z),100,1000)
    #delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    #with rsd
    delta = cic(x,y,z,np.ones_like(z),N_side,BoxSize)
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
    return delta

#periodic version 
#lets write a cloud-in-cell interpolation scheme!!!
# see http://background.uchicago.edu/~whu/courses/Ast321_11/pm.pdf - formalism for tx,dx etc
#this has periodic bouncdary conditions for the box
@jit(nopython=True)
def cic(x,y,z,weights,Nside,L):

    cell_density = np.zeros((Nside,Nside,Nside)) # create empty cell grid
    grid_spacing = L/Nside
    x_int = np.where(x/grid_spacing==Nside,0,x/grid_spacing)#;  del(x)
    y_int = np.where(y/grid_spacing==Nside,0,y/grid_spacing)#;  del(y)
    z_int = np.where(z/grid_spacing==Nside,0,z/grid_spacing)#;  del(z) #this gives x,y,z in scale with the integer bins
    #print(np.int64(x_int))
    #print(np.array(np.floor(x_int),dtype=np.int64)==np.int64(x_int))

    #get coordinates of cell point is paritally in 
    x_floor = x_int.astype(np.int64)#np.floor(x_int).astype(int)
    y_floor = y_int.astype(np.int64)#np.floor(y_int).astype(int)
    z_floor = z_int.astype(np.int64)#np.floor(z_int).astype(int) #this is all x,y,z index of delta 
    #print(x_int)
    #print(x_floor)
    #print(y_floor.max())
    """
    |   . |     |               boxes with particle (.)
       |.    |     |     x_floor (|) is the center of the box used to calculate tx,dx etc
    t     d    
    """
    dx, dy, dz = x_int-x_floor, y_int-y_floor, z_int-z_floor #; del(x_int,y_int,z_int) #portion of shape function in adjacent cell (i+1)
    tx, ty, tz = 1 - dx, 1 - dy, 1 - dz     #portion in current cell (i)

    for i in range(len(x)):
        cell_density[x_floor[i],y_floor[i],z_floor[i]] += weights[i]*tx[i]*ty[i]*tz[i] #linear interpolation of t and d for 3D
        cell_density[(x_floor[i]+1)%Nside,y_floor[i],z_floor[i]] += weights[i]*dx[i]*ty[i]*tz[i] #(x_floor[i]+1)%Nside implements periodic boundary condtions for the box
        cell_density[x_floor[i],(y_floor[i]+1)%Nside,z_floor[i]] += weights[i]*tx[i]*dy[i]*tz[i] 
        cell_density[x_floor[i],y_floor[i],(z_floor[i]+1)%Nside] += weights[i]*tx[i]*ty[i]*dz[i] 
        cell_density[(x_floor[i]+1)%Nside,(y_floor[i]+1)%Nside,z_floor[i]] += weights[i]*dx[i]*dy[i]*tz[i] 
        cell_density[(x_floor[i]+1)%Nside,y_floor[i],(z_floor[i]+1)%Nside] += weights[i]*dx[i]*ty[i]*dz[i] 
        cell_density[x_floor[i],(y_floor[i]+1)%Nside,(z_floor[i]+1)%Nside] += weights[i]*tx[i]*dy[i]*dz[i] 
        cell_density[(x_floor[i]+1)%Nside,(y_floor[i]+1)%Nside,(z_floor[i]+1)%Nside] += weights[i]*dx[i]*dy[i]*dz[i] 

    return cell_density