# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:45:53 2020

@author: thoma
"""


import numpy as np
import matplotlib.pyplot as plt

#input game temperature
T = int(input('Input game temperature in Fahrenheit at Coors: '))
print('')

#air density function
def rho(y, T0):
    a = 6.5E-3
    rho0 = 1.079    #average humidity air density
    alpha = 2.7
    return rho0*(1 - (a*y)/(T0))**alpha

#initial values for calculating air density 
y = 5200*0.3048     #converting Coors field's elevation to meters
T0 = (T - 32)*5/9 + 273.15
print('Coors Field is {:4.2f} m above sea level'.format(y))
print('The air density at Coors Field is roughly {:4.4f} kg/m^3'.format(rho(y, T0)))


#importing data
hdata = np.loadtxt('kyle home.txt')
rdata = np.loadtxt('kyle road.txt')

#home data for average horizontal/vertical break on a fastball per game
xh = hdata[:,0]
yh = hdata[:,1]

#road data for average horizontal/vertical break on a fastball per game
xr = rdata[:,0]
yr = rdata[:,1]

#plotting graph
fig = plt.figure()
plt.title('Horizontal Break (in) vs. Vertical Break (in)')
plt.plot(xh, yh, '.', color = 'purple', label = 'Coors')
plt.plot(xr, yr, '.', color = 'orange', label = 'Road')
plt.xlabel('Horizontal break (in)')
plt.ylabel('Vertical break (in)')
plt.legend()
plt.show()

print("Freeland's average horizontal break of a fastball at Coors is {:4.1f} inches".format(np.mean(xh)))
print("Freeland's average vertical break of a fastball at Coors is {:4.1f} inches".format(np.mean(yh)))
print("Freeland's average horizontal break of a fastball on the road is {:4.1f} inches".format(np.mean(xr)))
print("Freeland's average vertical break of a fastball on the road is {:4.1f} inches".format(np.mean(yr)))


from mpl_toolkits.mplot3d import Axes3D

#define the drag coefficient function
def k_D(v):
    delta = 5.0
    vd = 35.0
    return 0.0039 + 0.0058/(1 + np.exp((v-vd)/delta))

def k_L(v, omega):
    A = 0.00442     #cross-sectional area of baseball
    m = 0.145       #mass of baseball in kg
    R = 0.075*0.5
    return (omega*R*A)/(2*m*v*60.0*2*np.pi)

#define euler algorithm
def euler(vx,vy,vz, phi):
    #initial conditions
    x = 0
    y = 0
    z = 1.8  #release height in meters
    t = 0
    h = 0.001  #time step
    rho = 0.976     #value calculated from air density function
    rho0 = 1.225    #dry air density
    g = 9.81
    
    #creating arrays
    global X,Y,Z
    X = np.zeros(0)
    Y = np.zeros(0)
    Z = np.zeros(0)

    
    while x <= 18.44:  #distance to home base from pitcher's mound
        X = np.append(X, x)
        Y = np.append(Y, y)
        Z = np.append(Z, z)
        
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        #calculate acceleration components
        ax = -k_D(v)*(rho/rho0)*v*vx + k_L(v, omega)*rho*(vz*omega*np.sin(phi) - vy*omega*np.cos(phi))
        ay = -k_D(v)*(rho/rho0)*v*vy + k_L(v, omega)*rho*(vx*omega*np.cos(phi))
        az = -k_D(v)*(rho/rho0)*v*vz - k_L(v, omega)*rho*vx*omega*np.sin(phi) - g

        #apply Euler algorithm
        vx = vx + ax*h
        vy = vy + ay*h
        vz = vz + az*h
        x = x + vx*h
        y = y + vy*h
        z = z + vz*h
        t = t + h

def euler2(vx,vy,vz, phi):
    #initial conditions
    x = 0
    y = 0
    z = 1.8  #release height in meters
    t = 0
    h = 0.001  #time step
    rho = 1.225
    g = 9.81
    
    global X1,Y1,Z1
    X1 = np.zeros(0)
    Y1 = np.zeros(0)
    Z1 = np.zeros(0)
    
    while x <= 18.44:  #distance to home base from pitcher's mound
        X1 = np.append(X1, x)
        Y1 = np.append(Y1, y)
        Z1 = np.append(Z1, z)
        
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        #calculate acceleration components
        ax = -k_D(v)*v*vx + k_L(v, omega)*rho*(vz*omega*np.sin(phi) - vy*omega*np.cos(phi))
        ay = -k_D(v)*v*vy + k_L(v, omega)*rho*(vx*omega*np.cos(phi))
        az = -k_D(v)*v*vz - k_L(v, omega)*rho*vx*omega*np.sin(phi) - g

        #apply Euler algorithm
        vx = vx + ax*h
        vy = vy + ay*h
        vz = vz + az*h
        x = x + vx*h
        y = y + vy*h
        z = z + vz*h
        t = t + h        
        
        
TYPE = str(input("Type of pitch:  Fastball(f), Curveball(c), Slider(s), Changeup(ch) "))
if TYPE == 'c' or TYPE == 'C':
    v = 35.76   #initial velocity
    phi = 135*np.pi/180.0   #spin axis
    omega = 2444/60.0*2*np.pi   #spin rate
    theta = 1*np.pi/180.0  # angle from horizontal

    #velocities in each direction
    vx = v*np.cos(theta)
    vy = 0*v*np.sin(theta)
    vz = v*np.sin(theta)
    
    euler(vx,vy,vz, phi)
    euler2(vx,vy,vz,phi)
    
    #calculating horizontal and vertical break
    print('Horizontal break = {:4.1f} inches'.format((Y1[-1] - Y[-1])*39.37))
    print('Vertical break = {:4.1f} inches'.format((Z1[-1] - Z[-1])*39.37))
    
if TYPE == 's' or TYPE == 'S':
    v = 38.45
    phi = 180*np.pi/180
    omega = 2473.0/60.0*2*np.pi
    theta = 0*np.pi/180.0  # angle from horizontal

    vx = v*np.cos(theta)
    vy = 0*v*np.sin(theta)
    vz = v*np.sin(theta)
    
    euler(vx,vy,vz, phi)
    euler2(vx,vy,vz,phi)
    print('Horizontal break = {:4.1f} inches'.format((Y1[-1] - Y[-1])*39.37))
    print('Vertical break = {:4.1f} inches'.format((Z1[-1] - Z[-1])*39.37))

if TYPE =='f' or TYPE == 'F':
    v = 41.13
    phi = 315.0*np.pi/180
    omega = 2451/60.0*2*np.pi
    theta = -0*np.pi/180.0  # angle from horizontal

    vx = v*np.cos(theta)
    vy = 0*v*np.sin(theta)
    vz = v*np.sin(theta)
    
    euler(vx,vy,vz, phi)
    euler2(vx,vy,vz,phi)
    print('Horizontal break = {:4.1f} inches'.format((Y1[-1] - Y[-1])*39.37))
    print('Vertical break = {:4.1f} inches'.format((Z1[-1] - Z[-1])*39.37))    

if TYPE == 'ch' or TYPE == 'CH':
    v = 38.45
    phi = 0*np.pi/180
    omega = 1412.0/60.0*2*np.pi
    theta = -0*np.pi/180.0  # angle from horizontal

    vx = v*np.cos(theta)
    vy = 0*v*np.sin(theta)
    vz = v*np.sin(theta)
    
    euler(vx,vy,vz, phi)   
    euler2(vx,vy,vz,phi)
    print('Horizontal break = {:4.1f} inches'.format((Y1[-1] - Y[-1])*39.37))
    print('Vertical break = {:4.1f} inches'.format((Z1[-1] - Z[-1])*39.37))
    
#plotting simulation
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.plot(X,Y, zs = Z, zdir = 'z', color = 'purple', label = 'Coors')
ax.plot(X1, Y1, zs = Z1, zdir = 'z', color = 'orange', label = 'Other')
plt.show()

print('')
from scipy import stats

homedata = np.loadtxt('nolan home.txt')
roaddata = np.loadtxt('nolan road.txt')

#home run distance, exit velocity, and launch angle respectively
Hdis = homedata[:,0]
Hev = homedata[:,1]
Hla = homedata[:,2]

Rdis = roaddata[:,0]
Rev = roaddata[:,1]
Rla = roaddata[:,2]
print("Arenado's average home run distance at Coors is {:4.1f} ft".format(np.mean(Hdis)))
print("Arenado's average home run distance at other ballparks is {:4.1f} ft".format(np.mean(Rdis)))
print("Arenado's average exit velocity at Coors is {:4.1f} mph".format(np.mean(Hev)))
print("Arenado's average exit velocity at other ballparks is {:4.1f} mph".format(np.mean(Rev)))
print("Arenado's average launch angle at Coors is {:4.1f} degrees".format(np.mean(Hla)))
print("Arenado's average launch angle at other ballparks is {:4.1f} degrees".format(np.mean(Rla)))
print('Arenado has hit {:4.0f} home runs at Coors'.format(len(Hdis)))
print('Arenado has hit {:4.0f} home runs at other ballparks'.format(len(Rdis)))

#linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(Hev, Hdis)
A = intercept
B = slope
x = np.linspace(min(Hev), max(Hev))
y = B*x + A
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(Rev, Rdis)
D = intercept2
C = slope2
x2 = np.linspace(min(Rev), max(Rev))
y2 = C*x2 + D

print('')
print('Home slope = ', B)
print('Home intercept = ', A)
print('Road slope =', C)
print('Road intercept =', D)
print('')

#plotting data and linear regression
fig = plt.figure()
plt.title('Nolan Arenado: Exit Velocity vs. HR Distance')
plt.plot(Hev, Hdis, '.', color = 'purple', label = 'Home Runs at Coors')
plt.plot(Rev, Rdis, '.', color = 'orange', label = 'Home Runs on the road')
plt.plot(x, y, color = 'purple')
plt.plot(x2, y2, color = 'orange')
plt.xlabel('Exit velocity (mph)')
plt.ylabel('HR Distance (ft)')
plt.legend()

#calculating difference between means of home and away distance
diff = np.mean(Hdis) - np.mean(Rdis)
print('There is a {:4.2f} ft difference between the average distance of a home run hit at Coors and at other ballparks for Arenado'.format(diff))

#new euler functions for home runs
def eulerhr(vx,vy,vz, phi):
    #initial conditions
    x = 0
    y = 0
    z = 0.6  #release height in meters
    t = 0
    h = 0.001  #time step
    rho = 0.976
    rho0 = 1.225
#    omega = 1800.0/60.0*2*pi
    g = 9.81
    global X,Y,Z
    X = np.zeros(0)
    Y = np.zeros(0)
    Z = np.zeros(0)

    
    while z>= 0:  #distance to home base from pitcher's mound
        X = np.append(X, x)
        Y = np.append(Y, y)
        Z = np.append(Z, z)
        
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        #calculate acceleration components
        ax = -k_D(v)*(rho/rho0)*v*vx + k_L(v, omega)*rho*(vz*omega*np.sin(phi) - vy*omega*np.cos(phi))
        ay = -k_D(v)*(rho/rho0)*v*vy + k_L(v, omega)*rho*(vx*omega*np.cos(phi))
        az = -k_D(v)*(rho/rho0)*v*vz - k_L(v, omega)*rho*vx*omega*np.sin(phi) - g

        #apply Euler algorithm
        vx = vx + ax*h
        vy = vy + ay*h
        vz = vz + az*h
        x = x + vx*h
        y = y + vy*h
        z = z + vz*h
        t = t + h

def eulerhr2(vx,vy,vz, phi):
    #initial conditions
    x = 0
    y = 0
    z = 0.6  #release height in meters
    t = 0
    h = 0.001  #time step
    rho = 1.225
    
#    omega = 1800.0/60.0*2*pi
    g = 9.81
    global X1,Y1,Z1
    X1 = np.zeros(0)
    Y1 = np.zeros(0)
    Z1 = np.zeros(0)

    
    while z >= 0:  #distance to home base from pitcher's mound
        X1 = np.append(X1, x)
        Y1 = np.append(Y1, y)
        Z1 = np.append(Z1, z)
        
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        #calculate acceleration components
        ax = -k_D(v)*v*vx + k_L(v, omega)*rho*(vz*omega*np.sin(phi) - vy*omega*np.cos(phi))
        ay = -k_D(v)*v*vy + k_L(v, omega)*rho*(vx*omega*np.cos(phi))
        az = -k_D(v)*v*vz - k_L(v, omega)*rho*vx*omega*np.sin(phi) - g

        #apply Euler algorithm
        vx = vx + ax*h
        vy = vy + ay*h
        vz = vz + az*h
        x = x + vx*h
        y = y + vy*h
        z = z + vz*h
        t = t + h        
v = 45.662564530973455135
phi = 270.0*np.pi/180
omega = 2337/60.0*2*np.pi
   
theta = 29*np.pi/180.0  # angle from horizontal

vx = v*np.cos(theta)
vy = v*np.sin(theta)
vz = v*np.sin(theta)

eulerhr(vx,vy,vz, phi)
eulerhr2(vx,vy,vz, phi)

#calculating distance traveled of ball according to euler function
Hdis = np.sqrt(Y[-1]**2 + X[-1]**2)*3.28084
Rdis = np.sqrt(Y1[-1]**2 + X1[-1]**2)*3.28084


print('')
print('Average home run distance at Coors is {:4.4} ft'.format(Hdis))
print('Average home run distance at other ballparks is {:4.4} ft'.format(Rdis))
print('The added distance at Coors is {:4.4} ft'.format(Hdis - Rdis))

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim3d(0,120)
ax.set_ylim3d(0,120)
ax.set_zlim3d(0,120)
ax.plot(X,Y, zs = Z, zdir = 'z', color = 'purple', label = 'Coors')
ax.plot(X1, Y1, zs = Z1, zdir = 'z', color = 'orange', label = 'Other')
plt.show()








