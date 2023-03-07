from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import allantools

class calc_allan:
    def __init__(self,bag_file):
        # my_bag = bagreader(bag_file)
        # self.my_topic = my_bag.message_by_topic('/vectornav')
        self.my_topic = bag_file
    
    def cvt_bag_data_gyro(self):
        self.a = pd.read_csv(self.my_topic,usecols=['header.stamp.secs','data'])
        gyros = pd.DataFrame(columns = ['gx', 'gy', 'gz','time']) 
        z = []
        x = []
        y = []
        for i in range(len(self.a)):
                x.append(str(self.a.iat[i,1]).split(',')[10])
                y.append(str(self.a.iat[i,1]).split(',')[11])
                z.append(str(self.a.iat[i,1]).replace('*',",").split(',')[12])
        # print(z)
        sz = pd.Series(z)
        sx = pd.Series(x)
        sy = pd.Series(y)
        gyros['time'] = self.a['header.stamp.secs'] - self.a['header.stamp.secs'][0]
        gyros['gz'] = sz
        gyros['gy'] = sy
        gyros['gx'] = sx
        gyros.to_csv('demo_gyro_D.csv')
    
    def cvt_bag_data_accel(self):
        self.a = pd.read_csv(self.my_topic,usecols=['header.stamp.secs','data'])
        gyros = pd.DataFrame(columns = ['gx', 'gy', 'gz','time']) 
        z = []
        x = []
        y = []
        for i in range(len(self.a)):
                x.append(str(self.a.iat[i,1]).split(',')[7])
                y.append(str(self.a.iat[i,1]).split(',')[8])
                z.append(str(self.a.iat[i,1]).replace('*',",").split(',')[9])
        # print(z)
        sz = pd.Series(z)
        sx = pd.Series(x)
        sy = pd.Series(y)
        gyros['time'] = self.a['header.stamp.secs'] - self.a['header.stamp.secs'][0]
        gyros['gz'] = sz
        gyros['gy'] = sy
        gyros['gx'] = sx
        gyros.to_csv('demo_accel_A.csv')

    def a_varinace(self,axis):
        a = pd.read_csv(self.my_topic,usecols=[str(axis)]).to_numpy()
        # print(len(a))
        t0 = 1/40
        theta = np.cumsum(a,axis=0)*t0
        # print(np.size(theta))
        maxNumM = 100
        L = theta.shape[0]
        # print(np.size(L))
        maxM = 2**np.floor(np.log2(L/2))
        # print(np.size(maxM))
        m = np.logspace(np.log10(1), np.log10(maxM), maxNumM)
        m = np.ceil(m).astype(int) # m must be an integer.
        m = np.unique(m) # Remove duplicates.
        # print(np.size(m))

        tau = m*t0
        # tau = tau.reshape((-1, 1))
        # tau = m / 40
        avar = np.zeros_like(tau)
        for i, mi in enumerate(m):
            avar[i] = np.sum((theta[1+2*mi:L] - 2*theta[1+mi:L-mi] + theta[1:L-2*mi])**2)
        avar /= (2*tau**2 * (L - 2*m))
        adev = np.sqrt(avar)

        #for angle random walk
        slope = -0.5
        logtau = np.log10(tau)
        logadev = np.log10(adev)
        dlogadev = np.diff(logadev) / np.diff(logtau)
        i = np.argmin(abs(dlogadev - slope))
        b = logadev[i] - slope*logtau[i]
        logN = slope*np.log(1) + b
        N = 10**logN
        tauN = 1
        lineN = N / np.sqrt(tau)

        #for rate random walk.
        sloper = 0.5
        ir = np.argmin(abs(dlogadev - sloper))
        br = logadev[ir] - sloper*logtau[ir]
        logK = sloper*np.log10(3) + br
        K = 10**logK
        tauK = 3
        lineK = K * np.sqrt(tau/3)

        #for bias instability
        sloperb= 0
        ib = np.argmin(abs(dlogadev - sloperb))
        bb = logadev[ib] - sloperb*logtau[ib]
        scfB = np.sqrt(2*np.log(2)/np.pi)
        logB = bb - np.log10(scfB)
        B = 10**logB
        tauB = tau[ib]
        lineB = B * scfB * np.ones(np.size(tau))
        # print(lineB)

        # slopeb = 0
        # logtau = log10(tau);
        # logadev = log10(adev);
        # dlogadev = diff(logadev) ./ diff(logtau);
        # [~, i] = min(abs(dlogadev - slope));

        # % Find the y-intercept of the line.
        # b = logadev(i) - slope*logtau(i);

        # % Determine the bias instability coefficient from the line.
        # scfB = sqrt(2*log(2)/pi);
        # logB = b - log10(scfB);
        # B = 10^logB

        # % Plot the results.
        # tauB = tau(i);
        # lineB = B * scfB * ones(size(tau));
        # figure
        # loglog(tau, adev, tau, lineB, '--', tauB, scfB*B, 'o')
        # title('Allan Deviation with Bias Instability')
        # xlabel('\tau')
        # ylabel('\sigma(\tau)')
        # legend('\sigma', '\sigma_B')
        # text(tauB, scfB*B, '0.664B')
        # grid on
        # axis equal

        return tau,adev,lineN,tauN,N,lineK,tauK,K,lineB,tauB,B,scfB*B
    
    def plot_with_RRW(self,fig,ax):
        # plot graph with angle random walk
        tauy,adevy,lineNy,tauNy,Ny,lineKy,tauKy,Ky,lineB,tauB,B,_ = self.a_varinace('gy')
        ax.loglog(tauy, adevy,label = r'$\sigma_Y$')
        ax.loglog(tauy, lineKy,linestyle='dashed',label = r'$\sigma_{Ky}$')
        ax.loglog(tauKy, Ky,label = r'$N_Y$',marker = 'o',color = 'b')
        
        tauz,adevz,lineNz,tauNz,Nz,lineKz,tauKz,Kz,lineB,tauB,B,_ = self.a_varinace('gz')
        ax.loglog(tauz, adevz,label = r'$\sigma_Z$')
        ax.loglog(tauz, lineKz,linestyle='dashed',label = r'$\sigma_{Kz}$')
        ax.loglog(tauKz, Kz,label = r'$N_Z$',marker = 'o',color = 'g')

        taux,adevx,lineNx,tauNx,Nx,lineKx,tauKx,Kx,lineB,tauB,B,_ = self.a_varinace('gx')
        ax.loglog(taux, adevx,label = r'$\sigma_X$')
        ax.loglog(taux, lineKx,linestyle='dashed',label = r'$\sigma_{Kx}$')
        ax.loglog(tauKx, Kx,label = r'$N_X$',marker = 'o',color = 'r')

        ax.set_title('Allan Deviation with rate random walk (gyro)')
        # ax.set_title('Allan Deviation with rate random walk (accel)')
        ax.legend()
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$\sigma(\tau)$')
        ax.grid(True, which='both')
        ax.set_aspect('equal', 'box')

        # filename = 'plot-' + str('Allan Deviation with rate random walk (accel)') +'.png'
        filename = 'plot-' + str('Allan Deviation with rate random walk (gyro)') +'.png'
        plt.savefig(filename)
        
        return Kx,Ky,Kz
    
    def plot_with_ARW(self,fig,ax):
        # plot with angle random walk
        tauy,adevy,lineNy,tauNy,Ny,lineKy,tauKy,Ky,lineB,tauB,B,_ = self.a_varinace('gy')
        ax.loglog(tauy, adevy,label = r'$\sigma_Y$')
        ax.loglog(tauy, lineNy,linestyle='dashed',label = r'$\sigma_{Ny}$')
        ax.loglog(tauNy, Ny,label = r'$N_Y$',marker = 'o',color = 'b')

        tauz,adevz,lineNz,tauNz,Nz,lineKz,tauKz,Kz,lineB,tauB,B,_ = self.a_varinace('gz')
        ax.loglog(tauz, adevz,label = r'$\sigma_Z$')
        ax.loglog(tauz, lineNz,linestyle='dashed',label = r'$\sigma_{Nz}$')
        ax.loglog(tauNz, Nz,label = r'$N_Z$',marker = 'o',color = 'g')

        taux,adevx,lineNx,tauNx,Nx,lineKx,tauKx,Kx,lineB,tauB,B,_ = self.a_varinace('gx')
        ax.loglog(taux, adevx,label = r'$\sigma_X$')
        ax.loglog(taux, lineNx,linestyle='dashed',label = r'$\sigma_{Nx}$')
        ax.loglog(tauNx, Nx,label = r'$N_X$',marker = 'o',color = 'r')

        ax.set_title('Allan Deviation with angle random walk (gyro)')
        # ax.set_title('Allan Deviation with angle random walk (accel)')
        ax.legend()
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$\sigma(\tau)$')
        ax.grid(True, which='both')
        ax.set_aspect('equal', 'box')
        
        # filename = 'plot-' + str('Allan Deviation with angle random walk (accel)') +'.png'
        filename = 'plot-' + str('Allan Deviation with angle random walk (gyro)') +'.png'
        plt.savefig(filename)
        return Nx,Ny,Nz
    
    def plot_BS(self,fid,ax):
        tauy,adevy,lineNy,tauNy,Ny,lineKy,tauKy,Ky,lineBy,tauBy,By,scfby = self.a_varinace('gy')
        ax.loglog(tauy, adevy,label = r'$\sigma_Y$')
        ax.loglog(tauy, lineBy,linestyle='dashed',label = r'$\sigma_{By}$')
        ax.loglog(tauBy, scfby,label = r'$B_Y$',marker = 'o',color = 'b')

        tauz,adevz,lineNz,tauNz,Nz,lineKz,tauKz,Kz,lineBz,tauBz,Bz,scfbz = self.a_varinace('gz')
        ax.loglog(tauz, adevz,label = r'$\sigma_Z$')
        ax.loglog(tauz, lineBz,linestyle='dashed',label = r'$\sigma_{Bz}$')
        ax.loglog(tauBz, scfbz,label = r'$B_Z$',marker = 'o',color = 'g')

        taux,adevx,lineNx,tauNx,Nx,lineKx,tauKx,Kx,lineBy,tauBy,Bx,scfbx = self.a_varinace('gx')
        ax.loglog(taux, adevx,label = r'$\sigma_X$')
        ax.loglog(taux, lineBy,linestyle='dashed',label = r'$\sigma_{Bx}$')
        ax.loglog(tauBy, scfbx,label = r'$B_X$',marker = 'o',color = 'r')

        ax.set_title('Allan Deviation with bias instability (gyro)')
        # ax.set_title('Allan Deviation with bias instability (accel)')
        ax.legend()
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$\sigma(\tau)$')
        ax.grid(True, which='both')
        ax.set_aspect('equal', 'box')
        
        # filename = 'plot-' + str('Allan Deviation with bias instability (accel)') +'.png'
        filename = 'plot-' + str('Allan Deviation with bias instability (gyro)') +'.png'
        plt.savefig(filename)
        return Bx,By,Bz
        
    def plot_with_adev(self,fig,ax):
        tauz,adevz,lineNy,tauNy,Ny,lineKy,tauKy,Ky,lineBy,tauBy,By,scfby = self.a_varinace('gz')
        ax.loglog(tauz, adevz,label = 'Gyro-z ')

        taup,pitch,lineNy,tauNy,Ny,lineKy,tauKy,Ky,lineBy,tauBy,By,scfby = self.a_varinace('gy')
        ax.loglog(taup, pitch,label = 'Gyro-y')

        taur,roll,lineNy,tauNy,Ny,lineKy,tauKy,Ky,lineBy,tauBy,By,scfby = self.a_varinace('gx')
        ax.loglog(taur, roll,label = 'Gyro-x')
        ax.legend()
        plt.title('Allan deviation for Gyro (rad/s)',fontsize=20)
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$\sigma(\tau)$')
        ax.grid(True, which='both')
        ax.set_aspect('equal', 'box')

        # filename = 'plot-' + str('Allan Deviation with Accel') +'.png'
        filename = 'plot-' + str('Allan Deviation with Gyro') +'.png'
        plt.savefig(filename)

    def get_gyros(self):
        # my_bag = bagreader(bag_file)
        # self.my_topic = my_bag.message_by_topic('/vectornav')
        data = pd.read_csv(self.my_topic,usecols = ['gx','gy','gz','time'])
        print("got here")

        # data['time'] = data['header.stamp.secs']- data['header.stamp.secs'][0]
        data['gyro_x'] = data['gx']
        data['gyro_y'] = data['gy']
        data['gyro_z'] = data['gz']
        print("about to plot the graph")

        # concatenated = pd.concat([data['angular_velocity.x'],data['angular_velocity.y'],data['angular_velocity.z']])
        plt.plot(data['time'], data['gyro_x'],label = 'Gyro x(rad/s)')
        plt.plot(data['time'], data['gyro_y'],label = 'Gyro y(rad/s)')
        plt.plot(data['time'], data['gyro_z'],label = 'Gyro z(rad/s)')
        # plt.ylim(-0.035,0.035)
        plt.xlabel('Time (s)',fontsize=20)
        plt.ylabel('Gyro (rad/s)',fontsize=20)
        plt.title('Gyro vs Time',fontsize=20)
        plt.legend()
        print("about")
        # plt.show()
        filename = 'plot-' + str('Gyro vs Time_locA') +'.png'
        plt.savefig(filename)

    def allan_verify(self,axis):
        a = pd.read_csv(self.my_topic,usecols=[str(axis)]).to_numpy()
        t0 = 1/40
        (tau, adev, _, _) = allantools.adev(a, rate=40, data_type='freq', taus='decade')
        plt.loglog(tau, adev,label = 'from allan tools')
        plt.title('Allan Deviation')
        plt.xlabel(r'$\tau$ [s]')
        plt.ylabel(r'$\sigma_y(\tau)$ [rad/s]')
        plt.grid(True)
        # plt.show()



if __name__ == "__main__":
    loc_a = '/home/yash/Desktop/my_lab2/EECE5554/LAB3/src/Data/LocationA.bag'
    gyro_csv = '/home/yash/Desktop/my_lab2/EECE5554/LAB3/src/Data/LocationA/demo_gyro.csv'
    accel_csv = '/home/yash/Desktop/my_lab2/EECE5554/LAB3/src/Data/LocationA/demo_accel_A.csv'
    adev_gyro = calc_allan(gyro_csv)
    # adev_gyro.cvt_bag_data_accel()
    adev_accel = calc_allan(accel_csv)
    
    # adev.cvt_bag_data()
    fig, ax = plt.subplots()
    # adev_gyro.get_gyros()
    # Nx,Ny,Nz = adev_accel.plot_with_ARW(fig,ax)
    # Kx,Ky,Kz = adev_accel.plot_with_RRW(fig,ax)
    # Bx,By,Bz = adev_accel.plot_BS(fig,ax)
    # adev_accel.plot_with_adev(fig,ax)
# 
    # Nx,Ny,Nz = adev_gyro.plot_with_ARW(fig,ax)
    # Kx,Ky,Kz = adev_gyro.plot_with_RRW(fig,ax)
    # Bx,By,Bz = adev_gyro.plot_BS(fig,ax)
    # adev_gyro.plot_with_adev(fig,ax)

    # adev.get_gyros(loc_a)
    tau,adev,lineN,tauN,N,lineK,tauK,K,lineB,tauB,B,scf = adev_gyro.a_varinace('gz')
    print(N)
    print(K)
    print(B)
    # ax.loglog(taup, pitch,label = 'Gyro-y')

    # taur,roll = adev.a_varinace('gx')
    # ax.loglog(taur, roll,label = 'Gyro-x')
    # print("angle random walk Yaw ((rad/s)/sqrt(Hz))", Nz)
    # print("angle random walk Pitch ((rad/s)/sqrt(Hz))", Ny)
    # print("angle random walk Roll ((rad/s)/sqrt(Hz))", Nx)
    # print("\n")
    # print("rate random walk Yaw ((rad/s)/sqrt(Hz))", Kz)
    # print("rate random walk Pitch ((rad/s)/sqrt(Hz))", Ky)
    # print("rate random walk Roll ((rad/s)/sqrt(Hz))", Kx)
    # print("\n")
    # print("Bias Instability Yaw ((rad/s)/sqrt(Hz))", Bz)
    # print("Bias Instability Pitch ((rad/s)/sqrt(Hz))", By)
    # print("Bias Instability Roll ((rad/s)/sqrt(Hz))", Bx)

# #     angle random walk Yaw ((rad/s)/sqrt(Hz)) 9.116399915435881e-05
# # angle random walk Pitch ((rad/s)/sqrt(Hz)) 0.00011483658170033269
# # angle random walk Roll ((rad/s)/sqrt(Hz)) 9.008381891352987e-05
#     ax.set_title('Allan Deviation')
#     ax.legend()
#     ax.set_xlabel(r'$\tau$')
#     ax.set_ylabel(r'$\sigma(\tau)$')
#     ax.grid(True, which='both')
#     ax.set_aspect('equal', 'box')
#     # # # adev.allan_verify('gz')
    
    plt.show()

