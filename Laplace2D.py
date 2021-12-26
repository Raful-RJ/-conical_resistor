
import numpy as np
import numpy.linalg as alg
from mesh2D import *
import matplotlib.pyplot as plt


class Laplace2D:

    def __init__(self,x0,x1,h,sigma, save = True, **kwgs):        
        self.__save = save

        if 'f' in kwgs.keys():
            self.__sigma = sigma
            self.__gridX, self.__gridY, self.__lim_fun, self.__features, self.__mesh,general = mesh(h,x0,x1,kwgs['f'])
        else:
            self.__mesh = np.load(kwgs['path'],allow_pickle = True)['data']
            self.__gridX = np.load(kwgs['path'],allow_pickle = True)['gridX']
            self.__gridY = np.load(kwgs['path'],allow_pickle = True)['gridY']
            self.__lim_fun = np.load(kwgs['path'],allow_pickle = True)['lim_fun']
            self.__features = np.load(kwgs['path'],allow_pickle = True)['features']
            general = np.load(kwgs['path'],allow_pickle = True)['general_data']
            
        
        self.__x0 = general[0]
        self.__x1 = general[1]
        self.__h = general[2]
        self.__idx_V = np.where(self.__features == 'potential')[0]
        self.__idx_Jx = np.where(self.__features == 'Jx')[0]
        self.__idx_Jy = np.where(self.__features == 'Jy')[0]
        self.__idx_up = np.where(self.__features == 'up_neighbor')[0]
        self.__idx_down = np.where(self.__features == 'down_neighbor')[0]
        self.__idx_left = np.where(self.__features == 'left_neighbor')[0]
        self.__idx_right = np.where(self.__features == 'right_neighbor')[0]
            
            

    def compile_pot(self):

        mesh = self.get_mesh()
        h = self.get_h()
        matrix = np.zeros((mesh.shape[0],mesh.shape[0]), dtype = 'float')
        charge_vector = np.zeros(mesh.shape[0], dtype = 'float')
        potential_gradient = np.zeros(mesh.shape[0], dtype = 'float')

        idx_up = self.get_idx_up()
        idx_down = self.get_idx_down()
        idx_left = self.get_idx_left()
        idx_right = self.get_idx_right()
        idx_V = self.get_idx_V()
        
        for i,element in enumerate(mesh):
            if '-1' in element[idx_up:]:
                charge_vector[i] = element[idx_V]
                matrix[i][i] = (h**2)
            else:
                up_neighbor = np.where(mesh[:,0]==element[idx_up])[0][0]
                down_neighbor = np.where(mesh[:,0]==element[idx_down])[0][0]
                left_neighbor = np.where(mesh[:,0]==element[idx_left])[0][0]
                right_neighbor = np.where(mesh[:,0]==element[idx_right])[0][0]

                matrix[i][up_neighbor] = 1
                matrix[i][down_neighbor] = 1
                matrix[i][left_neighbor] = 1
                matrix[i][right_neighbor] = 1
                matrix[i][i] = -4

        inv_matrix = alg.inv(matrix)
        potentials = (inv_matrix@charge_vector)*(h**2)
        mesh[:,idx_V] = potentials

         
        self.set_mesh(mesh)
        self.compile_grad()
        #self.plot()

    def compile_grad(self):

        mesh = self.get_mesh()

        idx_up = self.get_idx_up()
        idx_down = self.get_idx_down()
        idx_left = self.get_idx_left()
        idx_right = self.get_idx_right()

        idx_Jx =  self.get_idx_Jx()
        idx_Jy = self.get_idx_Jy()
        idx_V = self.get_idx_V()
        h = self.get_h()
        sigma = self.get_sigma()

        for i,element in enumerate(mesh):
            '''if '-1' in element[idx_up:]:
                mesh[i][idx_Jx] = mesh[i][idx_Jy] = 0
            else:
                up_neighbor = np.where(mesh[:,0]==element[idx_up])[0][0]
                down_neighbor = np.where(mesh[:,0]==element[idx_down])[0][0]
                left_neighbor = np.where(mesh[:,0]==element[idx_left])[0][0]
                right_neighbor = np.where(mesh[:,0]==element[idx_right])[0][0]

                mesh[i][idx_Jy] = -1*sigma*(mesh[up_neighbor][idx_V] - mesh[down_neighbor][idx_V])/(2*h)
                mesh[i][idx_Jx] = -1*sigma*(mesh[right_neighbor][idx_V] - mesh[left_neighbor][idx_V])/(2*h)'''
            def search_V(mesh,element,idx, idx_V =3):
                loc = np.where(mesh[:,0]==element[idx])[0][0]
                return mesh[loc][idx_V]

            list_idx = [idx_right,idx_left,idx_up,idx_down]
            count = [0,0]
                
            list_temp_V = []
            CoefDiff = []
            
            for idx in list_idx:
                if '-1' == element[idx]:
                    list_temp_V.append(element[idx_V])
                    CoefDiff.append(1)
                    count[0] += 1
                else:
                    list_temp_V.append(search_V(mesh,element,idx,idx_V))
                    CoefDiff.append(2)
                    count[1] += 1
                    
            mesh[i][idx_Jx] = -1*sigma*(list_temp_V[0] - list_temp_V[1])/(h*int((CoefDiff[0]*CoefDiff[1])**.5))
            mesh[i][idx_Jy] = -1*sigma*((list_temp_V[2] - list_temp_V[3]))/(h*int((CoefDiff[2]*CoefDiff[3])**.5))
            print(count)

        self.set_mesh(mesh)

        #compute current
        
        name = 'mesh'
        if self.__save: self.save_mesh(name)
                
        self.plot(name)
        

    def plot(self, *args):
        
        X = self.get_gridX()
        Y  = self.get_gridY()
        idx_Jx =  self.get_idx_Jx()
        idx_Jy = self.get_idx_Jy()
        idx_V = self.get_idx_V()
        if args == ():
            name = 'mesh'
        else: name = args[0]

        V = np.empty((X.shape[0],X.shape[1]))
        V[:] = np.nan
        Jx = np.empty((X.shape[0],X.shape[1]))
        Jx[:] = 0
        Jy = np.empty((X.shape[0],X.shape[1]))
        Jy[:] = 0

        for element in self.get_mesh():
            i = int(element[0].split('.')[0])
            j = int(element[0].split('.')[1])
            V[i][j] = element[idx_V]
            Jx[i][j] = element[idx_Jx]
            Jy[i][j] = element[idx_Jy]

        I = 2*np.pi*self.get_h()*(np.transpose(Jx)@Y[:,0])
        self.Jx = Jx
        self.__I = I

        cmap = 'jet'
        num = 2
        fig, ax = plt.subplots(ncols=num,constrained_layout=True, dpi = 200, figsize=plt.figaspect(0.44))
        #plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

        c = ax[0].pcolormesh(Y, X, V, cmap=cmap, shading='gouraud',vmin=min(self.get_mesh()[:,idx_V]), vmax=max(self.get_mesh()[:,idx_V]),snap = True)
        ax[0].set_title('Pontenciais (V)')
        fig.colorbar(c, ax=ax[0])
        
        ax[1].quiver(Y[::3,::3],X[::3,::3],1e4*Jy[::3,::3],1e4*Jx[::3,::3],color = 'k',angles='xy', scale_units='xy', scale=2)
        ax[1].set_title('Densidades de Corrente ($\\frac{A}{cm^2}$)')
        x, y = self.get_limfun()
        ax[1].plot(y,x,'--', color = 'gray', linewidth =0.7,alpha = 0.6)
        ax[1].set_yticks(np.linspace(0,1,6))
        ax[1].plot(-1*y,x,'--', color = 'gray',linewidth =0.7,alpha = 0.6)

        for i in range(num):
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].set_ylabel('cm')
            ax[i].set_xlabel('cm')

        
        #h = self.get_h()
            
        # set the limits of the plot to the limits of the data
        #ax.axis([X.min(), X.max(), Y.min(), Y.max()])
            
            
        #x,y = self.get_limfun()
        
        #ax.plot(x,y,'-', color = 'k', linewidth =4,alpha = 0.7)
        #ax.plot(x,-1*y,'-', color = 'k',linewidth =4,alpha = 0.7)

        plt.savefig(name + '.png')

    def save_mesh(self,filename):

        data = self.get_mesh()
        features = self.features()
        X = self.get_gridX()
        Y = self.get_gridY()
        lim_fun = self.get_limfun()
        general = np.array([self.get_x0(),self.get_x1(),self.get_h(),self.get_sigma()])

        np.savez_compressed(filename + '.npz',data = data, features = features,gridX = X, gridY = Y, lim_fun = lim_fun, general_data = general)        

    def get_x0(self):
        return self.__x0

    def get_x1(self):
        return self.__x1
    
    def get_h(self):
        return self.__h

    def features(self):
        return self.__features

    def get_mesh(self):
        return np.copy(self.__mesh)

    def get_gridX(self):
        return self.__gridX

    def get_gridY(self):
        return self.__gridY        

    def set_mesh(self,new_mesh):
        if self.__mesh.shape == new_mesh.shape:
            self.__mesh = new_mesh[:]
        else: raise ValueError('Could not set mesh with different shape.')

    def get_limfun(self):
        return self.__lim_fun[0],self.__lim_fun[1]

    def get_sigma(self):
        return self.__sigma
    def set_sigma(self, v):
        self.__sigma = v

    def get_idx_V(self):
        return int(self.__idx_V)

    def get_idx_Jx(self):
        return int(self.__idx_Jx)

    def get_idx_Jy(self):
        return int(self.__idx_Jy)

    def get_idx_up(self):
            return int(self.__idx_up)

    def get_idx_down(self):
            return int(self.__idx_down)

    def get_idx_left(self):
            return int(self.__idx_left)

    def get_idx_right(self):
            return int(self.__idx_right)

    def get_I(self):
        return self.__I
    
def fun(x):
    return 1 -0.5*x
        
        

    
        
        
        

    
