import scipy
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt

# Energía inicial
E0=0 #########################################################

# Número y longitud de los intervalos. Normalmente 6 de 250, así se ven bien las gráficas de SV
# y se barre una longitud total de 1500 que tarda una noche
numinterval=6 #########################################################
longitudint=250 #########################################################

# Dibujar autofunciones sí o no
dibujarprimeras = True #########################################################
dibujarultimas = False #########################################################

# Caótico sí o no? Para comparar el histograma con la distribución teórica correcta
caotico = True

for contador in range(numinterval):

    # Partición del intervalo de energías
    E1 = E0+longitudint*contador
    E2 = E1+longitudint
    Area = 6 # Área de la figura dibujada #########################################################
    NE = int(4*Area*longitudint)
    Energ = np.linspace(E1, E2, NE)

    # Partición del intervalo [0,2pi] (N trozos, N nodos contando t=0 y sin contar t=2pi)
    L = 13 # Longitud de la figura dibujada #########################################################
    Nt = min(int(6*L*np.sqrt(E2)/(2*np.pi)), 360)

    Ntchiq = int(0.1*Nt)
    Nt = 10*Ntchiq

    delta_t = 2*np.pi/Nt

    # Introducir función analítica que describa el borde del dominio
    # IMPORTANTE: QUE VAYA EN SENTIDO CONTRARIO A LAS AGUJAS DEL RELOJ!!!
    t = np.linspace(0, 2*np.pi, Nt, endpoint=False)

    # Array de esquinas de la estrella
    esquinas = np.zeros((10,2))
    for i in range(5):
        # Exteriores
        esquinas[2*i] = 1.25*np.array([np.cos(2*i*np.pi/5), np.sin(2*i*np.pi/5)])
        # Interiores
        esquinas[2*i+1] = 1.25*(np.sqrt(5)-1)/(np.sqrt(5)+1)*np.array([np.cos((2*i+1)*np.pi/5),
                                                                             np.sin((2*i+1)*np.pi/5)])
        
    # Rompo la simetría
    esquinas[1] = 1.25*np.array([np.cos(np.pi/5)*np.cos(np.pi/5), 0.5*np.sin(2*np.pi/5)])

    # Creo los segmentos que van de una esquina a la siguiente y los junto todos
    x = np.zeros((10,Ntchiq))
    y = np.zeros((10,Ntchiq))
    xprima = np.zeros((10,Ntchiq))
    yprima = np.zeros((10,Ntchiq))

    for i in range(9):
        x[i] = np.linspace(esquinas[i,0], esquinas[i+1,0], Ntchiq, endpoint = False)
        y[i] = np.linspace(esquinas[i,1], esquinas[i+1,1], Ntchiq, endpoint = False)
        xprima[i] = np.full(Ntchiq, 5*(esquinas[i+1,0]-esquinas[i,0])/np.pi)
        yprima[i] = np.full(Ntchiq, 5*(esquinas[i+1,1]-esquinas[i,1])/np.pi)

    x[9] = np.linspace(esquinas[9,0], esquinas[0,0], Ntchiq, endpoint = False)
    y[9] = np.linspace(esquinas[9,1], esquinas[0,1], Ntchiq, endpoint = False)
    xprima[9] = np.full(Ntchiq, 5*(esquinas[0,0]-esquinas[9,0])/np.pi)
    yprima[9] = np.full(Ntchiq, 5*(esquinas[0,1]-esquinas[9,1])/np.pi)

    tchiq = t[:Ntchiq]
    tchiq2 = t[Ntchiq:2*Ntchiq]
    x[0] += 0.2*np.sin(10*tchiq)*np.cos(tchiq)
    y[0] += 0.2*np.sin(10*tchiq)*np.sin(tchiq)
    x[1] += 0.2*np.sin(10*tchiq2)*np.cos(tchiq2)
    y[1] += 0.2*np.sin(10*tchiq2)*np.sin(tchiq2)

    xprima[0] += 10*0.2*np.cos(10*tchiq)*np.cos(tchiq) - 0.2*np.sin(10*tchiq)*np.sin(tchiq)
    yprima[0] += 10*0.2*np.cos(10*tchiq)*np.sin(tchiq) + 0.2*np.sin(10*tchiq)*np.cos(tchiq)
    xprima[1] += 10*0.2*np.cos(10*tchiq2)*np.cos(tchiq2) - 0.2*np.sin(10*tchiq2)*np.sin(tchiq2)
    yprima[1] += 10*0.2*np.cos(10*tchiq2)*np.sin(tchiq2) + 0.2*np.sin(10*tchiq2)*np.cos(tchiq2)


    x = np.concatenate([x[0],x[1],x[2],x[3],x[4],x[5],x[6],
                        x[7],x[8],x[9]]) #########################################################
    y = np.concatenate([y[0],y[1],y[2],y[3],y[4],y[5],y[6],
                        y[7],y[8],y[9]]) #########################################################
    
    xprima = np.concatenate([xprima[0],xprima[1],xprima[2],xprima[3],xprima[4],xprima[5],xprima[6],
                        xprima[7],xprima[8],xprima[9]]) #########################################################
    yprima = np.concatenate([yprima[0],yprima[1],yprima[2],yprima[3],yprima[4],yprima[5],yprima[6],
                        yprima[7],yprima[8],yprima[9]]) #########################################################
    
    xsegunda = np.zeros(Nt) #########################################################
    ysegunda = np.zeros(Nt) #########################################################

    for i in range(2*Ntchiq):
        xsegunda[i] = -(10*10*0.2+0.2)*np.sin(10*t[i])*np.cos(t[i]) - 2*0.2*0.2*10*np.cos(10*t[i])*np.sin(t[i])
        ysegunda[i] = -(10*10*0.2+0.2)*np.sin(10*t[i])*np.sin(t[i]) + 2*0.2*0.2*10*np.cos(10*t[i])*np.cos(t[i])
    

    # Arrays de puntos en el plano
    ejex = np.linspace(-2.25/np.sqrt(3), 2.5/np.sqrt(3), Nt, endpoint=False) #########################################################
    ejey = np.linspace(2.375/np.sqrt(3), -2.375/np.sqrt(3), Nt, endpoint=False) #########################################################


    # Si el primer elemento es 0 me lo quito de en medio. Es para ahorrar un if luego
    if(Energ[0]==0): Energ[0]=Energ[1]
    k = np.sqrt(Energ)

    # Defino los arrays de singular values y vectores si procede
    sv = np.zeros((NE,Nt))
    mayorsv = np.zeros(NE)
    tresultimossv = np.zeros((NE,3))
    if ((dibujarprimeras and contador==0) or (dibujarultimas and contador==numinterval-1)):
        autovector_u = np.zeros((NE,Nt), dtype=complex) 

    #Para cada valor de la energía
    for m in range(NE):
        # Con la función calculo la matriz A(t_i,t_j) para cada k
        A = np.zeros((Nt,Nt), dtype=complex)

        for i in range(Nt):
            for j in range(Nt):
                if i!=j: # Si i es distinto de j uso la fórmula sin problema
                    rx = x[i]-x[j]
                    ry = y[i]-y[j]
                    r = np.sqrt(rx**2+ry**2)

                    factorprimai = np.sqrt(xprima[i]**2+yprima[i]**2)
                    factorprimaj = np.sqrt(xprima[j]**2+yprima[j]**2)
                    A[i][j] = (-0.5j * k[m] * scipy.special.hankel1(1, k[m]*r) *
                                (rx*yprima[i] - ry*xprima[i])/r * factorprimaj/factorprimai * delta_t)
                    
                else:
                    # EN EL PAPER ESTA FÓRMULA ESTÁ MAL (HACE FALTA UN CAMBIO DE SIGNO)
                    factorprimacuad = xprima[i]**2+yprima[i]**2
                    A[i][j] = 1/(2*np.pi) * (xsegunda[i]*yprima[i]-ysegunda[i]*xprima[i])/factorprimacuad * delta_t - 1

        # Calculo la singular value decomposition (SVD)
        # Si dibuja autofunciones, hay que guardar los autovectores
        if ((dibujarprimeras and contador==0) or (dibujarultimas and contador==numinterval-1)):
            U = np.zeros((Nt,Nt), dtype=complex)
            Vh = np.zeros((Nt,Nt), dtype=complex)
            U, sv, Vh = np.linalg.svd(A)

            # Guardo los datos que me interesan
            mayorsv[m] = sv[0]
            tresultimossv[m] = (sv[-3],sv[-2],sv[-1])
            autovector_u[m] = np.conjugate(Vh[-1,:]) # m-ésimo candidato a autovector
            print(f"Matriz {m} SVDescompuesta\n")

        # Si no, nada, solo los SV
        else:
            sv = np.linalg.svd(A, compute_uv=False)

            # Guardo los datos que me interesan
            mayorsv[m] = sv[0]
            tresultimossv[m] = (sv[-3],sv[-2],sv[-1])
            print(f"Matriz {m} SVDescompuesta\n")

    # Python ordena los SV de mayor a menor, así que nos interesa el último elemento (-1) para cada energía
    # Localizamos los mínimos locales
    indicesminimos = argrelextrema(tresultimossv[:,-1], np.less)[0]

    # Descarto los que no nos interesan
    for m in range(len(indicesminimos)):
        if (tresultimossv[indicesminimos[m]][-1]/mayorsv[indicesminimos[m]]>0.02):
            indicesminimos[m] = -1

    # Escribo las energías permitidas en un archivo que se va agrandando
    with open(f"Autoenergías.csv", "a+") as f:
        for m in range(len(indicesminimos)):
            if (indicesminimos[m]!=-1):
                f.write(f"{Energ[indicesminimos[m]]}, {tresultimossv[indicesminimos[m]][-1]/mayorsv[indicesminimos[m]]}\n")
        
    # Dibujo los valores singulares
    plt.rcParams["font.family"] = "DejaVu Sans"
    ax = plt.subplot()

    ax.set_xlabel('Energy', fontname='DejaVu Sans', fontsize='12')
    ax.set_ylabel('Singular values', fontname='DejaVu Sans', fontsize='12')

    ax.plot(Energ, tresultimossv[:,-1], linestyle='-', marker='', markersize=4, color='#B4045F', label="$SV_1$", linewidth=1.0)
    ax.plot(Energ, tresultimossv[:,-2], linestyle='-', marker='', markersize=4, color='#5FB404', label="$SV_2$", linewidth=1.0)
    ax.plot(Energ, tresultimossv[:,-3], linestyle='-', marker='', markersize=4,color='#045FB4', label="$SV_3$", linewidth=1.0)

    ax.set_ylim(bottom=0)
    plt.legend(loc="upper right")
    plt.savefig(f'SV {E1} a {E2}.pdf',dpi=300, bbox_inches = "tight")

    plt.close()

    # Dibujo las autofunciones si procede
    if ((dibujarprimeras and contador==0) or (dibujarultimas and contador==numinterval-1)):

        # Para no calcularlo varias veces
        criticolinea = np.sqrt(Area)/Nt
        xdibujo=np.append(x, x[0])
        ydibujo=np.append(y, y[0])
        
        # Empiezo a iterar
        for m in range(len(indicesminimos)):
            # Si no era un verdadero mínimo, no hay nada que hacer
            if indicesminimos[m]==-1: continue
            else:
                psi = np.zeros((Nt,Nt), dtype=complex) # Esta es mi m-ésima función de onda
                mod2 = np.zeros((Nt,Nt))

                for i in range(Nt):
                    for j in range(Nt):
                        # Calculo la función de onda evaluada en el punto correspondiente como una integral
                        for l in range(Nt):
                            r = np.sqrt((ejex[j]-x[l])**2+(ejey[i]-y[l])**2)

                            # Si el punto cae muy cerca de la frontera, vale 0 directamente
                            if (r <= criticolinea):
                                psi[i][j]=0
                                break
                            else:
                                factorprima = np.sqrt(xprima[l]**2+yprima[l]**2)
                                psi[i][j] += (scipy.special.hankel1(0, k[indicesminimos[m]]*r) *
                                            autovector_u[indicesminimos[m]][l] * factorprima * delta_t)

                mod2 = psi.real**2 + psi.imag**2
                mod2=255*mod2/np.max(mod2) # Normalizo

                plt.rcParams["font.family"] = "DejaVu Sans"
                ax = plt.subplot()

                # Hago el mapa
                ax.imshow(mod2, extent=[ejex[0], ejex[-1], ejey[-1], ejey[0]], cmap='gray')
                ax.set_xlabel(f'$E={Energ[indicesminimos[m]]:.4f}$', fontname='DejaVu Sans', fontsize='12')

                # Dibujo el borde del dominio?
                ax.plot(xdibujo, ydibujo, color="0.6", linewidth=0.6)

                plt.savefig(f'E={Energ[indicesminimos[m]]:.4f}.pdf',dpi=300, bbox_inches = "tight")

                plt.close()
                print(f"Gráfica {m} dibujada\n")

