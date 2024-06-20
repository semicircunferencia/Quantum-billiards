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
caotico = False

for contador in range(numinterval):

    # Partición del intervalo de energías
    E1 = E0+longitudint*contador
    E2 = E1+longitudint
    Area = 2.3 # Área de la figura dibujada #########################################################
    NE = int(8*Area*longitudint)
    Energ = np.linspace(E1, E2, NE)

    # Partición del intervalo [0,2pi] (N trozos, N nodos contando t=0 y sin contar t=2pi)
    L = 5.6 # Longitud de la figura dibujada #########################################################
    Nt = min(int(10*L*np.sqrt(4500)/(2*np.pi)),300)
    delta_t = 2*np.pi/Nt

    # Introducir función analítica que describa el borde del dominio
    # IMPORTANTE: QUE VAYA EN SENTIDO CONTRARIO A LAS AGUJAS DEL RELOJ!!!
    t = np.linspace(0, 2*np.pi, Nt, endpoint=False)
    r = 0.7*(1+np.cos(t))
    rprima = -0.7*np.sin(t)
    rsegunda = -0.7*np.cos(t)

    x = r*np.cos(t) #########################################################
    y = r*np.sin(t) #########################################################

    xprima = rprima*np.cos(t) - r*np.sin(t) #########################################################
    yprima = rprima*np.sin(t) + r*np.cos(t) #########################################################

    xsegunda = rsegunda*np.cos(t) -2*rprima*np.sin(t) -r*np.cos(t) #########################################################
    ysegunda = rsegunda*np.sin(t) +2*rprima*np.cos(t) -r*np.sin(t) #########################################################

    # Arrays de puntos en el plano
    ejex = np.linspace(-0.45, 1.65, Nt, endpoint=False) #########################################################
    ejey = np.linspace(1.05, -1.05, Nt, endpoint=False) #########################################################

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
    epsilon = 0.01
    for m in range(len(indicesminimos)):
        if (tresultimossv[indicesminimos[m]][-1]>epsilon):
            indicesminimos[m] = -1

    # Escribo las energías permitidas en un archivo que se va agrandando
    with open(f"Autoenergías.csv", "a+") as f:
        for m in range(len(indicesminimos)):
            if (indicesminimos[m]!=-1): f.write(f"{Energ[indicesminimos[m]]}\n")
        
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

# Ahora que ha terminado lo gordo calculamos la separación entre niveles normalizada
# y dibujamos el correspondiente histograma
plt.rcParams["font.family"] = "DejaVu Sans"
ax = plt.subplot()      # subfigura

# Almaceno las separaciones entre energías consecutivas
datos = np.loadtxt('Autoenergías.csv')
separaciones = np.zeros(len(datos)-1)
for i in range(len(separaciones)): separaciones[i] = datos[i+1]-datos[i]

# Normalizo
promedio = np.mean(separaciones)
separaciones=separaciones/promedio

# Dibujo. Si es caótico la distribución de Wigner
if caotico:
    x = np.linspace(0, 3, 200)
    y = 0.5*np.pi*x*np.exp(-0.25*np.pi*x*x)

else: # Si no, es exponencial
    x = np.linspace(0, 4, 200)
    y = np.exp(-x)

# Configurar ejes
ax.set_ylabel('Frequency', fontname='DejaVu Sans', fontsize='12')
ax.set_xlabel('Level spacing', fontname='DejaVu Sans', fontsize='12')

# Creación de la gráfica
plt.hist(separaciones, bins='auto', density=True, color='#CBC3E3')
ax.plot(x, y, linestyle='-', marker='', markersize=4, color='#B4045F')  #marker=puntos

# Guardar la gráfica
plt.savefig('Histograma.pdf',dpi=300, bbox_inches = "tight")