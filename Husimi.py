import scipy
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Energía inicial
E0=0 #########################################################

# Número y longitud de los intervalos. Normalmente 6 de 250, así se ven bien las gráficas de SV
# y se barre una longitud total de 1500 que tarda una noche
numinterval=2 #########################################################
longitudint=250 #########################################################

# Dibujar Husimi sí o no
dibujarprimeras = True #########################################################
dibujarultimas = True #########################################################

for contador in range(numinterval):
    # Partición del intervalo de energías
    E1 = E0+longitudint*contador
    E2 = E1+longitudint

    if(contador==1):
        E1=4250
        E2=4500

    Area = 2.3 # Área de la figura dibujada #########################################################
    NE = int(8*Area*longitudint)
    Energ = np.linspace(E1, E2, NE)

    # Partición del intervalo [0,2pi] (N trozos, N nodos contando t=0 y sin contar t=2pi)
    L = 5.6 # Longitud de la figura dibujada #########################################################
    Nt = min(int(10*L*np.sqrt(E2)/(2*np.pi)),300)
    delta_t = 2*np.pi/Nt

    Energ = Energ[(160 < Energ) & (Energ < 161)]
    NE = len(Energ)
    

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
    if(contador==0): epsilon = 0.001
    else: epsilon = 0.0001
    for m in range(len(indicesminimos)):
        if (tresultimossv[indicesminimos[m]][-1]/mayorsv[indicesminimos[m]]>epsilon):
            indicesminimos[m] = -1

    # Escribo las energías permitidas en un archivo que se va agrandando
    with open(f"Autoenergias H cardioide.csv", "a+") as f:
        for m in range(len(indicesminimos)):
            if (indicesminimos[m]!=-1): f.write(f"{Energ[indicesminimos[m]]}\n")

    plt.close()

    mitad = int(Nt/2) # El eje x va de -pi a pi
    # Dibujo Husimi si procede
    if ((dibujarprimeras and contador==0) or (dibujarultimas and contador==numinterval-1)):
        
        # Empiezo a iterar
        for m in range(len(indicesminimos)):
            # Si no era un verdadero mínimo, no hay nada que hacer
            if indicesminimos[m]==-1: continue
            else:
                husimi = np.zeros((Nt,Nt), dtype=complex) # Este es el diagrama de Husimi (s,p)
                husimi2 = np.zeros((Nt,Nt)) # Este es su módulo al cuadrado, el que verdaderamente representamos
                p = np.linspace(1,-1, Nt, endpoint=False) # El momento normalizado de -1 a 1

                for i in range(Nt):
                    for j in range(Nt):
                        for l in range(Nt):

                            # Calculo el valor del estado coherente
                            #########################################################
                            parent = t[j]-t[l] + 2*np.pi*np.arange(-5, 6, 1)
                            c = np.sum(np.exp(k[indicesminimos[m]]*(-1j*p[i]*parent-parent**2/(2*np.pi))))
                            
                            # Calculo la contribución a la integral de Husimi en este punto
                            factorprima = np.sqrt(xprima[l]**2+yprima[l]**2)
                            husimi[i][j-mitad] += c * autovector_u[indicesminimos[m]][l] * factorprima * delta_t

                    print("Ciclo", i, "terminado\n")

                husimi2 = husimi.real**2 + husimi.imag**2
                minimo = np.min(husimi2)
                husimi2=255*(husimi2-minimo)/(np.max(husimi2)-minimo) # Normalizo

                plt.rcParams["font.family"] = "DejaVu Sans"
                ax = plt.subplot()

                # Hago el mapa
                ax.imshow(husimi2, extent=[-1, 1, -1, 1], cmap='gray')
                ax.set_ylabel('$p$', fontname='DejaVu Sans', fontsize='12')
                ax.set_xlabel(f'$s/\pi$ \n $E={Energ[indicesminimos[m]]:.4f}$', fontname='DejaVu Sans', fontsize='12')

                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                
                
                plt.savefig(f'Husimi E={Energ[indicesminimos[m]]:.4f} cardioide.pdf',dpi=300, bbox_inches = "tight")

                plt.close()
                print(f"Gráfica {m} dibujada\n")

# Energía inicial
E0=0 #########################################################

# Número y longitud de los intervalos. Normalmente 6 de 250, así se ven bien las gráficas de SV
# y se barre una longitud total de 1500 que tarda una noche
numinterval=2 #########################################################
longitudint=250 #########################################################

# Dibujar Husimi sí o no
dibujarprimeras = True #########################################################
dibujarultimas = True #########################################################

for contador in range(numinterval):

    # Partición del intervalo de energías
    E1 = E0+longitudint*contador
    E2 = E1+longitudint

    if(contador==1):
        E1=4250
        E2=4500

    Area = np.pi # Área de la figura dibujada #########################################################
    NE = int(5*Area*longitudint)
    Energ = np.linspace(E1, E2, NE)

    # Partición del intervalo [0,2pi] (N trozos, N nodos contando t=0 y sin contar t=2pi)
    L = 2*np.pi # Longitud de la figura dibujada #########################################################
    Nt = min(int(10*L*np.sqrt(E2)/(2*np.pi)),300)
    delta_t = 2*np.pi/Nt
    

    # Introducir función analítica que describa el borde del dominio
    # IMPORTANTE: QUE VAYA EN SENTIDO CONTRARIO A LAS AGUJAS DEL RELOJ!!!
    t = np.linspace(0, 2*np.pi, Nt, endpoint=False)

    x = np.cos(t) #########################################################
    y = np.sin(t) #########################################################

    xprima = -y #########################################################
    yprima = x #########################################################

    xsegunda = -x #########################################################
    ysegunda = -y #########################################################

    # Arrays de puntos en el plano
    ejex = np.linspace(-1.2, 1.2, Nt, endpoint=False) #########################################################
    ejey = np.linspace(1.1, -1.2, Nt, endpoint=False) #########################################################

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
    epsilon = 0.1
    for m in range(len(indicesminimos)):
        if (tresultimossv[indicesminimos[m]][-1]>epsilon):
            indicesminimos[m] = -1

    # Escribo las energías permitidas en un archivo que se va agrandando
    with open(f"Autoenergias H circulo.csv", "a+") as f:
        for m in range(len(indicesminimos)):
            if (indicesminimos[m]!=-1): f.write(f"{Energ[indicesminimos[m]]}\n")

    plt.close()

    mitad = int(Nt/2) # El eje x va de -pi a pi
    # Dibujo Husimi si procede
    if ((dibujarprimeras and contador==0) or (dibujarultimas and contador==numinterval-1)):
        
        # Empiezo a iterar
        for m in range(len(indicesminimos)):
            # Si no era un verdadero mínimo, no hay nada que hacer
            if indicesminimos[m]==-1: continue
            else:
                husimi = np.zeros((Nt,Nt), dtype=complex) # Este es el diagrama de Husimi (s,p)
                husimi2 = np.zeros((Nt,Nt)) # Este es su módulo al cuadrado, el que verdaderamente representamos
                p = np.linspace(1,-1, Nt, endpoint=False) # El momento normalizado de -1 a 1

                for i in range(Nt):
                    for j in range(Nt):
                        for l in range(Nt):

                            # Calculo el valor del estado coherente
                            #########################################################
                            parent = t[j]-t[l] + 2*np.pi*np.arange(-5, 6, 1)
                            c = np.sum(np.exp(k[indicesminimos[m]]*(-1j*p[i]*parent-parent**2/(2*np.pi))))
                            
                            # Calculo la contribución a la integral de Husimi en este punto
                            factorprima = np.sqrt(xprima[l]**2+yprima[l]**2)
                            husimi[i][j-mitad] += c * autovector_u[indicesminimos[m]][l] * factorprima * delta_t

                    print("Ciclo", i, "terminado\n")

                husimi2 = husimi.real**2 + husimi.imag**2
                husimi2=255*husimi2/np.max(husimi2) # Normalizo

                plt.rcParams["font.family"] = "DejaVu Sans"
                ax = plt.subplot()

                # Hago el mapa
                ax.imshow(husimi2, extent=[-1, 1, -1, 1], cmap='gray')
                ax.set_ylabel('$p$', fontname='DejaVu Sans', fontsize='12')
                ax.set_xlabel(f'$s/\pi$ \n $E={Energ[indicesminimos[m]]:.4f}$', fontname='DejaVu Sans', fontsize='12')

                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                
                
                plt.savefig(f'Husimi E={Energ[indicesminimos[m]]:.4f} circulo.pdf',dpi=300, bbox_inches = "tight")

                plt.close()
                print(f"Gráfica {m} dibujada\n")

# Energía inicial
E0=0 #########################################################

# Número y longitud de los intervalos. Normalmente 6 de 250, así se ven bien las gráficas de SV
# y se barre una longitud total de 1500 que tarda una noche
numinterval=2 #########################################################
longitudint=250 #########################################################

# Dibujar Husimi sí o no
dibujarprimeras = True #########################################################
dibujarultimas = True #########################################################

for contador in range(numinterval):

    # Partición del intervalo de energías
    E1 = E0+longitudint*contador
    E2 = E1+longitudint

    if(contador==1):
        E1=4250
        E2=4500

    Area = 4 # Área de la figura dibujada #########################################################
    NE = int(5*Area*longitudint)
    Energ = np.linspace(E1, E2, NE)

    # Partición del intervalo [0,2pi] (N trozos, N nodos contando t=0 y sin contar t=2pi)
    L = 8 # Longitud de la figura dibujada #########################################################
    Nt = min(int(10*L*np.sqrt(E2)/(2*np.pi)),300)

    Ntchiq = int(0.25*Nt)
    Nt = 4*Ntchiq # Para que sea divisible entre 4, estamos en un cuadrado dep

    delta_t = 2*np.pi/Nt

    # Introducir función analítica que describa el borde del dominio
    # IMPORTANTE: QUE VAYA EN SENTIDO CONTRARIO A LAS AGUJAS DEL RELOJ!!!
    t = np.linspace(0, 2*np.pi, Nt, endpoint=False)

    x1 = np.linspace(1, -1, Ntchiq, endpoint=False)
    y1 = np.full(Ntchiq, 1)
    x2 = np.full(Ntchiq, -1)
    y2 = np.linspace(1, -1, Ntchiq, endpoint=False)
    x3 = np.linspace(-1, 1, Ntchiq, endpoint=False)
    y3 = np.full(Ntchiq, -1)
    x4 = np.full(Ntchiq, 1)
    y4 = np.linspace(-1, 1, Ntchiq, endpoint=False)

    x1prima = np.full(Ntchiq, -4/np.pi)
    y1prima = np.zeros(Ntchiq)
    x2prima = np.zeros(Ntchiq)
    y2prima = np.full(Ntchiq, -4/np.pi)
    x3prima = np.full(Ntchiq, 4/np.pi)
    y3prima = np.zeros(Ntchiq)
    x4prima = np.zeros(Ntchiq)
    y4prima = np.full(Ntchiq, 4/np.pi)

    x = np.concatenate([x1,x2,x3,x4]) #########################################################
    y = np.concatenate([y1,y2,y3,y4]) #########################################################

    xprima = np.concatenate([x1prima,x2prima,x3prima,x4prima]) #########################################################
    yprima = np.concatenate([y1prima,y2prima,y3prima,y4prima]) #########################################################

    xsegunda = np.zeros(Nt) #########################################################
    ysegunda = np.zeros(Nt)  #########################################################

    # Arrays de puntos en el plano
    ejex = np.linspace(-1.2, 1.2, Nt, endpoint=False) #########################################################
    ejey = np.linspace(1.2, -1.2, Nt, endpoint=False) #########################################################

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
    epsilon = 0.1
    for m in range(len(indicesminimos)):
        if (tresultimossv[indicesminimos[m]][-1]>epsilon):
            indicesminimos[m] = -1

    # Escribo las energías permitidas en un archivo que se va agrandando
    with open(f"Autoenergias H cuadrado.csv", "a+") as f:
        for m in range(len(indicesminimos)):
            if (indicesminimos[m]!=-1): f.write(f"{Energ[indicesminimos[m]]}\n")

    plt.close()

    mitad = int(Nt/2) # El eje x va de -pi a pi
    # Dibujo Husimi si procede
    if ((dibujarprimeras and contador==0) or (dibujarultimas and contador==numinterval-1)):
        
        # Empiezo a iterar
        for m in range(len(indicesminimos)):
            # Si no era un verdadero mínimo, no hay nada que hacer
            if indicesminimos[m]==-1: continue
            else:
                husimi = np.zeros((Nt,Nt), dtype=complex) # Este es el diagrama de Husimi (s,p)
                husimi2 = np.zeros((Nt,Nt)) # Este es su módulo al cuadrado, el que verdaderamente representamos
                p = np.linspace(1,-1, Nt, endpoint=False) # El momento normalizado de -1 a 1

                for i in range(Nt):
                    for j in range(Nt):
                        for l in range(Nt):

                            # Calculo el valor del estado coherente
                            #########################################################
                            parent = t[j]-t[l] + 2*np.pi*np.arange(-5, 6, 1)
                            c = np.sum(np.exp(k[indicesminimos[m]]*(-1j*p[i]*parent-parent**2/(2*np.pi))))
                            
                            # Calculo la contribución a la integral de Husimi en este punto
                            factorprima = np.sqrt(xprima[l]**2+yprima[l]**2)
                            husimi[i][j-mitad] += c * autovector_u[indicesminimos[m]][l] * factorprima * delta_t
                        
                    print("Ciclo", i, "terminado\n")

                husimi2 = husimi.real**2 + husimi.imag**2
                husimi2=255*husimi2/np.max(husimi2) # Normalizo

                plt.rcParams["font.family"] = "DejaVu Sans"
                ax = plt.subplot()

                # Hago el mapa
                ax.imshow(husimi2, extent=[-1, 1, -1, 1], cmap='gray')
                ax.set_ylabel('$p$', fontname='DejaVu Sans', fontsize='12')
                ax.set_xlabel(f'$s/\pi$ \n $E={Energ[indicesminimos[m]]:.4f}$', fontname='DejaVu Sans', fontsize='12')

                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                
                
                plt.savefig(f'Husimi E={Energ[indicesminimos[m]]:.4f} cuadrado.pdf',dpi=300, bbox_inches = "tight")

                plt.close()
                print(f"Gráfica {m} dibujada\n")

# Energía inicial
E0=0 #########################################################

# Número y longitud de los intervalos. Normalmente 6 de 250, así se ven bien las gráficas de SV
# y se barre una longitud total de 1500 que tarda una noche
numinterval=2 #########################################################
longitudint=250 #########################################################

# Dibujar Husimi sí o no
dibujarprimeras = True #########################################################
dibujarultimas = True #########################################################

for contador in range(numinterval):

    # Partición del intervalo de energías
    E1 = E0+longitudint*contador
    E2 = E1+longitudint

    if(contador==1):
        E1=4250
        E2=4500

    Area = 6 # Área de la figura dibujada #########################################################
    NE = int(4*Area*longitudint)
    Energ = np.linspace(E1, E2, NE)

    # Partición del intervalo [0,2pi] (N trozos, N nodos contando t=0 y sin contar t=2pi)
    L = 12.2 # Longitud de la figura dibujada #########################################################
    Nt = min(int(10*L*np.sqrt(E2)/(2*np.pi)),300)

    if(contador == 1): Nt = min(int(10*L*np.sqrt(E2)/(2*np.pi)),360)

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
        esquinas[2*i] = np.sqrt(3)*np.array([np.cos(2*i*np.pi/5), np.sin(2*i*np.pi/5)])
        # Interiores
        esquinas[2*i+1] = np.sqrt(3)*(np.sqrt(5)-1)/(np.sqrt(5)+1)*np.array([np.cos((2*i+1)*np.pi/5),
                                                                             np.sin((2*i+1)*np.pi/5)])

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
    

    # Arrays de puntos en el plano
    ejex = np.linspace(-1.8, 2.0, Nt, endpoint=False) #########################################################
    ejey = np.linspace(1.9, -1.9, Nt, endpoint=False) #########################################################

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
    if(contador==0): epsilon = 0.02
    else: epsilon = 0.001
    for m in range(len(indicesminimos)):
        if (tresultimossv[indicesminimos[m]][-1]/mayorsv[indicesminimos[m]]>epsilon):
            indicesminimos[m] = -1

    # Escribo las energías permitidas en un archivo que se va agrandando
    with open(f"Autoenergias H estrella.csv", "a+") as f:
        for m in range(len(indicesminimos)):
            if (indicesminimos[m]!=-1): f.write(f"{Energ[indicesminimos[m]]}\n")

    plt.close()

    mitad = int(Nt/2) # El eje x va de -pi a pi
    # Dibujo Husimi si procede
    if ((dibujarprimeras and contador==0) or (dibujarultimas and contador==numinterval-1)):
        
        # Empiezo a iterar
        for m in range(len(indicesminimos)):
            # Si no era un verdadero mínimo, no hay nada que hacer
            if indicesminimos[m]==-1: continue
            else:
                husimi = np.zeros((Nt,Nt), dtype=complex) # Este es el diagrama de Husimi (s,p)
                husimi2 = np.zeros((Nt,Nt)) # Este es su módulo al cuadrado, el que verdaderamente representamos
                p = np.linspace(1,-1, Nt, endpoint=False) # El momento normalizado de -1 a 1

                for i in range(Nt):
                    for j in range(Nt):
                        for l in range(Nt):

                            # Calculo el valor del estado coherente
                            #########################################################
                            parent = t[j]-t[l] + 2*np.pi*np.arange(-5, 6, 1)
                            c = np.sum(np.exp(k[indicesminimos[m]]*(-1j*p[i]*parent-parent**2/(2*np.pi))))
                            
                            # Calculo la contribución a la integral de Husimi en este punto
                            factorprima = np.sqrt(xprima[l]**2+yprima[l]**2)
                            husimi[i][j-mitad] += c * autovector_u[indicesminimos[m]][l] * factorprima * delta_t

                    print("Ciclo", i, "terminado\n")

                husimi2 = husimi.real**2 + husimi.imag**2
                husimi2=255*husimi2/np.max(husimi2) # Normalizo

                plt.rcParams["font.family"] = "DejaVu Sans"
                ax = plt.subplot()

                # Hago el mapa
                ax.imshow(husimi2, extent=[-1, 1, -1, 1], cmap='gray')
                ax.set_ylabel('$p$', fontname='DejaVu Sans', fontsize='12')
                ax.set_xlabel(f'$s/\pi$ \n $E={Energ[indicesminimos[m]]:.4f}$', fontname='DejaVu Sans', fontsize='12')

                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                
                
                plt.savefig(f'Husimi E={Energ[indicesminimos[m]]:.4f} estrella.pdf',dpi=300, bbox_inches = "tight")

                plt.close()
                print(f"Gráfica {m} dibujada\n")