import matplotlib.pyplot as plt
import numpy as np
import csv

# Es caótico sí o no
caotico = True #########################################################

plt.rcParams["font.family"] = "DejaVu Sans"
ax = plt.subplot()      # subfigura

# Almaceno las separaciones entre energías consecutivas
with open('Autoenergías.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')

    datos = []
    for row in csvreader:
            datos.append(float(row[0]))

    datos = np.array(datos)

            
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

# ax.set_xlim(right=4.4)

plt.savefig('Histograma.pdf',dpi=300, bbox_inches = "tight")