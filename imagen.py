import math
import random
from functools import partial
rand = partial (random.randint)

entradas=[[1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1],#0
    [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],#1
    [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1],#2
    [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1],#3
    [1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1],#4
    [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1],#5
    [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],#6
    [1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],#7
    [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],#8
    [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]]#9

def imprimir(entrada):
    print("")
    print("\t",end="") 
    y = ''.join([str(x) for x in entrada ])
    y=y.replace("0",".")
    y=y.replace("1","@")
    for i,z in enumerate(y):
        print(z,end='')
        if (i+1)%5==0:
            print('')
            print("\t",end='') 

objetivos = [[1 if i==j else 0 for i in range(10)] for j in range(10)]

def producto_punto(x, y):
    return sum([i * j for i, j in zip(x,y)])

def sigmoide(x):
    return 1 / (1 + math.exp(-x))

def salida_neurona(pesos, entradas):
    return sigmoide(producto_punto(pesos, entradas))

def ffnn(red_neuronal, entrada):
    salidas = []
    for capa in red_neuronal:
        entrada = entrada + [1]
        #print("capa: ", capa, "\nentrada: ", entrada)
        salida = [salida_neurona(neurona, entrada) for neurona in capa]
        #print("salida: ", salida, "\n")
        salidas.append(salida)
        entrada = salida
    return salidas

def random_nn():
    random.seed(7)
    pesos = [
        [   #Capa oculta, 5 neuronas
            [rand(-100, 100)/ 100 for _ in range(26)],
            [rand(-100, 100)/ 100 for _ in range(26)],
            [rand(-100, 100)/ 100 for _ in range(26)],
            [rand(-100, 100)/ 100 for _ in range(26)],
            [rand(-100, 100)/ 100 for _ in range(26)],
        ],
        [   #Capa salida, 10 neurona
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
            [rand(-100, 100)/ 100 for _ in range(6)],
        ]
    ]
    print("Pesos aleatorios: ",pesos)
    return pesos

def backpropagation(xor_nn, v_entrada, v_objetivo):
    salidas_ocultas, salidas = ffnn(xor_nn, v_entrada)
    salida_nuevo = []
    oculta_nuevo = []
    alfa = 0.1 
    #error = 0.5 * sum((salida - objetivo) * (salida - objetivo) for salida, objetivo in zip(salidas, v_objetivo))
    error = (1/(2*10))*sum((salida - objetivo) * (salida - objetivo) for salida, objetivo in zip(salidas, v_objetivo))
    salida_deltas = [salida * (1 - salida) * (salida - objetivo) for salida, objetivo in zip(salidas, v_objetivo)]
    for i, neurona_salida in enumerate(xor_nn[-1]):
        for j, salida_oculta in enumerate(salidas_ocultas + [1]):
            neurona_salida[j] -= salida_deltas[i] * salida_oculta * alfa
        salida_nuevo.append(neurona_salida)
        #print("pesos neurona salida: ", i, neurona_salida)
    ocultas_deltas = [salida_oculta * (1 -  salida_oculta) * producto_punto(salida_deltas, [n[i] for n in xor_nn[-1]]) for i, salida_oculta in enumerate(salidas_ocultas)]
    for i, neurona_oculta in enumerate(xor_nn[0]):
        for j, inPut in enumerate(v_entrada + [1]):
            neurona_oculta[j] -= ocultas_deltas[i] * inPut * alfa
        oculta_nuevo.append(neurona_oculta)
        #print("pesos neurona oculta: ", i, neurona_oculta)
    return oculta_nuevo, salida_nuevo, error

pesos=random_nn()


promedio_errores_cuadrados = 1
i = 1
while promedio_errores_cuadrados>0.0005:
    #normalizamos los datos de entrada de [-1, 1] mediante la formula v' = (v-min)/(max-min)*(newmax-newmin)+newmin
    oculta, salida, error1 = backpropagation(pesos,entradas[0], objetivos[0])
    pesos = [oculta, salida]
    oculta, salida, error2 = backpropagation(pesos,entradas[1], objetivos[1])
    pesos = [oculta, salida]
    oculta, salida, error3 = backpropagation(pesos,entradas[2], objetivos[2])
    pesos = [oculta, salida]
    oculta, salida, error4 = backpropagation(pesos,entradas[3], objetivos[3])
    pesos = [oculta, salida]
    oculta, salida, error5 = backpropagation(pesos,entradas[4], objetivos[4])
    pesos = [oculta, salida]
    oculta, salida, error6 = backpropagation(pesos,entradas[5], objetivos[5])
    pesos = [oculta, salida]
    oculta, salida, error7 = backpropagation(pesos,entradas[6], objetivos[6])
    pesos = [oculta, salida]
    oculta, salida, error8 = backpropagation(pesos,entradas[7], objetivos[7])
    pesos = [oculta, salida]
    oculta, salida, error9 = backpropagation(pesos,entradas[8], objetivos[8])
    pesos = [oculta, salida]
    oculta, salida, error10 = backpropagation(pesos,entradas[9],objetivos[9])
    pesos = [oculta, salida]

    promedio_errores_cuadrados = (error1 + error2 + error3 + error4 + error5 + error6 + error7 + error8 + error9 + error10) / 10
    i += 1
print("\nPesos despues del traning: ",pesos)
print("\nPromedio de errores:",promedio_errores_cuadrados)
print("Iteraciones:",i)

print("\n\nProbabilidades de que un 0 sea",ffnn(pesos,entradas[0])[1])
print("\n\nProbabilidades de que un 1 sea",ffnn(pesos,entradas[1])[1])
print("\n\nProbabilidades de que un 2 sea",ffnn(pesos,entradas[2])[1])
print("\n\nProbabilidades de que un 3 sea",ffnn(pesos,entradas[3])[1])
print("\n\nProbabilidades de que un 4 sea",ffnn(pesos,entradas[4])[1])
print("\n\nProbabilidades de que un 5 sea",ffnn(pesos,entradas[5])[1])
print("\n\nProbabilidades de que un 6 sea",ffnn(pesos,entradas[6])[1])
print("\n\nProbabilidades de que un 7 sea",ffnn(pesos,entradas[7])[1])
print("\n\nProbabilidades de que un 8 sea",ffnn(pesos,entradas[8])[1])
print("\n\nProbabilidades de que un 9 sea",ffnn(pesos,entradas[9])[1])