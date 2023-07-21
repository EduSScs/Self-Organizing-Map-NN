import re
import numpy as np
import matplotlib.cm as colormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# Opens the file and reads the first line
# Code to read from file remains the same as last assignemnt
file = open("L30fft_32.out")
first = file.readline()
numbers = re.findall(r'\d+', first)
for i in range(len(numbers)):
    numbers[i] = int(numbers[i])

data = []

for x in range(numbers[0]):
    temp = file.readline()
    temp = re.findall(r'\d+\.*\d*',temp)
    for i in range(len(temp)):
        temp[i] = float(temp[i])    
    data.append(temp)

data = data/np.amax(data)

# SOM SIZE
SOM_SIZE = 10

# NUMBER OF INPUTS PER DATA
INPUTS_PER_DATA = np.shape(data)[1]

# Create the weight matrices
weights = np.random.rand(SOM_SIZE,SOM_SIZE,INPUTS_PER_DATA)


# Defining the alpha and sigma functions
alpha = 0.2

# Number of iterations
iterations = 500

# Getting distance between two arrays
def distance(x, y):
    result = x-y
    result = np.square(result)
    result = np.sum(result)
    result = np.sqrt(result)
    return result

# Getting the neighborhood function
def closest_match(current_vector):
    result = None
    smallest_distance = 99999999999999
    for x in range(SOM_SIZE):
        for y in range(SOM_SIZE):
            temp = distance(weights[x,y], current_vector)
            if temp < smallest_distance:
                smallest_distance = temp
                result = [x,y]

    return result

# Gaussian function
def gaussian(x):
    return np.exp(-x**2)

# Calculate the change
def calculate_change(chosen, weight):
    return (chosen-weight)*alpha

# Training the Neural Network
for i in range(iterations):

    chosen = data[i%len(data)]
    coordinates = closest_match(chosen)
    alpha = alpha*0.99

    for x in range(SOM_SIZE):
        for y in range(SOM_SIZE):
            # Gets euclidean distance
            current_dist = distance(np.array(coordinates), np.array([x,y]))
            if current_dist == 1:
                # If the distance is 1 then update the weights
                change = gaussian(current_dist)
                weights[x,y] = (calculate_change(chosen, weights[x,y])*change+weights[x,y])*alpha  

# Plotting the data
heat_map = colormap.get_cmap('plasma')
color = np.empty((SOM_SIZE,SOM_SIZE))
for x in range(SOM_SIZE):
    for y in range(SOM_SIZE):
        #Gets the average of the weights to determien the color
        color[x,y] = np.average(weights[x,y])
        # Fills in the color
        plt.fill_between([x, x + 1], y, y + 1, color=heat_map(color[x,y]))


#Create lines
plt.yticks(np.arange(0, SOM_SIZE, 1))
plt.xticks(np.arange(0, SOM_SIZE, 1))
plt.grid(True)

plt.title("Heat Map")
plt.show()