import matplotlib.pyplot as plt

# Assuming sNeuron is a class with ID, setYPos, setXPos methods and y and x attributes
class sNeuron:
    def __init__(self, ID):
        self.ID = ID
        self.x = 0
        self.y = 0

    def setYPos(self, y):
        self.y = y

    def setXPos(self, x):
        self.x = x

# Initialize neurons and set their positions based on the given logic
neuronList = []
for i in range(12):  # 6 legs * 3 joints per leg
    neuron = sNeuron(i)
    leg_id = neuron.ID //2  # Determine leg id (0-5)
    joint_id = neuron.ID % 2  # Determine joint type (0: coxa, 1: femur, 2: tibia)

    # Set y-axis based on leg (less spacing on y-axis)
    neuron.setYPos(-0.3 + (leg_id * 0.15))

    # Set x-axis based on joint type (more spacing on x-axis)
    neuron.setXPos(-0.8 + (joint_id * 1.5))

    neuronList.append(neuron)

# Extract the x and y coordinates for plotting
x_coords = [neuron.x for neuron in neuronList]
y_coords = [neuron.y for neuron in neuronList]


# Create a scatter plot to represent the substrate layout
plt.figure(figsize=(8, 6))
plt.scatter(x_coords, y_coords, c='blue')
plt.title('Substrate Layout for Hexapod Robot Neurons')
plt.xlabel('X-axis (Joint Type)')
plt.ylabel('Y-axis (Leg ID)')
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
