# Neuron Object - Represents and Individual SUPG
class sNeuron(object):
    
    def __init__(self, neuronID):
        self.neuronID = neuronID
        self.xPos = 0
        self.yPos = 0
        self.timeCounter = 0
        self.firstStepComplete = False

    #return the nueron ID -> value between 0 and 11 
    def ID(self):    
        return self.neuronID
    #x-position of nueron in morphological abstraction
    def getXPos(self):
        return self.xPos
    #y-position of nueron in morphological abstraction
    def getYPos(self):
        return self.yPos

    #return true once the neuron has fired at least once
    def getFirstStepComplete(self):
        return self.firstStepComplete
    #return true once the neuron has fired at least once
    def setStepComplete(self):
        self.firstStepComplete = True

    #set x position in morphological abstraction. Used when initializing all the SUPGs
    def setXPos(self, x):
        self.xPos = x

    #set y position in morphological abstraction. Used when initializing all the SUPGs
    def setYPos(self, y):
        self.yPos = y

    #return the timer value
    def getTimeCounter(self):
        return self.timeCounter

    #set the timer value
    def setTimeCounter(self, val):
        self.timeCounter = val