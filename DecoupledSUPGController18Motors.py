'''
This module implements the Decoupled SUPG controller for a hexapod, 
adapted from  "Single Unit Pattern Generators for Quadruped Locomotion" (https://doi.org/10.1145/2463372.2463461)
by Morse et al. (2013).
By "decoupled", we mean that the output of a SUPG neuron is not computed based on the output of other SUPG neurons.
This is how the original implementation from the paper works.
'''
import copy
import math
from sNeuron import sNeuron
import numpy as np
from typing import List

# radius, offset, step_height, phase, duty_factor
tripod_gait = [0.15, 0, 0.05, 0.5, 0.5,  # leg 1
               0.15, 0, 0.05, 0.0, 0.5,  # leg 2
               0.15, 0, 0.05, 0.5, 0.5,  # leg 3
               0.15, 0, 0.05, 0.0, 0.5,  # leg 4
               0.15, 0, 0.05, 0.5, 0.5,  # leg 5
               0.15, 0, 0.05, 0.0, 0.5]  # leg 6


class DecoupledSUPGController18Motors:
    """
    The DecoupledSUPGController class is responsible for 
    managing the behaviour of a robot's legs using SUPG (Single Unit Pattern Generator) neurons. 
    It handles the generation of joint angles for each leg based on the SUPG model, 
    taking into account broken legs and initial offsets.

    Attributes:
        l_1 (float): The length of the first link in the robot's leg.
        l_2 (float): The length of the second link in the robot's leg.
        l_3 (float): The length of the third link in the robot's leg.
        dt (float): The time step for updating the SUPG neurons.
        period (float): The period of the SUPG oscillations.
        velocity (float): The velocity of the robot.
        crab_angle (float): The angle for crab-like movement.
        body_height (float): The height of the robot's body.
        brokenLegs (list[int]): List of indices of broken legs.
        wavelength (int): Wavelength parameter for the SUPG.
        cppn (object): The CPPN (Compositional Pattern Producing Network) used for neuron activation.
        neuronList (list[sNeuron]): List of sNeuron objects representing the SUPG neurons.
        firstStep (bool): Flag to indicate if the first step has been taken.
        initialOutputs (list[float]): Initial joint angles for the robot's legs.

    Notes:
        - joint_angles() and IMU_feedback() are required by the Simulator class.
    """

    def __init__(self, cppn, brokenLegs: List[int], params: List[float] = tripod_gait,
                 body_height: float = 0.15, period: float = 1.0, velocity: float = 0.46,
                 crab_angle: float = 0.0, dt: float = 1/240):
        """
        Initializes the BSUPGController with the given parameters.

        Args:
            cppn (object): The CPPN used for neuron activation.
            brokenLegs (list[int]): List of indices of broken legs.
            params (list[float], optional): Parameters for the tripod gait. Defaults to tripod_gait.
            body_height (float, optional): The height of the robot's body. Defaults to 0.15.
            period (float, optional): The period of the SUPG oscillations. Defaults to 1.0.
            velocity (float, optional): The velocity of the robot. Defaults to 0.46.
            crab_angle (float, optional): The angle for crab-like movement. Defaults to 0.0.
            dt (float, optional): The time step for updating the SUPG neurons. Defaults to 1/240.
        """
        # link lengths
        self.l_1 = 0.05317
        self.l_2 = 0.10188
        self.l_3 = 0.14735

        self.dt = dt
        self.period = period
        self.velocity = velocity
        self.crab_angle = crab_angle
        self.body_height = body_height
        self.brokenLegs = brokenLegs

        self.coxa_angle_range = [-0.906256, 0.90724405]
        self.femur_angle_range = [-.2, 0.64]
        self.tibia_angle_range = [-1.7, -1]

        # set index for broken legs. Index begins at 0, whereas leg numbers start at 1
        for i in range(len(self.brokenLegs)):
            self.brokenLegs[i] = self.brokenLegs[i] - 1

        # reshape values to correspond to respective SUPG neuron
        self.brokenLegsR = []
        for i in self.brokenLegs:
            self.brokenLegsR.append(2*i)

        self.wavelength = 100  # SUPG-Wavelength

        # CPPN has 3 inputs: x (x-coordinate of motor on substrate), y (y-coordinate of motor on substrate), and t (time)
        self.cppn = cppn
        self.firstStep = True

        # neurons in the list are ordered in the list in such a way 
        self.neuronList = []

        self.initializeSubstrate()

        self.initialOutputs = []

        # for dmg scenarios, if leg is broken, set leg angles to fixed positions
        for i in range(6):
            # if in broken leg array, set to fixed position
            if (i in self.brokenLegs):
                # all three servos
                self.initialOutputs.append(np.radians(0))
                self.initialOutputs.append(np.radians(90))
                self.initialOutputs.append(np.radians(-150))
            # otherwise, continue as normal
            else:
                # all three servos
                self.initialOutputs.append(0)
                self.initialOutputs.append(0.8994219)
                self.initialOutputs.append(-1.487756)

    def initializeSubstrate(self):
        """
        This method organizes the neurons (nodes) spatially on a substrate, modeling the robot's leg layout.

        Each neuron corresponds to a joint in a specific leg. The neurons are ordered sequentially
        in the list based on their leg and joint association. For a hexapod robot with 6 legs
        and 3 joints per leg (coxa, femur, and tibia), the neurons are arranged in the list as follows:

        - Leg 0: Coxa (ID 0), Femur (ID 1), Tibia (ID 2)
        - Leg 1: Coxa (ID 3), Femur (ID 4), Tibia (ID 5)
        - ...
        - Leg 5: Coxa (ID 15), Femur (ID 16), Tibia (ID 17)

        This arrangement simplifies mapping each neuron to its corresponding physical part in the robot.
        
        The x and y coordinates of each neuron are set to represent their spatial organization:
        - The x-axis position is set based on the joint type, with neurons representing the same type of joint (coxa, femur, tibia) 
        being more spaced out.
        - The y-axis position is set based on the leg ID, with neurons representing the same leg being less spaced out.
        """
        # Create nodes for each joint in each leg
        for i in range(18):  # 6 legs * 3 joints per leg
            self.neuronList.append(sNeuron(i))

        # Set coordinates in the substrate
        for neuron in self.neuronList:
            leg_id = neuron.ID() // 3  # Determine leg id (0-5)
            joint_id = neuron.ID() % 3  # Determine joint type (0: coxa, 1: femur, 2: tibia)

            # Set y-axis based on leg (less spacing on y-axis)
            neuron.setYPos(-0.3 + (leg_id * 0.15))

            # Set x-axis based on joint type (more spacing on x-axis)
            neuron.setXPos(-0.8 + (joint_id * 0.8))



    def getOffset(self, neuron: sNeuron) -> float:
        """
        Calculates the initial offset for a given neuron. 
        This offset is used to ensure that the robot does not fire all legs at once on the initial angle request.

        Args:
            neuron (sNeuron): The neuron for which to calculate the offset.

        Returns:
            float: The calculated offset for the neuron.
        """
        offset = 0
        # only use y angle to ensure all servos on same leg move at same time. Remember that the y-axis is the leg ID
        inputs = [0, neuron.getYPos(), 0]

        activation = self.cppn.activate(inputs)
        offset = (activation[1] + 1)

        if 0 <= offset <= 1:
            return offset
        else:
            return 1

    # return output of individual SUPG
    def getSUPGActivation(self, neuron) -> float:
        """
        Returns the activation output of an individual SUPG neuron. 

        Args:
            neuron (sNeuron): The SUPG neuron to be activated.

        Returns:
            float: The activation output of the neuron.
        """

        inputs = [neuron.getXPos(), neuron.getYPos(), neuron.getTimeCounter()]

        activation = self.cppn.activate(inputs)

        # with the SUPG architecture, we need the outputs to be normalized between 0 and 1
        # because the CPPN uses bipolar sigmoid for all outputs, which increases the range to -1, 1
        output = (activation[0]+1)/2

        return output

    def updateTime(self):
        """
        Updates the time counter for each neuron in the neuron list. 
        This method is used after CPPN input has been requested.
        """
        for neuron in self.neuronList:
            if neuron.getTimeCounter() >= 1:
                neuron.setTimeCounter(0)
            else:
                neuron.setTimeCounter((neuron.getTimeCounter() + self.dt))

    def reshapeServoOutput(self, neuron: sNeuron, output: float) -> float:
        """
        Reshapes the output of a neuron to be within the correct scale for each joint type.

        Args:
            neuron (sNeuron): The neuron whose output is being reshaped.
            output (float): The original output from the neuron.

        Returns:
            float: The reshaped output value suitable for the servo.
        """
        joint_type = neuron.ID() % 3  # Determine joint type: 0 - coxa, 1 - femur, 2 - tibia

        if joint_type == 0:  # coxa
            OldRange = (1 - 0)
            NewRange = (self.coxa_angle_range[1] - self.coxa_angle_range[0])  # Range for coxa joint
            NewValue = (((output - 0) * NewRange) / OldRange) + self.coxa_angle_range[0]
        elif joint_type == 1:  # femur
            OldRange = (1 - 0)
            NewRange = (self.femur_angle_range[1] - self.femur_angle_range[0])  # Range for femur joint
            NewValue = (((output - 0) * NewRange) / OldRange) + self.femur_angle_range[0]
        else:  # tibia
            OldRange = (1 - 0)
            NewRange = (self.tibia_angle_range[1] - self.tibia_angle_range[0]) # Range for tibia joint
            NewValue = (((output - 0) * NewRange) / OldRange) + self.tibia_angle_range[0]

        return NewValue

    ### Simulator methods ###

    def IMU_feedback(self, measured_attitude):
        """
        Placeholder for IMU feedback integration. Currently, it does not perform any operations.
        It's required by the simulator to be implemented.

        Args:
            measured_attitude (Any): The measured attitude from the IMU.
        """
        return

    def joint_angles(self, contact, t) -> np.ndarray:
        """
        Queries each SUPG neuron for its output and assembles the joint angles for the robot. It handles initial standing positions, broken legs, and offsets.

        Args:
            contact (list[bool]): A list indicating which legs are in contact with the ground.
            t (int): The current time step.

        Returns:
            np.ndarray: An array of joint angles for the robot's legs.
        """
        outputs = []

        # set up initial standing position
        if t == 0:
            return self.initialOutputs

        # set timer to offset to kickstart legs/avoid pronk
        # legs where offset == true , remain at T = zero
        if self.firstStep == True:
            for neuron in self.neuronList:
                neuron.setTimeCounter(self.getOffset(neuron))

            self.firstStep = False

        else:
            # Use triggers for legs in contact with the ground
            if len(contact) > 0:
                for i, val in enumerate(contact):
                    if val:
                        # Reset time counter for neurons associated with this leg
                        start_index = i * 3  # Three neurons per leg
                        for j in range(start_index, start_index + 3):
                            self.neuronList[j].setTimeCounter(0)

        # Compute output for each SUPG neuron
        for neuron in self.neuronList:
            # Check if neuron is associated with a broken leg
            if neuron.ID() in self.brokenLegsR:
                if neuron.ID() % 3 == 0:
                    outputs.append(np.radians(0))
                elif neuron.ID() % 3 == 1:
                    outputs.append(np.radians(90))
                else: 
                    outputs.append(np.radians(-150))
            else:
                # Get activation only if neuron's timer is within the valid range
                if 0 <= neuron.getTimeCounter() <= 1:
                    output = self.reshapeServoOutput(neuron, self.getSUPGActivation(neuron))
                    outputs.append(output)
                else:
                    # Keep neuron at stationary value if not ready to move. 

                    if not neuron.ID() % 3 == 0:
                        # if the servo is the coxa, set angle to 0 for stationary position
                        outputs.append(0)
                    elif neuron.ID() % 3 == 1:
                        # else (femur, tibia), set angle to close to .
                        outputs.append(0.8994219)
                    else: 
                        outputs.append(-1.487756)

        self.updateTime()

        return np.array(outputs)

