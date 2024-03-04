from mailer.hexapod.simulator import Simulator
import pickle
from .DecoupledSUPGController18Motors import DecoupledSUPGController18Motors
import os 

def get_cppn(file_name: str):
    with open(file_name, 'rb') as f:
        CPPN = pickle.load(f)
    return CPPN

if __name__ == "__main__":

    cppn = get_cppn(os.path.join("DecoupledSUPG", "evolution_output", "pickles", "decoupled_supg_12_motors1.pkl"))
    simulator = Simulator(DecoupledSUPGController18Motors(
        cppn, []), follow=True, visualiser=True, collision_fatal=False, failed_legs=[])
    while True:
        simulator.step()

