from ophyd import sim

# motor = sim.motor

from ophyd import PseudoPositioner, PseudoSingle, SoftPositioner
from ophyd.pseudopos import pseudo_position_argument, real_position_argument


class DummyPseudoMotor(PseudoPositioner):
    """
    A simple in-memory PseudoPositioner.
    - pseudo axis:  p  (what you read/set)
    - real axis:    r  (not connected to hardware; just a SoftPositioner)
    Mapping: p <-> r is identity.
    """

    # pseudo axis you interact with
    p = PseudoSingle(name="p")

    # "real" axis (soft/in-memory)
    r = SoftPositioner(name="r", limits=(-1e6, 1e6), init_pos=0.0)

    @pseudo_position_argument
    def forward(self, pseudo_pos):
        # pseudo -> real
        return self.RealPosition(r=float(pseudo_pos.p))

    @real_position_argument
    def inverse(self, real_pos):
        # real -> pseudo
        return self.PseudoPosition(p=float(real_pos.r))


# --- use it ---
motor = DummyPseudoMotor(name="dummy")