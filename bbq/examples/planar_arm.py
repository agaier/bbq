import math
import numpy as np
from bbq.domains._domain import BbqDomain
from bbq.utils import scale

class PlanarArm(BbqDomain):
    def __init__(self, param_bounds=[0, 1], **kwargs):
        self.param_bounds = param_bounds
        BbqDomain.__init__(self, **kwargs)        
    
    def _fitness(self, x):        
        return 1 - np.std(x)
    
    def _desc(self, thetas):
        c = np.cumsum(thetas)
        x = np.sum(np.cos(c)) / (2. * len(thetas)) + 0.5
        y = np.sum(np.sin(c)) / (2. * len(thetas)) + 0.5
        return np.array((x,y))

    def express(self, x):
        pheno = scale(x, [-math.pi, math.pi])
        return pheno

    def evaluate(self, x):
        """ Evaluates a single individual, giving an objective and descriptor
        value, along with any metadata to be saved (such as the full phenotype)

        Args:
            x ([numpy array]): Raw parameter values between 0 and 1

        Returns:
            obj, desc, pheno: objective, descriptor, metadata
        """
        pheno = self.express(x)
        obj = self._fitness(x) # fitness based on [0:1] variance
        desc = self._desc(pheno)
        return obj, desc, pheno        


def visualize(solution, ax):
    """Plots an arm with the given angles and link lengths on ax.
    
    Args:
        solution (np.ndarray): A (dim,) array with the joint angles of the arm.
        link_lengths (np.ndarray): The length of each link the arm.
        objective (float): The objective value of this solution.
        ax (plt.Axes): A matplotlib axis on which to display the arm.

        Adapted from: 
            https://docs.pyribs.org/en/stable/tutorials/arm_repertoire.html
    """
    #if np.isnan(solution): return
    link_lengths = np.ones(len(solution))/len(solution)
    lim = 1.05 * np.sum(link_lengths)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Plot each link / joint.
    pos = np.array([0, 0])  # Starting position of the next joint.
    cum_thetas = np.cumsum(solution)
    for link_length, cum_theta in zip(link_lengths, cum_thetas):
        # Calculate the end of this link.
        next_pos = pos + link_length * np.array(
            [np.cos(cum_theta), np.sin(cum_theta)])
        ax.plot([pos[0], next_pos[0]], [pos[1], next_pos[1]], "-ko", ms=3)
        pos = next_pos

    # Add points for the start and end positions.
    ax.plot(0, 0, "ro", ms=6)
    final_label = f"Final: ({pos[0]:.2f}, {pos[1]:.2f})"
    ax.plot(pos[0], pos[1], "go", ms=6, label=final_label)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend()

