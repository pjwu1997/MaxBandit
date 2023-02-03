## Define the basic properties of sampler

## Basically, the sampler is represented by a backbone distribution
import numpy as np
class uniform_sampler(object):
    def __init__(self, center, sample_range):
        self.dimension = len(center)
        self.center = center
        self.sample_range = sample_range
    
    def get_sample_range(self):
        return self.sample_range
    
    def get_sample_points(self):
        """Obtains a numpy array of a sample point."""
        return self.sample_range * (0.5 - np.random.random_sample(self.dimension)) + self.center
    
    def get_multiple_sample_points(self, num_points):
        """Obtains a 2D numpy array of a sample points."""
        points = np.zeros((num_points, self.dimension))
        for i in range(num_points):
            points[i] = self.get_sample_points()
        return points
    
    def update_center(self, new_max_point, rate = 1):
        """Update the center of sample"""
        self.center = rate * (new_max_point - self.center) + self.center
        ## TODO: add noise to prevent stuck
