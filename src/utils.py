import numpy as np
from typing import Callable, Iterable, Union
from numpy.typing import NDArray

class Bin:
    """
    Represents a histogram bin with a start position, width, and height.

    Parameters
    ----------
    start : float
        The starting position of the bin.
    width : float
        The width of the bin.
    height : float
        The height of the bin.
    """

    def __init__(self, start: float, width: float, height: float) -> None:
        self.start = start
        self.width = width
        self.height = height

    def __lt__(self, other: Union['Bin', NDArray, float, int]) -> Union[bool, NDArray]:
        """
        Less than comparison based on bin height.

        Parameters
        ----------
        other : Bin, NDArray, float, or int
            The object to compare with.

        Returns
        -------
        bool or NDArray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                res = self.height < other.height
            case int() | float():
                res = self.height < other
            case np.ndarray():
                vec_fn = np.vectorize(lambda x1, x2: x1 < x2)
                res = vec_fn(self, other)
        return res

    def __gt__(self, other: Union['Bin', NDArray, float, int]) -> Union[bool, NDArray]:
        """
        Greater than comparison based on bin height.

        Parameters
        ----------
        other : Bin, NDArray, float, or int
            The object to compare with.

        Returns
        -------
        bool or NDArray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                res = self.height > other.height
            case int() | float():
                res = self.height > other
            case np.ndarray():
                vec_fn = np.vectorize(lambda x1, x2: x1 > x2)
                res = vec_fn(self, other)
        return res

    def __le__(self, other: Union['Bin', NDArray, float, int]) -> Union[bool, NDArray]:
        """
        Less than or equal to comparison based on bin height.

        Parameters
        ----------
        other : Bin, NDArray, float, or int
            The object to compare with.

        Returns
        -------
        bool or NDArray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                res = self.height <= other.height
            case int() | float():
                res = self.height <= other
            case np.ndarray():
                vec_fn = np.vectorize(lambda x1, x2: x1 <= x2)
                res = vec_fn(self, other)
        return res

    def __ge__(self, other: Union['Bin', NDArray, float, int]) -> Union[bool, NDArray]:
        """
        Greater than or equal to comparison based on bin height.

        Parameters
        ----------
        other : Bin, NDArray, float, or int
            The object to compare with.

        Returns
        -------
        bool or NDArray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                res = self.height >= other.height
            case int() | float():
                res = self.height >= other
            case np.ndarray():
                vec_fn = np.vectorize(lambda x1, x2: x1 >= x2)
                res = vec_fn(self, other)
        return res

import numpy as np
from typing import Iterable, Callable, Union

class Ground:
    """
    Represents a collection of bins.

    Parameters
    ----------
    bins : Iterable[Bin]
        The bins that make up the ground.
    """

    def __init__(self, bins: Iterable['Bin']) -> None:
        self.bins = np.array(bins)

    def __getitem__(self, idxs) -> 'Ground':
        """
        Get a subset of the bins.

        Parameters
        ----------
        idxs : int, slice, or array-like
            Indices to access the bins.

        Returns
        -------
        Ground
            A new Ground instance with the selected bins.
        """
        return Ground(self.bins[idxs])
    
    def get_bins_width(self) -> np.ndarray:
        """
        Get the widths of the bins.

        Returns
        -------
        np.ndarray
            The widths of the bins.
        """
        width_fn: Callable[[Bin], float] = lambda x: x.width
        width_fn_vec = np.vectorize(width_fn)
        return width_fn_vec(self.bins)
    
    def get_bins_height(self) -> np.ndarray:
        """
        Get the heights of the bins.

        Returns
        -------
        np.ndarray
            The heights of the bins.
        """
        height_fn: Callable[[Bin], float] = lambda x: x.height
        height_fn_vec = np.vectorize(height_fn)
        return height_fn_vec(self.bins)
    
    def get_bins_start(self) -> np.ndarray:
        """
        Get the start positions of the bins.

        Returns
        -------
        np.ndarray
            The start positions of the bins.
        """
        start_fn: Callable[[Bin], float] = lambda x: x.start
        start_fn_vec = np.vectorize(start_fn)
        return start_fn_vec(self.bins)
    
    def get_area(self) -> np.ndarray:
        """
        Get the area of all bins.

        Returns
        -------
        np.ndarray
            The area of all bins.
        """
        return np.sum(self.get_bins_width() * self.get_bins_height())
    
    def get_water_area(self, level: Union[int, float]):
        """
        Get the area lower water level and upper then bins tops.

        Returns
        -------
        np.ndarray
            The area lower water level and upper then bins tops.
        """
        return np.sum(self.get_bins_width() * (level - self.get_bins_height()))

    def add_bin(self, bin: 'Bin') -> None:
        """
        Add a bin to the ground.

        Parameters
        ----------
        bin : Bin
            The bin to be added.
        """
        self.bins = np.append(self.bins, bin)
    
    def merge(self, other: 'Ground') -> None:
        """
        Merge another ground into this ground.

        Parameters
        ----------
        other : Ground
            The other ground to be merged.
        """
        self.bins = np.concatenate([self.bins, other.bins])


class Extremum:
    """
    Represents an extremum with a bin and a sign.

    Parameters
    ----------
    bin : Bin
        The bin associated with the extremum.
    sign : int
        The sign of the extremum (e.g., -1 for minimum, 1 for maximum).
    """

    def __init__(self, bin: 'Bin', sign: int) -> None:
        self.bin = bin
        self.sign = sign
        self.height = self.bin.height
        self.start = self.bin.start
        self.width = self.bin.width
    
    def __sub__(self, other: Union['Extremum', float, int]) -> float:
        """
        Subtract the height of another extremum or a numeric value from this extremum's height.

        Parameters
        ----------
        other : Extremum, float, or int
            The other extremum or numeric value to subtract.

        Returns
        -------
        float
            The result of the subtraction.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin.height - other.bin.height
            case int() | float():
                return self.bin.height - other
    
    def __add__(self, other: Union['Extremum', float, int]) -> float:
        """
        Add the height of another extremum or a numeric value to this extremum's height.

        Parameters
        ----------
        other : Extremum, float, or int
            The other extremum or numeric value to add.

        Returns
        -------
        float
            The result of the addition.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin.height + other.bin.height
            case int() | float():
                return self.bin.height + other
            
    def __mul__(self, other: Union['Extremum', float, int]) -> float:
        """
        Multiply the height of another extremum or a numeric value with this extremum's height.

        Parameters
        ----------
        other : Extremum, float, or int
            The other extremum or numeric value to multiply.

        Returns
        -------
        float
            The result of the multiplication.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin.height * other.bin.height
            case int() | float():
                return self.bin.height * other
    
    def __truediv__(self, other: Union['Extremum', float, int]) -> float:
        """
        Divide the height of this extremum by another extremum's height or a numeric value.

        Parameters
        ----------
        other : Extremum, float, or int
            The other extremum or numeric value to divide by.

        Returns
        -------
        float
            The result of the division.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin.height / other.bin.height
            case int() | float():
                return self.bin.height / other
            
    def __lt__(self, other: Union['Extremum', float, int, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Less than comparison based on bin height.

        Parameters
        ----------
        other : Extremum, float, int, or np.ndarray
            The object to compare with.

        Returns
        -------
        bool or np.ndarray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin < other.bin
            case int() | float() | np.ndarray():
                return self.bin < other
    
    def __gt__(self, other: Union['Extremum', float, int, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Greater than comparison based on bin height.

        Parameters
        ----------
        other : Extremum, float, int, or np.ndarray
            The object to compare with.

        Returns
        -------
        bool or np.ndarray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin > other.bin
            case int() | float() | np.ndarray():
                return self.bin > other
            
    def __le__(self, other: Union['Extremum', float, int, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Less than or equal to comparison based on bin height.

        Parameters
        ----------
        other : Extremum, float, int, or np.ndarray
            The object to compare with.

        Returns
        -------
        bool or np.ndarray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin <= other.bin
            case int() | float() | np.ndarray():
                return self.bin <= other
    
    def __ge__(self, other: Union['Extremum', float, int, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Greater than or equal to comparison based on bin height.

        Parameters
        ----------
        other : Extremum, float, int, or np.ndarray
            The object to compare with.

        Returns
        -------
        bool or np.ndarray
            The result of the comparison.
        """
        match other:
            case other if type(other) == type(self):
                return self.bin >= other.bin
            case int() | float() | np.ndarray():
                return self.bin >= other

            
class Pond:
    """
    Represents a pond defined by start and end peaks, ground, and a threshold.

    Parameters
    ----------
    start_peak : Extremum
        The starting peak of the pond.
    end_peak : Extremum
        The ending peak of the pond.
    ground : Ground
        The ground on which the pond is situated.
    threshold : float, optional
        The threshold for determining if the pond is a lowland, by default 0.5.
    """

    def __init__(self, start_peak: 'Extremum', end_peak: 'Extremum', ground: 'Ground', threshold: float = 0.5) -> None:
        self.start_peak = start_peak
        self.end_peak = end_peak
        self.ground = ground
        self.threshold = threshold
        self.lowland = None
        self.merged = False
        self.get_underwater_ground()
        self.check_lowland()

    def get_underwater_ground(self) -> None:
        """
        Get the part of the ground that is underwater.
        """
        self.ground_underwater = self.ground[(self.start_peak.height > self.ground.get_bins_height()) & 
                                             (self.end_peak.height > self.ground.get_bins_height())]    

    def get_water_and_underwater_area(self) -> None:
        """
        Calculate the water and underwater area of the pond.
        """
        self.level = min(self.start_peak.height, self.end_peak.height)
        self.underwater_area = self.ground_underwater.get_area()
        self.water_area = self.ground_underwater.get_water_area(self.level)
    
    def check_lowland(self) -> None:
        """
        Check if the pond is a lowland based on the water and underwater areas.
        """
        self.get_water_and_underwater_area()
        if self.water_area / (self.water_area + self.underwater_area) >= self.threshold:
            self.lowland = True
        else:
            self.lowland = False
    
    def is_lowland(self) -> bool:
        """
        Determine if the pond is a lowland.

        Returns
        -------
        bool
            True if the pond is a lowland, False otherwise.
        """
        return self.lowland
    
    def merge(self, other: 'Pond') -> None:
        """
        Merge another pond into this pond.

        Parameters
        ----------
        other : Pond
            The other pond to be merged.
        """
        self.ground.add_bin(self.end_peak)
        self.end_peak = other.end_peak
        self.ground.merge(other.ground)
        self.get_underwater_ground()
        self.check_lowland()
        other.merged = True

class ExtremumArray:
    """
    A class representing an array of Extremum objects.

    Parameters
    ----------
    array : Iterable[Extremum]
        An iterable of Extremum objects.
    """

    def __init__(self, array: Iterable['Extremum']) -> None:
        self.array = np.array(array)
        self.shape = self.array.shape

    def __getitem__(self, idxs) -> Union['Extremum' , 'ExtremumArray']:
        """
        Get an item or a subarray from the ExtremumArray.

        Parameters
        ----------
        idxs : int, slice, or array-like
            The indices to retrieve.

        Returns
        -------
        Extremum or ExtremumArray
            The requested item or subarray.
        """
        out_arr = self.array[idxs]
        match out_arr:
            case Extremum():
                out = out_arr
            case np.ndarray():
                out = ExtremumArray(self.array[idxs])
        return out
    
    def __lt__(self, other: Union['ExtremumArray', np.ndarray, float, int]) -> np.ndarray:
        """
        Check if elements in the array are less than the given value.

        Parameters
        ----------
        other : ExtremumArray, np.ndarray, float, or int
            The value to compare against.

        Returns
        -------
        np.ndarray
            Boolean array of comparison results.
        """
        match other:
            case other if type(other) == type(self):
                less_fn = np.vectorize(lambda x1, x2: x1.height < x2.height)
            case Extremum():
                less_fn = np.vectorize(lambda x1, x2: x1.height < x2.height)
            case int() | float():
                less_fn = np.vectorize(lambda x1, x2: x1.height < x2)
        return less_fn(self.array, other)
    
    def __gt__(self, other: Union['ExtremumArray', np.ndarray, float, int]) -> np.ndarray:
        """
        Check if elements in the array are greater than the given value.

        Parameters
        ----------
        other : ExtremumArray, np.ndarray, float, or int
            The value to compare against.

        Returns
        -------
        np.ndarray
            Boolean array of comparison results.
        """
        match other:
            case other if type(other) == type(self):
                less_fn = np.vectorize(lambda x1, x2: x1.height > x2.height)
            case Extremum():
                less_fn = np.vectorize(lambda x1, x2: x1.height > x2.height)
            case int() | float():
                less_fn = np.vectorize(lambda x1, x2: x1.height > x2)
        return less_fn(self.array, other)
    
    def __le__(self, other: Union['ExtremumArray', np.ndarray, float, int]) -> np.ndarray:
        """
        Check if elements in the array are less than or equal to the given value.

        Parameters
        ----------
        other : ExtremumArray, np.ndarray, float, or int
            The value to compare against.

        Returns
        -------
        np.ndarray
            Boolean array of comparison results.
        """
        match other:
            case other if type(other) == type(self):
                less_fn = np.vectorize(lambda x1, x2: x1.height <= x2.height)
            case Extremum():
                less_fn = np.vectorize(lambda x1, x2: x1.height <= x2.height)
            case int() | float():
                less_fn = np.vectorize(lambda x1, x2: x1.height <= x2)
        return less_fn(self.array, other)
    
    def __ge__(self, other: Union['ExtremumArray', np.ndarray, float, int]) -> np.ndarray:
        """
        Check if elements in the array are greater than or equal to the given value.

        Parameters
        ----------
        other : ExtremumArray, np.ndarray, float, or int
            The value to compare against.

        Returns
        -------
        np.ndarray
            Boolean array of comparison results.
        """
        match other:
            case other if type(other) == type(self):
                less_fn = np.vectorize(lambda x1, x2: x1.height >= x2.height)
            case Extremum():
                less_fn = np.vectorize(lambda x1, x2: x1.height >= x2.height)
            case int() | float():
                less_fn = np.vectorize(lambda x1, x2: x1.height >= x2)
        return less_fn(self.array, other)
    
    def __len__(self) -> int:
        """
        Get the length of the ExtremumArray.

        Returns
        -------
        int
            The length of the array.
        """
        return len(self.array)
    
    def signs(self) -> np.ndarray:
        """
        Get the signs of the extremum objects in the array.

        Returns
        -------
        np.ndarray
            Array of signs.
        """
        signs_fn = np.vectorize(lambda x: x.sign)
        return signs_fn(self.array)
    
    def diff(self, dim: str = 'height', n: int = 1) -> 'ExtremumArray':
        """
        Compute the n-th discrete difference along the specified dimension.

        Parameters
        ----------
        dim : str, optional
            The dimension along which the difference is computed, by default 'height'.
        n : int, optional
            The number of times values are differenced, by default 1.

        Returns
        -------
        ExtremumArray
            The array of differences as ExtremumArray.
        """
        match dim:
            case 'start':
                vec_start = np.vectorize(lambda x: x.start)
                start_diff = np.diff(vec_start(self.array), n=n)
                vec_diff_arr = np.vectorize(lambda x1, x2: Extremum(Bin(x2, x1.width, x1.height), x1.sign))
                return ExtremumArray(vec_diff_arr(self[1:].array, start_diff))
            case 'height':
                vec_height = np.vectorize(lambda x: x.height)
                height_diff = np.diff(vec_height(self.array), n=n)
                vec_diff_arr = np.vectorize(lambda x1, x2: Extremum(Bin(x2.start, x1.width, x2), x1.sign))
                return ExtremumArray(vec_diff_arr(self[1:].array, height_diff))
    
    def get_bins_width(self) -> np.ndarray:
        """
        Get the widths of the bins in the array.

        Returns
        -------
        np.ndarray
            Array of widths.
        """
        width_fn: Callable[[Bin], float] = lambda x: x.width
        width_fn_vec = np.vectorize(width_fn)
        return width_fn_vec(self.array)
    
    def get_bins_height(self) -> np.ndarray:
        """
        Get the heights of the bins in the array.

        Returns
        -------
        np.ndarray
            Array of heights.
        """
        height_fn: Callable[[Bin], float] = lambda x: x.height
        height_fn_vec = np.vectorize(height_fn)
        return height_fn_vec(self.array)
    
    def get_bins_start(self) -> np.ndarray:
        """
        Get the start positions of the bins in the array.

        Returns
        -------
        np.ndarray
            Array of start positions.
        """
        start_fn: Callable[[Extremum], float] = lambda x: x.start
        start_fn_vec = np.vectorize(start_fn)
        return start_fn_vec(self.array)
    
    def argsort(self, order: str = 'start') -> np.ndarray:
        """
        Get the indices that would sort the array.

        Parameters
        ----------
        order : str, optional
            The order to sort by ('start' or 'height'), by default 'start'.

        Returns
        -------
        np.ndarray
            Array of indices that would sort the array.
        """
        match order:
            case 'start':
                vec_start = np.vectorize(lambda x: x.start)
                return vec_start(self.array).argsort()
            case 'height':
                vec_height = np.vectorize(lambda x: x.height)
                return vec_height(self.array).argsort()
