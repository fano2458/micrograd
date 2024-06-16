import random


class DataLoader:
  def __init__(self, x_data, y_data, batch_size=None, shuffle=False):
    """
    Initializes a DataLoader object for iterating through batches of data.
    Args:
      x_data: A list containing the features (independent variables).
      y_data: A list containing the labels (dependent variables).
      batch_size: The number of samples to include in each batch (default: None, iterates over all data at once).
      shuffle: A boolean indicating whether to shuffle the data before each iteration (default: False).
    """
    self.batch_size = batch_size
    self.x_data = x_data
    self.y_data = y_data
    self.shuffle = shuffle
    self.data = list(zip(self.x_data, self.y_data))  # Combine data
    self.index = 0

  def __iter__(self):
    """
    Returns an iterator over batches of data.
    If shuffle is enabled, shuffles the data before iterating.
    """
    if self.shuffle:
      random.shuffle(self.data)
      
    return self

  def __next__(self):
    """
    Returns the next batch of data.
    Raises:
      StopIteration: If there are no more batches left.
    """
    if self.index >= len(self.data):
      raise StopIteration

    if self.batch_size is None:
      return zip(*self.data)

    batch_data = self.data[self.index: self.index + self.batch_size]
    batch_x, batch_y = zip(*batch_data)

    # Update index for next iteration
    self.index += self.batch_size

    return batch_x, batch_y
