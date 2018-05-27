import cv2
import numpy as np
from collections import defaultdict
import torch

class MulticlassWrapper(object):

	def __init__(self, img):

		channels = defaultdict(lambda:np.zeros(img.shape[:2], dtype=np.uint8))
		
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				channels[tuple(img[i, j, :])][i, j] = 255


		self.colors = list(filter(lambda x: x != (0, 0, 0), list(channels.keys())))
		self.n_channels = len(self.colors)

	def img_to_channels(self, img):
		channels = defaultdict(lambda:np.zeros(img.shape[:2], dtype=np.uint8))
		
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				channels[tuple(img[i, j, :])][i, j] = 255

		output_channels = np.zeros((img.shape[0], img.shape[1], len(self.colors)))
		for i, color in enumerate(self.colors):
			output_channels[:, :, i] = channels[color]

		return output_channels

	def channels_to_img(self, channels):

		img = np.zeros((channels.shape[0], channels.shape[1], 3), dtype=np.float)
		for i in range(len(self.colors)):
			img += channels[:, :, i][:, :, None] * np.array(self.colors[i])[None, None, :]

		return img

