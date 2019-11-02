import numpy as np
import sys
from datetime import datetime
sys.path.append("game/")

from coord import CoordinateChannel2D

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Activation, Input, Concatenate, Reshape, Permute, Average, Add
from keras.layers import Conv2D, ReLU, BatchNormalization, LeakyReLU, Softmax
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
from keras.optimizers import RMSprop, Adam, SGD
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import LearningRateScheduler, History
from keras.datasets import mnist, cifar10
import tensorflow as tf

from keras.layers import Lambda
from keras.backend import slice

import pygame
from pygame import Rect, Color
import canvas as game

import scipy.misc
import scipy.stats as st

import threading
import time
import math

import matplotlib.pyplot as plt

GAMMA = 0.99                # discount value
LAMBDA = 0.99               # bias variance trade off (high Lambda = high variance, low bias)
BETA = 0.01                 # regularisation coefficient
VAR_SCALE = 1
GP_LAMBDA = 1
IMAGE_ROWS = 28
IMAGE_COLS = 28
TIME_SLICES = 2
NUM_ACTIONS = 5
IMAGE_CHANNELS = 1
LEARNING_RATE_RL = 1e-4
LEARNING_RATE_CRITC = 1e-4
LOSS_CLIPPING = 0.2
FRAMES_TO_PRETRAIN_DISC = 0
TIME_SIG_GAIN = 1
INPUT_GAIN = 1
LEARNING_RATE_DISC = 5e-5
# TEMPERATURE = 0
# TEMP_INCR = 1e-6

EPOCHS = 3
THREADS = 16
T_MAX = 16
BATCH_SIZE = 32
T = 0
EPISODE = 0
HORIZON = 512
# MSE_TERM_THRESHOLD = 1 # This is adjusted dynamically and only applied when MSE gets worse after stepping the simulation
# R_TERM_THRESHOLD = -100
# MSE_UPPER_BOUND_OFFSET = 0.1
MSE_TERM_PROB = 1 # Probability of terminating when MSE is above threshold
WARMUP_TIME = 0 # Warmup time before the agent can be terminated

(real, _), (_, _) = mnist.load_data()
real = real.reshape(-1, IMAGE_COLS, IMAGE_ROWS, IMAGE_CHANNELS)
if IMAGE_CHANNELS < 3:
	real = np.mean(real, axis=-1, keepdims=True)
img_mean = np.mean(real, axis=(0, 1, 2))
img_std = np.std(real, axis=(0, 1, 2))
real = (real - img_mean) / img_std
white_img = (np.array([[[[255] * IMAGE_CHANNELS]]]) - img_mean) / img_std
black_img = (np.array([[[[0] * IMAGE_CHANNELS]]]) - img_mean) / img_std
max_mse = np.mean(np.square(white_img - black_img))

episode_r = np.empty((0, 1), dtype=np.float32)
episode_time = np.zeros((0, 1))
episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS * TIME_SLICES))
episode_refs = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
episode_action = np.empty((0, NUM_ACTIONS), dtype=np.float32)
episode_pred = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
episode_critic = np.empty((0, 1), dtype=np.float32)

now = datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.FileWriter("logs/" + now, tf.get_default_graph())
# tb_callback = keras.callbacks.TensorBoard(log_dir='logs/softplus_glorot/', histogram_freq=1, write_grads=True, write_graph=True)

DUMMY_ADVANTAGE = np.zeros((1, 1))
DUMMY_OLD_PRED  = np.zeros((1, NUM_ACTIONS * 2))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#loss function for policy output
def logloss(advantage):     #policy loss
	def logloss_impl(y_true, y_pred):
		mu = y_pred[:,:NUM_ACTIONS]
		sigma_sq = y_pred[:,NUM_ACTIONS:]
		pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(y_true - mu) / (2. * sigma_sq))
		log_pdf = K.log(pdf + K.epsilon())
		aloss = -K.mean(log_pdf * advantage)

		entropy = 0.5 * (K.log(2. * np.pi * sigma_sq + K.epsilon()) + 1.)
		entropy_bonus = -BETA * K.mean(entropy)

		return aloss + entropy_bonus
	return logloss_impl

def ppo_loss(advantage, new_pi, old_pi):
	def loss(y_true, y_pred):
		mu_pi = new_pi[:,:NUM_ACTIONS]
		var_pi = new_pi[:,NUM_ACTIONS:]
		mu_old_pi = old_pi[:,:NUM_ACTIONS]
		var_old_pi = old_pi[:,NUM_ACTIONS:]
		denom = K.sqrt(2 * np.pi * var_pi)
		denom_old = K.sqrt(2 * np.pi * var_old_pi)
		likelihood = K.exp(- K.square(y_true - mu_pi) / (2 * var_pi + 1e-10))
		old_likelihood = K.exp(- K.square(y_true - mu_old_pi) / (2 * var_old_pi + 1e-10))

		# pdf = lambda x: K.exp(- K.square(x - K.expand_dims(mu_pi)) / (2 * K.expand_dims(var_pi))) / K.sqrt(2 * np.pi * K.expand_dims(var_pi))

		likelihood = likelihood/(denom + 1e-10)
		old_likelihood = old_likelihood/(denom_old + 1e-10)
		r = likelihood/(old_likelihood + 1e-10)

		surr1 = r * advantage
		surr2 = K.clip(r, (1 - LOSS_CLIPPING), (1 + LOSS_CLIPPING)) * advantage
		aloss = -K.mean(K.minimum(surr1, surr2))

		entropy = 0.5 * (K.log(2. * np.pi * var_pi + K.epsilon()) + 1.)
		entropy_bonus = -BETA * K.mean(entropy)

		# out_of_bounds_penalty = K.mean(K.abs(mu_pi) * K.cast(K.abs(mu_pi) > 1, 'float32'))
		# out_of_bounds_penalty += K.mean(K.abs(var_pi) * K.cast(K.abs(var_pi) > 1, 'float32'))

		# shape = [K.shape(y_pred)[0], NUM_ACTIONS]
		# eps = K.random_normal(shape)
		# actions = mu_pi + K.sqrt(var_pi) * eps
		# sq_actions = K.square(actions)
		# invalid_action_penalty = 0.5 * K.mean(K.cast(sq_actions > 1, 'float32') * (sq_actions - 1))
		# energy_penalty = -0.1 * K.mean(K.square(actions))
		# var_bonus = -K.mean(K.square(actions - K.mean(actions, axis=0, keepdims=True)))

		return aloss + entropy_bonus# + invalid_action_penalty# + var_bonus# + energy_penalty# + out_of_bounds_penalty
	return loss

def gp_loss(interp_x):
	def gp_loss_internal(y_true, y_pred):
		gradients = K.gradients(y_pred, interp_x)[0]
		gradients_sqr = K.square(gradients)
		gradients_sqr_sum = K.sum(gradients_sqr, axis = np.arange(1, len(gradients_sqr.shape)))
		gradient_l2_norm = K.sqrt(gradients_sqr_sum)
		gradient_penalty = GP_LAMBDA * K.square(gradient_l2_norm)
		gradient_penalty = K.mean(gradient_penalty)

		return gradient_penalty
	return gp_loss_internal

def wasserstein_loss(y_true, y_pred):
	return -K.mean(y_true * y_pred) # -(D(X) - D(G))

def normalized_mse(y_true, y_pred):
	mean = K.mean(y_true)
	std  = K.std(y_true) + K.epsilon()
	normalized_y_true = (y_true - mean) / std
	normalized_y_pred = (y_pred - mean) / std
	return K.mean(K.square(normalized_y_true - normalized_y_pred))

#loss function for critic output
# def sumofsquares(y_true, y_pred):        #critic loss
# 	return K.mean(K.square(y_pred - y_true))

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class TrainablePosParamLike(Layer):
	def __init__(self, **kwargs):
		super(TrainablePosParamLike, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='kernel',
										shape=(1, 1),
										initializer=keras.layers.initializers.constant(0.1),
										trainable=False)
		self.tuner = self.add_weight(name='tuner',
										shape=(1, 1),
										initializer='random_uniform',
										trainable=True)
		super(TrainablePosParamLike, self).build(input_shape)

	def call(self, x, **kwargs):
		return K.tile(keras.layers.activations.sigmoid(self.tuner) * self.kernel, K.shape(x))

class AddNoise(Layer):
	def __init__(self, **kwargs):
		super(AddNoise, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.

		self.noise_scale = self.add_weight(name='noise_scale',
										shape=[input_shape[-1]],
										initializer='zeros',
										trainable=True)
		super(AddNoise, self).build(input_shape)

	def call(self, x, **kwargs):
		noise = K.random_normal(K.concatenate([K.shape(x)[:-1], [1]]))
		return x + noise * K.reshape(self.noise_scale, [1] * (K.ndim(x) - 1) + [-1])

def dot(x):
	a = x[0]
	b = x[1]
	return K.batch_dot(a, b)

#function buildmodel() to define the structure of the neural network in use 
def buildmodel():
	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS * TIME_SLICES, ), name = 'Input')
	Ref = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Ref_Image')

	In = Concatenate()([S, Ref])
	In = Lambda(lambda x: x * INPUT_GAIN)(In)

	from_color = CoordinateChannel2D()(In)
	from_color = Conv2D(128, kernel_size = (1,1), strides = (1,1), activation = 'linear')(from_color)

	Q   = Reshape([-1, 128])(Conv2D(128, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(from_color))
	Key = Reshape([-1, 128])(Conv2D(128, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(from_color))
	Key = Permute([2, 1])(Key)
	V   = Reshape([-1, 128])(Conv2D(128, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(from_color))
	QK = Lambda(dot)([Q, Key])
	a = Softmax()(QK)
	atten = Reshape(from_color._keras_shape[1:])(Lambda(dot)([a, V]))
	from_color = atten
	
	h0 = CoordinateChannel2D()(from_color)
	h0 = Conv2D(256, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h0)
	h0 = LeakyReLU(alpha=0.2)(h0)

	Q   = Reshape([-1, 256])(Conv2D(256, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h0))
	Key = Reshape([-1, 256])(Conv2D(256, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h0))
	Key = Permute([2, 1])(Key)
	V   = Reshape([-1, 256])(Conv2D(256, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h0))
	QK = Lambda(dot)([Q, Key])
	a = Softmax()(QK)
	atten = Reshape(h0._keras_shape[1:])(Lambda(dot)([a, V]))
	h0 = atten

	h1 = CoordinateChannel2D()(h0)
	h1 = Conv2D(512, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h1)
	h1 = LeakyReLU(alpha=0.2)(h1)

	Q   = Reshape([-1, 512])(Conv2D(512, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h1))
	Key = Reshape([-1, 512])(Conv2D(512, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h1))
	Key = Permute([2, 1])(Key)
	V   = Reshape([-1, 512])(Conv2D(512, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h1))
	QK = Lambda(dot)([Q, Key])
	a = Softmax()(QK)
	atten = Reshape(h1._keras_shape[1:])(Lambda(dot)([a, V]))
	h1 = atten

	h2 = CoordinateChannel2D()(h1)
	h2 = Conv2D(512, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h2)
	h2 = LeakyReLU(alpha=0.2)(h2)

	h2 = Flatten()(h2)

	h3 = Dense(512, kernel_initializer = 'he_uniform') (h2)
	h3 = LeakyReLU(alpha=0.2)(h3)

	PI_mu = Dense(NUM_ACTIONS, activation = 'tanh', kernel_initializer = 'glorot_uniform') (h3)
	PI_var = Dense(NUM_ACTIONS, activation = 'sigmoid', kernel_initializer = 'glorot_uniform') (h3)
	PI_var = Lambda(lambda x: x * VAR_SCALE)(PI_var)
	PI = Concatenate(name = 'PI')([PI_mu, PI_var])

	Act = Lambda(lambda x: x[:,:NUM_ACTIONS] + K.sqrt(x[:,NUM_ACTIONS:]) * K.random_normal([K.shape(x)[0], NUM_ACTIONS]), name = 'o_Act')(PI)
	
	A = Input(shape = (1,), name = 'Advantage')
	PI_old = Input(shape = (NUM_ACTIONS * 2,), name = 'Old_Prediction')
	model = Model(inputs = [S,Ref,A,PI_old], outputs = Act)
	# optimizer = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	# optimizer = Adam(lr = LEARNING_RATE_RL)
	optimizer = SGD(LEARNING_RATE_RL, momentum = 0.9)
	model.compile(loss = ppo_loss(A, PI, PI_old), optimizer = optimizer)
	return model

def build_critic():
	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS * TIME_SLICES, ), name = 'Input')
	Ref = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Ref_Image')
	T = Input(shape = (1,), name = 'Time_Signal')
	
	In = Concatenate()([S, Ref])
	In = Lambda(lambda x: x * INPUT_GAIN)(In)

	from_color = CoordinateChannel2D()(In)
	from_color = Conv2D(128, kernel_size = (1,1), strides = (1,1), activation = 'linear')(from_color)

	Q   = Reshape([-1, 128])(Conv2D(128, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(from_color))
	Key = Reshape([-1, 128])(Conv2D(128, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(from_color))
	Key = Permute([2, 1])(Key)
	V   = Reshape([-1, 128])(Conv2D(128, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(from_color))
	QK = Lambda(dot)([Q, Key])
	a = Softmax()(QK)
	atten = Reshape(from_color._keras_shape[1:])(Lambda(dot)([a, V]))
	from_color = atten
	
	h0 = CoordinateChannel2D()(from_color)
	h0 = Conv2D(256, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h0)
	h0 = LeakyReLU(alpha=0.2)(h0)

	Q   = Reshape([-1, 256])(Conv2D(256, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h0))
	Key = Reshape([-1, 256])(Conv2D(256, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h0))
	Key = Permute([2, 1])(Key)
	V   = Reshape([-1, 256])(Conv2D(256, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h0))
	QK = Lambda(dot)([Q, Key])
	a = Softmax()(QK)
	atten = Reshape(h0._keras_shape[1:])(Lambda(dot)([a, V]))
	h0 = atten

	h1 = CoordinateChannel2D()(h0)
	h1 = Conv2D(512, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h1)
	h1 = LeakyReLU(alpha=0.2)(h1)

	Q   = Reshape([-1, 512])(Conv2D(512, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h1))
	Key = Reshape([-1, 512])(Conv2D(512, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h1))
	Key = Permute([2, 1])(Key)
	V   = Reshape([-1, 512])(Conv2D(512, kernel_size = (1,1), strides = (1,1), padding = 'same', bias_initializer = 'he_uniform')(h1))
	QK = Lambda(dot)([Q, Key])
	a = Softmax()(QK)
	atten = Reshape(h1._keras_shape[1:])(Lambda(dot)([a, V]))
	h1 = atten

	h2 = CoordinateChannel2D()(h1)
	h2 = Conv2D(512, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h2)
	h2 = LeakyReLU(alpha=0.2)(h2)

	h2 = Flatten()(h2)
	h2 = Concatenate()([h2, T])

	h3 = Dense(512, kernel_initializer = 'he_uniform') (h2)
	h3 = LeakyReLU(alpha=0.2)(h3)
	
	C = Dense(1, activation = 'linear')(h3)

	model = Model(inputs = [S, Ref, T], outputs = C)
	# optimizer = Adam(lr = LEARNING_RATE_CRITC)
	optimizer = SGD(LEARNING_RATE_CRITC, momentum = 0.9)
	model.compile(loss = 'mse', optimizer = optimizer)
	return model

def build_discriminator():
	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Input')

	from_color = CoordinateChannel2D()(S)
	from_color = Conv2D(32, kernel_size = (1,1), strides = (1,1), activation = 'linear')(from_color)
	
	h0 = CoordinateChannel2D()(from_color)
	h0 = Conv2D(32, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer = 'he_uniform')(h0)
	h0 = LeakyReLU(alpha=0.2)(h0)
	h0 = CoordinateChannel2D()(h0)
	h0 = Conv2D(32, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h0)
	h0 = LeakyReLU(alpha=0.2)(h0)
	h0 = CoordinateChannel2D()(h0)
	h0 = Conv2D(32, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer = 'he_uniform')(h0)
	h0 = LeakyReLU(alpha=0.2)(h0)

	h1 = CoordinateChannel2D()(h0)
	h1 = Conv2D(64, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h1)
	h1 = LeakyReLU(alpha=0.2)(h1)
	h1 = CoordinateChannel2D()(h1)
	h1 = Conv2D(64, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer = 'he_uniform')(h1)
	h1 = LeakyReLU(alpha=0.2)(h1)

	h2 = CoordinateChannel2D()(h1)
	h2 = Conv2D(128, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'he_uniform')(h2)
	h2 = LeakyReLU(alpha=0.2)(h2)
	h2 = CoordinateChannel2D()(h2)
	h2 = Conv2D(128, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer = 'he_uniform')(h2)
	h2 = LeakyReLU(alpha=0.2)(h2)
	h2 = Flatten()(h2)

	h3 = Dense(512, kernel_initializer = 'he_uniform') (h2)
	h3 = LeakyReLU(alpha=0.2)(h3)
	h3 = Dense(512, kernel_initializer = 'he_uniform') (h3)
	h3 = LeakyReLU(alpha=0.2)(h3)
	h3 = Dense(512, kernel_initializer = 'he_uniform') (h3)
	h3 = LeakyReLU(alpha=0.2)(h3)
	h3 = Dense(512, kernel_initializer = 'he_uniform') (h3)
	h3 = LeakyReLU(alpha=0.2)(h3)

	out = Dense(1, activation = 'linear')(h3)

	D = Model(inputs = S, outputs = out)

	real = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Real_Input')
	fake = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Fake_Input')
	interp = RandomWeightedAverage()([real, fake])

	real_out = Lambda(lambda x: x, name = 'real_out')(D(real))
	fake_out = Lambda(lambda x: x, name = 'fake_out')(D(fake))
	interp_out = Lambda(lambda x: x, name = 'interp_out')(D(interp))

	optimizer = Adam(lr = LEARNING_RATE_DISC)
	model_train = Model(inputs = [real, fake], outputs = [real_out, fake_out, interp_out])
	model_train.compile(loss = {'real_out': wasserstein_loss, 'fake_out': wasserstein_loss, 'interp_out': gp_loss(interp)}, optimizer = optimizer)
	return D, model_train

#function to preprocess an image before giving as input to the neural network
def preprocess(image):
	global img_mean
	global img_std
	
	if image.ndim == 3:
		image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
	
	if image.shape[-1] > IMAGE_CHANNELS:
		image = np.mean(image, axis=-1, keepdims=True)

	return (image - img_mean) / img_std

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def sigmoid(X):
   return 1/(1+np.exp(-X))

# initialize a new model using buildmodel() or use load_model to resume training an already trained model
model = buildmodel()
critic = build_critic()
discriminator, D_train = build_discriminator()
PI = Model(inputs=model.input, outputs=model.get_layer('PI').output)
# model.load_weights("saved_models/model_updates10000")
model._make_predict_function()
critic._make_predict_function()
discriminator._make_predict_function()
PI._make_predict_function()

game_state = []
# prev_disc = []
best_state = []
# max_score = 0
mse_upper_bound_offset = []
mse_upper_bound_center = []
mse_history = []
for i in range(0,THREADS):
	game_state.append(game.GameState(1000000))
	current_frame = game_state[i].get_current_frame()
	current_frame = preprocess(current_frame)
	mse_upper_bound_offset.append(None)
	mse_upper_bound_center.append(None)
	mse_history.append([])
	# prev_disc.append(discriminator.predict(current_frame)[0][0])

def runprocess(thread_id, s_t, ref_image):
	global T
	global model
	global critic
	global discriminator
	# global max_score

	t = 0
	t_start = t
	terminal = False
	r_t = 0
	r_store = np.empty((0, 1), dtype=np.float32)
	state_store = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS * TIME_SLICES))
	refs_store = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	action_store = np.empty((0, NUM_ACTIONS), dtype=np.float32)
	time_store = np.zeros((0, 1))
	pred_store = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
	critic_store = np.empty((0, 1), dtype=np.float32)

	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
	ref_image = np.expand_dims(ref_image, 0)

	start_img = np.array([[[game.BACKGROUND_COLOR]]])
	if start_img.shape[-1] > IMAGE_CHANNELS:
		start_img = np.mean(start_img, axis=-1, keepdims=True)
	mse_0 = np.mean(np.square(ref_image - start_img))

	while t-t_start < T_MAX and terminal == False:
		t += 1
		T += 1

		time = game_state[thread_id].frame_count
		time_sig = TIME_SIG_GAIN * np.array(time / (HORIZON - 1) - 0.5).reshape(1, 1)

		mse_t = np.mean(np.square(ref_image - s_t[:,:,:, 0:ref_image.shape[3]]))
		if len(mse_history[thread_id]) == 0:
			mse_history[thread_id].append(mse_t)
			mse_mean = np.mean(mse_history[thread_id])
			mse_std = np.std(mse_history[thread_id])

		# Save best state so far
		if (mse_t < best_state[thread_id]['mse']):
			best_state[thread_id]['frame'] = s_t[0, ..., 0:IMAGE_CHANNELS] * img_std + img_mean
			best_state[thread_id]['ref_image'] = ref_image
			best_state[thread_id]['mse'] = mse_t

		critic_reward = critic.predict([s_t, ref_image, time_sig])
		pi = PI.predict([s_t, ref_image, DUMMY_ADVANTAGE, DUMMY_OLD_PRED])
		actions = model.predict([s_t, ref_image, DUMMY_ADVANTAGE, DUMMY_OLD_PRED])[0]

		x_t = game_state[thread_id].frame_step(actions)

		x_t = preprocess(x_t)

		mse_t_1 = np.mean(np.square(ref_image - x_t))
		r_t = (mse_t - mse_t_1) / mse_0

		# Calculate MSE termination threshold from moving average of MSE mean and standard deviation
		center_smoothing = 0
		offset_smoothing = 1 / (1 + time) # Unbias offset at start of rollout
		offset_scaling = 1
		mse_history[thread_id].append(mse_t_1)
		mse_mean = np.mean(mse_history[thread_id])
		mse_std = np.std(mse_history[thread_id])
		old_center = mse_upper_bound_center[thread_id]
		old_offset = mse_upper_bound_offset[thread_id]
		if time == 0 and mse_upper_bound_center[thread_id] == None:
			mse_upper_bound_center[thread_id] = mse_mean
			mse_upper_bound_offset[thread_id] = 1 # Start with a large upper bound and get smaller
		elif time == 0:
			mse_upper_bound_center[thread_id] = mse_mean # Snap center on reset
			mse_upper_bound_offset[thread_id] = old_offset # Keep offset on reset
		else:
			new_center = mse_mean
			new_offset = offset_scaling * mse_std
			mse_upper_bound_center[thread_id] = center_smoothing * old_center + (1 - center_smoothing) * new_center
			mse_upper_bound_offset[thread_id] = offset_smoothing * old_offset + (1 - offset_smoothing) * new_offset
		mse_upper_bound = mse_upper_bound_center[thread_id] + mse_upper_bound_offset[thread_id]

		# Restrict movement during rollout
		if time != 0:
			old_mse_upper_bound = old_center + old_offset
			# Prevent moving bound upwards during rollout
			mse_upper_bound = min(mse_upper_bound, old_mse_upper_bound)
			# Allow moving upwards if our center gets higher than our upper bound
			# mse_upper_bound = max(mse_upper_bound, mse_upper_bound_center[thread_id])
			mse_upper_bound_offset[thread_id] = mse_upper_bound - mse_upper_bound_center[thread_id]

		# Early termination conditions
		# mse_ref_to_black = np.mean(np.square(ref_image - black_img))
		if (r_t == 0) or ((time + 1) == HORIZON): # np.all(x_t == s_t[..., 0:IMAGE_CHANNELS]) or 
			terminal = True
		elif ((time + 1) >= WARMUP_TIME) and (mse_t_1 > mse_upper_bound):
			terminal = np.random.random() < MSE_TERM_PROB

		actions = np.reshape(actions, (1, -1))
		pi = np.reshape(pi, (1, -1))
		critic_reward = np.reshape(critic_reward, (1, -1))

		r_store = np.append(r_store, [[r_t] * 1], axis = 0)
		state_store = np.append(state_store, s_t, axis = 0)
		refs_store = np.append(refs_store, ref_image, axis = 0)
		action_store = np.append(action_store, actions, axis=0)
		time_store = np.append(time_store, time_sig, axis = 0)
		pred_store = np.append(pred_store, pi, axis = 0)
		critic_store = np.append(critic_store, critic_reward, axis=0)
		
		s_t = np.append(x_t, s_t[..., :-IMAGE_CHANNELS], axis = -1)
		print("Frame = " + str(T) + ", Updates = " + str(EPISODE) + ", Thread = " + str(thread_id) + ", Action = " + str(actions) + ", Output = "+ str(pi))
	
	episode_end_t = len(r_store)-1
	if not terminal:
		time_t_1 = time + 1
		time_sig_t_1 = TIME_SIG_GAIN * np.array(time_t_1 / (HORIZON - 1) - 0.5).reshape(1, 1)
		v_t_1 = critic.predict([s_t, ref_image, time_sig_t_1])
		r_store[episode_end_t] = r_t + GAMMA * v_t_1
	else:
		# On termination we have three options for reset
		# 1 reset to blank canvas
		# 2 reset to a random state in current memory
		# 3 reset to the best previous state for this thread
		choice = np.random.choice(['hard', 'memory', 'best'])
		if choice == 'hard':
			game_state[thread_id].reset()
			# Also pick a new random reference image for a hard reset
			ref_image = np.expand_dims(real[np.random.randint(0, real.shape[0])], 0) 
		elif choice == 'memory':
			memory = np.append(episode_state, state_store, axis = 0)
			ref_memory = np.append(episode_refs, refs_store, axis = 0)
			returns = np.append(episode_r, r_store, axis = 0)
			# returns = (returns - np.mean(returns)) / np.std(returns) # Normalize
			returns = np.reshape(returns, [-1])
			indices = np.argsort(returns)[::-1]
			indices = indices[:max(indices.size//4, 1)] # Select from top 25% most valuable
			mem_idx = indices[np.random.randint(0, indices.size)]
			# Prefer hard reset over negative valued states
			if returns[mem_idx] < 0:
				game_state[thread_id].reset()
				ref_image = np.expand_dims(real[np.random.randint(0, real.shape[0])], 0) 
			else:
				# mem_idx = np.random.randint(0, memory.shape[0])
				random_state = memory[mem_idx]
				random_state = random_state[..., 0:IMAGE_CHANNELS]
				random_state = random_state * img_std + img_mean
				game_state[thread_id].reset(random_state)
				# Also restore reference image from memory
				ref_image = np.expand_dims(ref_memory[mem_idx], 0)
		else:
			game_state[thread_id].reset(best_state[thread_id]['frame'])
			# Also restore reference image for best state
			ref_image = best_state[thread_id]['ref_image']
			
		s_t = game_state[thread_id].get_current_frame()
		s_t = preprocess(s_t)
		s_t = np.concatenate([s_t] * TIME_SLICES, axis = -1)

		mse_history[thread_id] = [] # Clear mse history

		r_store[episode_end_t] = r_t # R_t_(1) = r_t + gamma * (V(s_t_1) = 0) when next state (s_t_1) is terminal
	
	# Generalized advantage estimation (Calculating returns R_t for all t in episode, where A_t = R_t - V_t)
	for i in range(2,len(r_store)+1):
		_t = len(r_store) - i
		r_store[_t] = r_store[_t] + LAMBDA * GAMMA * r_store[_t + 1] + (1-LAMBDA) * GAMMA * critic_store[_t + 1]

	return s_t, ref_image, state_store, refs_store, action_store, time_store, pred_store, r_store, critic_store

#function to decrease the learning rate after every epoch. In this manner, the learning rate reaches 0, by 20,000 epochs
def create_lr_fn(lr):
	def step_decay(epoch):
		decay = lr / 20000.
		lrate = lr - epoch*decay
		lrate = max(lrate, 0.)
		return lrate
	return step_decay

class actorthread(threading.Thread):
	def __init__(self,thread_id, s_t, ref_image):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.next_state = s_t
		self.ref_image = ref_image

	def run(self):
		global episode_action
		global episode_pred
		global episode_r
		global episode_critic
		global episode_state
		global episode_time
		global episode_refs

		threadLock.acquire()
		self.next_state, self.ref_image, state_store, refs_store, action_store, time_store, pred_store, r_store, critic_store = runprocess(self.thread_id, self.next_state, self.ref_image)
		self.next_state = self.next_state.reshape(self.next_state.shape[1], self.next_state.shape[2], self.next_state.shape[3])

		episode_r = np.append(episode_r, r_store, axis = 0)
		episode_pred = np.append(episode_pred, pred_store, axis = 0)
		episode_action = np.append(episode_action, action_store, axis = 0)
		episode_time = np.append(episode_time, time_store, axis = 0)
		episode_state = np.append(episode_state, state_store, axis = 0)
		episode_refs = np.append(episode_refs, refs_store, axis = 0)
		episode_critic = np.append(episode_critic, critic_store, axis = 0)

		threadLock.release()

states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS * TIME_SLICES))
ref_images = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))

#initializing state of each thread
for i in range(0, len(game_state)):
	ref_image = np.expand_dims(real[np.random.randint(0, real.shape[0])], 0)
	image = game_state[i].get_current_frame()
	best_state.append({'frame': image, 'mse': 1000, 'ref_image': ref_image}) # Set mse large so that any mse we first encounter will be better
	image = preprocess(image)
	state = np.concatenate([image] * TIME_SLICES, axis=3)
	states = np.append(states, state, axis = 0)
	ref_images = np.append(ref_images, ref_image, axis = 0)

while True:	
	threadLock = threading.Lock()
	threads = []
	for i in range(0,THREADS):
		# Only spawn enough threads to fill the buffer for this episode
		# Can go slightly over
		new_samples_generated = T_MAX * i
		samples_needed = T_MAX * THREADS
		samples_so_far = episode_state.shape[0]
		if (new_samples_generated + samples_so_far) >= samples_needed:
			break

		threads.append(actorthread(i, states[i], ref_images[i]))

	# states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS * TIME_SLICES))
	# ref_images = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))

	for i in range(0,len(threads)):
		threads[i].start()

	#thread.join() ensures that all threads fininsh execution before proceeding further
	for i in range(0,len(threads)):
		threads[i].join()

	pygame.event.clear()

	for i in range(0,len(threads)):
		state = threads[i].next_state
		ref_image = threads[i].ref_image
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+1])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+2])
		# plt.show()
		states[i] = state
		ref_images[i] = ref_image

	# Only update once we have enough samples in memory
	if episode_state.shape[0] >= T_MAX * THREADS:

		# Normalized returns
		ret_mean = np.mean(episode_r)
		# ret_std = np.std(episode_r)
		# episode_r = (episode_r - ret_mean) / ret_std

		#advantage calculation for each action taken
		advantage = episode_r - episode_critic
		adv_mean = np.mean(advantage)
		adv_std = np.std(advantage)
		advantage = (advantage - adv_mean) / (adv_std + 1e-10)

		# Update termination thresholds
		mse = np.mean(np.square(episode_state[:,:,:,0:episode_refs.shape[3]] - episode_refs), axis = (1, 2, 3))
		mse_mean = np.mean(mse)
		mse_std = np.std(mse)
		# MSE_TERM_THRESHOLD = min(MSE_TERM_THRESHOLD, mse_mean + mse_std)
		# R_TERM_THRESHOLD = ret_mean - ret_std
		
		print("backpropagating")

		lr_fn_rl = create_lr_fn(LEARNING_RATE_RL)
		lr_fn_disc = create_lr_fn(LEARNING_RATE_DISC)
		lrate_rl = LearningRateScheduler(lr_fn_rl)
		lrate_disc = LearningRateScheduler(lr_fn_disc)
		callbacks_rl = []
		callbacks_disc = []

		#backpropagation
		# real_episode = real[np.random.randint(low = 0, high = real.shape[0], size = episode_state.shape[0])]
		# positive_y = np.ones((episode_state.shape[0], 1), dtype=np.float32)
		# negative_y = -positive_y
		# dummy_y = np.zeros((episode_state.shape[0], 1), dtype=np.float32)
		# if T < FRAMES_TO_PRETRAIN_DISC:
		# 	disc_hist = D_train.fit([real_episode, episode_state], [positive_y, negative_y, dummy_y], callbacks = callbacks_disc, epochs = EPISODE + EPOCHS, batch_size = BATCH_SIZE, initial_epoch = EPISODE)
		# else:
			# disc_eval = D_train.evaluate([real_episode, episode_state], [positive_y, negative_y, dummy_y], batch_size = episode_state.shape[0])
			# disc_hist = D_train.fit([real_episode, episode_state], [positive_y, negative_y, dummy_y], callbacks = callbacks_disc, epochs = EPISODE + EPOCHS * 3, batch_size = BATCH_SIZE, initial_epoch = EPISODE)
		critic_hist = critic.fit([episode_state, episode_refs, episode_time], episode_r, epochs = EPISODE + EPOCHS, batch_size = BATCH_SIZE, initial_epoch = EPISODE)
		history = model.fit([episode_state, episode_refs, advantage, episode_pred], episode_action, callbacks = callbacks_rl, epochs = EPISODE + EPOCHS, batch_size = BATCH_SIZE, initial_epoch = EPISODE)

		episode_r = np.empty((0, 1), dtype=np.float32)
		episode_pred = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
		episode_action = np.empty((0, NUM_ACTIONS), dtype=np.float32)
		episode_time = np.zeros((0, 1))
		episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS * TIME_SLICES))
		episode_refs = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
		episode_critic = np.empty((0, 1), dtype=np.float32)

		if T >= FRAMES_TO_PRETRAIN_DISC:
			avg_mse_upper_bound_center = np.mean(mse_upper_bound_center)
			avg_mse_upper_bound_offset = np.mean(mse_upper_bound_offset)
			avg_mse_upper_bound = np.mean(np.array(mse_upper_bound_center) + np.array(mse_upper_bound_offset))

			summary = tf.Summary(value=[
				tf.Summary.Value(tag="reward mean", simple_value=float(ret_mean)),
				tf.Summary.Value(tag="action loss", simple_value=float(history.history['loss'][-1])),
				tf.Summary.Value(tag="critic loss", simple_value=float(critic_hist.history['loss'][-1])),
				tf.Summary.Value(tag="mse", simple_value=float(mse_mean)),
				tf.Summary.Value(tag="mse upper bound center - thread {}".format(0), simple_value=float(mse_upper_bound_center[0])),
				tf.Summary.Value(tag="mse upper bound offset - thread {}".format(0), simple_value=float(mse_upper_bound_offset[0])),
				tf.Summary.Value(tag="mse upper bound - thread {}".format(0), simple_value=float(mse_upper_bound_center[0] + mse_upper_bound_offset[0])),
				tf.Summary.Value(tag="mse average upper bound center", simple_value=float(avg_mse_upper_bound_center)),
				tf.Summary.Value(tag="mse average upper bound offset", simple_value=float(avg_mse_upper_bound_offset)),
				tf.Summary.Value(tag="mse average upper bound", simple_value=float(avg_mse_upper_bound)),
				# tf.Summary.Value(tag="mse termination threshold", simple_value=float(MSE_TERM_THRESHOLD))
				# tf.Summary.Value(tag="return termination threshold", simple_value=float(R_TERM_THRESHOLD))
				# tf.Summary.Value(tag="discriminator loss", simple_value=float(disc_eval[0])),
				# tf.Summary.Value(tag="real loss", simple_value=float(disc_eval[1])),
				# tf.Summary.Value(tag="fake loss", simple_value=float(disc_eval[2])),
				# tf.Summary.Value(tag="gradient penalty loss", simple_value=float(disc_eval[3])),
				# tf.Summary.Value(tag="discriminator loss", simple_value=float(disc_hist.history['loss'][-1])),
				# tf.Summary.Value(tag="real loss", simple_value=float(disc_hist.history['real_out_loss'][-1])),
				# tf.Summary.Value(tag="fake loss", simple_value=float(disc_hist.history['fake_out_loss'][-1])),
				# tf.Summary.Value(tag="gradient penalty loss", simple_value=float(disc_hist.history['interp_out_loss'][-1])),
				# tf.Summary.Value(tag="max score", simple_value=float(max_score))
			])
			summary_writer.add_summary(summary, T)

			# if EPISODE % (50 * EPOCHS) == 0: 
			# 	model.save("saved_models/model_updates" +	str(EPISODE)) 
			# 	discriminator.save("saved_models/discrimiator_updates" + str(EPISODE))
			# 	critic.save("saved_models/critic_updates", str(EPISODE))
			EPISODE += EPOCHS
