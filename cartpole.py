"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.friction_cart = 0
        self.friction_pole = 0
        self.Fmax = 10
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
#        self.friction_cart, self.friction_pole = friction
        state = self.state
        x, x_dot, theta, theta_dot = state
        #force = self.force_mag if action==1 else -self.force_mag # change
        force = np.min((np.max((-self.Fmax,action)),self.Fmax)) # action
#        force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp1 = force + self.polemass_length * theta_dot * theta_dot * sintheta
        temp2 = (temp1 - self.friction_cart*np.sign(x_dot)) / self.total_mass
        temp = self.gravity * sintheta - costheta* temp2 - self.friction_pole*(theta_dot/self.polemass_length)
        thetaacc = (temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = (temp1 - self.polemass_length * thetaacc * costheta - self.friction_cart*np.sign(x_dot)) / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done_hit =  bool( x < -self.x_threshold \
                or x > self.x_threshold )
        done_fall = bool( theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians)
        
        if done_hit:
            done = done_hit
            info = 'get out' + [' -',' +'][bool(x > self.x_threshold)]
        elif done_fall:
            done = done_fall
            info = 'fall down'
        else:
            done = False
            info = 'balance'

        reward = np.cos(15*theta/2) - 0.2*np.abs(force-action) #+ ((x/2.4)**2)
        if done_hit == True:
            reward += -10. - np.cos(15*theta/2)
        if done_fall == True:
            reward += -30 - np.cos(15*theta/2)

        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 300
        screen_height = 300

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 90 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
