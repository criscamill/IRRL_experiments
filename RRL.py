# -*- coding: utf-8 -*-

#from docopt import docopt
import gym
#import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import tensorflow_probability as tfp
import time

###############################################################################
    
class RRL_actor_disturber_critic:
    """
    Continuous actor-critic reinforcement learning with human feedback:
        Implementation of the actor critic reinforcement learning (RL) in
        continuous action and state space [1] using a neural network architecture
        and Gaussian exploration [2].
    
    References
    ----------
    [1] Kenji Doya. Reinforcement learning in continuous time and space. 
        Neural computation, 12:219–245, 2000.
    [2] Ronald J Williams. Simple statistical gradient-following algorithms for 
        connectionist reinforcement learning. 
        In Reinforcement Learning, pages 5–32. Springer, 1992.
    """
    def __init__(self, **kwargs):    
        self.alpha_theta = kwargs.get('alpha_theta', 0.001)
        self.alpha_upsilon = kwargs.get('alpha_upsilon', 0.0001)
        self.alpha_omega = kwargs.get('alpha_omega', 0.0001)
        self.gamma = kwargs.get('gamma', 0.9)
        self.lamb = kwargs.get('lamb', 0.0)
        
        self.robustness = kwargs.get('robustness', 0.25)
                
        # Network Parameters
        self.exploration = kwargs.get('exploration', 1)
        self.hidden_units = kwargs.get('hidden_units', [50, 20, 20])
        self.state_dim = kwargs.get('state_dim', 4)
        self.disturbance_dim = kwargs.get('disturbance_dim', 2)
        self.noise = kwargs.get('noise', np.identity(self.disturbance_dim,dtype=np.float32))
        
        self.hidden_units_critic = [self.state_dim ,self.hidden_units[0]]
        self.hidden_units_actor = [self.state_dim ,self.hidden_units[1]]
        self.hidden_units_disturber = [self.state_dim,self.hidden_units[2]]

        tf.reset_default_graph()
#        tf.random.set_random_seed(1234) # seed
#        np.random.seed(1234) # seed
        
        # tensorflow Graph input
        self.state, self.value, self.optimizer_critic, self.theta, self.action, self.mean, self.log_policy, self.optimizer_actor, self.action_choose, self.upsilon, self.disturbance, self.mean_disturber, self.log_generator, self.optimizer_disturber, self.disturbance_choose, self.variables_disturber = self._actor_disturber_critic_model_()
        
        with tf.name_scope('critic_training'):
            critic_variables = [var for var in tf.global_variables() if 'Critic' in var.op.name]
            self.theta_grads_and_vars = self.optimizer_critic.compute_gradients(self.value, critic_variables)
            self.theta_gradients = [gdt[0] for gdt in self.theta_grads_and_vars]
            
            self.theta_trace = self._get_e_trace_(self.theta_gradients)
            
            self.theta_grad_placeholder = [(tf.placeholder('float', shape=gdt[0].get_shape()), gdt[1]) 
                for gdt in self.theta_grads_and_vars]
            self.train_critic = self.optimizer_critic.apply_gradients(self.theta_grad_placeholder)
            
        with tf.name_scope('actor_training'):
            actor_variables = [var for var in tf.global_variables() if 'Actor' in var.op.name]
            self.upsilon_grads_and_vars = self.optimizer_actor.compute_gradients(self.log_policy, actor_variables)
            self.upsilon_gradients = [gdu[0] for gdu in self.upsilon_grads_and_vars]
            
            self.upsilon_trace = self._get_e_trace_(self.upsilon_gradients)
            
            self.upsilon_grad_placeholder = [(tf.placeholder('float', shape=gdu[0].get_shape()), gdu[1])
                for gdu in self.upsilon_grads_and_vars]
            self.train_actor = self.optimizer_actor.apply_gradients(self.upsilon_grad_placeholder)
            
        with tf.name_scope('disturber_training'):
            disturber_variables = [var for var in tf.global_variables() if 'Disturber' in var.op.name]
            self.omega_grads_and_vars = self.optimizer_disturber.compute_gradients(self.log_generator, disturber_variables)
            self.omega_gradients = [gdo[0] for gdo in self.omega_grads_and_vars]
            
            self.omega_trace = self._get_e_trace_(self.omega_gradients)
            
            self.omega_grad_placeholder = [(tf.placeholder('float', shape=gdo[0].get_shape()), gdo[1])
                for gdo in self.omega_grads_and_vars]
            self.train_disturber = self.optimizer_disturber.apply_gradients(self.omega_grad_placeholder)
            
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 1
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
    
    def _actor_disturber_critic_model_(self):
        
        state = tf.placeholder('float', [None, self.state_dim], 'state')
        action = tf.placeholder('float', None, 'action')
        disturbance = tf.placeholder('float', [None, self.disturbance_dim], 'disturbance')
#        advice = tf.placeholder('float', None, 'advice')
        
        # Actor part of the algorithm
        with tf.name_scope('Actor'):
            # Variables (Parameters) for the mean of the actions 
            with tf.name_scope('variables'):
                variables_actor = {
                    'upsilon_ker': tf.Variable(tf.random_normal(self.hidden_units_actor), 
                                        name='upsilon_ker'),
                    'upsilon_out': tf.Variable(tf.random_normal([self.hidden_units_actor[1], 1]),
                                       name='upsilon_out'),
                    'upsilon_b_ker': tf.Variable(tf.random_normal([self.hidden_units_actor[1]]),
                                      name='upsilon_b_ker'),
                    'upsilon_b_out': tf.Variable(tf.random_normal([1]), name='upsilon_b_out')
                }
            # Function aproximation for the mean of the actions
            mean = self._multilayer_actor_(state, variables_actor)

            policy = tfp.distributions.Normal(mean, self.exploration)
            log_policy = policy.log_prob(action)
            
            # Training and optimize the policy
            with tf.name_scope('training'):
                rate_scale_williams = self.alpha_upsilon * self.exploration**2
                optimizer_actor = tf.train.GradientDescentOptimizer(rate_scale_williams)
            
            # Drawing action of the policy
            with tf.name_scope('selection'):
                action_choose = tf.squeeze(policy.sample(1))
        
        # Disturber part of the algorithm
        with tf.name_scope('Disturber'):
            # Variables (Parameters) for the mean of the disturbance
            with tf.name_scope('variables'):
                variables_disturber = {
                    'omega_ker': tf.Variable(tf.random_normal(self.hidden_units_disturber), 
                        name='omega_ker'),
                    'omega_out': tf.Variable(tf.random_normal([self.hidden_units_disturber[1], 1]),
                        name='omega_out'),
                    'omega_b_ker': tf.Variable(tf.random_normal([self.hidden_units_disturber[1]]),
                        name='omega_b_ker'),
                    'omega_b_out': tf.Variable(tf.random_normal([1]), name='omega_b_out')
                }
            # Function aproximation for the mean of the disturbance
            mean_disturber = self._multilayer_disturber_(state, variables_disturber)
            
            generator = tfp.distributions.MultivariateNormalFullCovariance(
                    loc = mean_disturber, covariance_matrix = self.noise)
            log_generator = generator.log_prob(disturbance)
            
            # Training and optimize the noise generator
            with tf.name_scope('training'):
                rate_scale_williams_d = self.alpha_omega * tf.linalg.det(self.noise)
                optimizer_disturber = tf.train.GradientDescentOptimizer(rate_scale_williams_d)
                
            # Drawing disturbance of the noise generator
            with tf.name_scope('selection'):
                disturbance_choose = tf.squeeze(generator.sample(1))
        
        # Critic part of the algorithm
        with tf.name_scope('Critic'):
            # Variables (Parameters) of the value function
            with tf.name_scope('variables'):
                variables_critic = {
                    'theta_ker': tf.Variable(tf.random_normal(self.hidden_units_critic), 
                        name='theta_ker'),
                    'theta_out': tf.Variable(tf.random_normal([self.hidden_units_critic[1], 1]),
                        name='theta_out'),
                    'theta_b_ker': tf.Variable(tf.random_normal([self.hidden_units_critic[1]]),
                        name='theta_b_ker'),
                    'theta_b_out': tf.Variable(tf.random_normal([1]), name='theta_b_out')
                }
            # Function aproximation for the state-value function    
            value = self._multilayer_critic_(state, variables_critic)
            
            # Training and optimize the state-value function
            with tf.name_scope('optimizer'):
                optimizer_critic = tf.train.GradientDescentOptimizer(self.alpha_theta)
            
        return state, value, optimizer_critic, variables_critic, action, mean, log_policy, optimizer_actor, action_choose, variables_actor, disturbance, mean_disturber, log_generator, optimizer_disturber, disturbance_choose, variables_disturber
                              
    def _multilayer_critic_(self, state, variables):
        """
        Neural network architecture for the critic.
        
        Parameters
        ----------
        state: tensor
            Input of the network, this represent the current state of the cart-pole.
        variables: dic
            Dictionary with the variables of the neural network.
                
        Returns
        -------
        value: tensor
            Output of the neural network (represent the state-value function).
        """ 
        layer_critic = tf.math.atan(
                tf.add(tf.matmul(state, variables['theta_ker']), variables['theta_b_ker']),
                                    name='layer_critic')
        value = tf.squeeze(
                tf.matmul(layer_critic, variables['theta_out']) + variables['theta_b_out'],
                           name='value')
        return value
    
    def _multilayer_actor_(self, state, variables):
        """
        Neural network architecture for the actor.
        
        Parameters
        ----------
        state: tensor
            Input of the network, this represent the current state of the cart-pole.
        variables: dic
            Dictionary with the variables of the neural network.
                
        Returns
        -------
        mean_action: tensor
            Output of the neural network (represent the action representative of the
            current state).
        """
        layer_actor = tf.math.atan( 
                tf.add(tf.matmul(state, variables['upsilon_ker']), variables['upsilon_b_ker']),
                                    name='layer_actor')
        mean_action = tf.squeeze(
                tf.matmul(layer_actor, variables['upsilon_out']) + variables['upsilon_b_out'],
                           name='mean_action')
        return mean_action

    def _multilayer_disturber_(self, state, variables):
        """
        Neural network architecture for the disturber.
        
        Parameters
        ----------
        state: tensor
            Input of the network, this represent the current state of the cart-pole.
        variables: dic
            Dictionary with the variables of the neural network.
                
        Returns
        -------
        mean_disturbance: tensor
            Output of the neural network (represent the environment disturbance of the
            current state).
        """
        layer_disturber = tf.math.atan( 
                tf.add(tf.matmul(state, variables['omega_ker']), variables['omega_b_ker']),
                                    name='layer_disturber')
        mean_disturbance = tf.squeeze(
                tf.matmul(layer_disturber, variables['omega_out']) + variables['omega_b_out'],
                           name='mean_disturbance')
        return mean_disturbance    
    
    def _get_e_trace_(self, gradient):
        e_trace = []
        for grad in gradient:
            e = np.zeros(grad.get_shape())
            e_trace.append(e)
        return e_trace

    def _update_e_trace_(self, gradient, e_trace, gamma, lamb):
        for i in range(len(e_trace)):
            e_trace[i] = gamma * lamb * e_trace[i] + gradient[i]
            assert(e_trace[i].shape == gradient[i].shape)
        return e_trace
    
    def get_action(self, state):
        action, mean = self.sess.run([self.action_choose, self.mean], {self.state: state})
        return action, mean
    
    def get_disturbance(self, state):
        disturbance, mean_disturber = self.sess.run([self.disturbance_choose, self.mean_disturber], {self.state: state})
        return disturbance, mean_disturber
    
    def save_model(self, path):
        self.saver.save(self.sess, path)
    
    def learn(self, state, action, disturbance, reward, next_state):
        old_value = self.sess.run(self.value, {self.state: state})
        new_value = self.sess.run(self.value, {self.state: next_state})
        augmented_reward = reward + self.robustness**2 * np.sum(disturbance**2) ###
        value_next = augmented_reward + self.gamma * new_value
        delta = value_next - old_value

        # update actor
        upsilon_gradients_eval = self.sess.run(self.upsilon_gradients, 
                                          {self.state: state, self.action: action})
        self.upsilon_trace = self._update_e_trace_(upsilon_gradients_eval, self.upsilon_trace, self.gamma, self.lamb)
        
        upsilon_change = [-delta * eu for eu in self.upsilon_trace]
        
        feed_dict_upsilon = {}
        for j in range(len(self.upsilon_grad_placeholder)):
            feed_dict_upsilon[self.upsilon_grad_placeholder[j][0]] = upsilon_change[j]
        # end for
            
        self.sess.run(self.train_actor, feed_dict = feed_dict_upsilon)

        # update disturber
        omega_gradients_val = self.sess.run(self.omega_gradients, 
                                            {self.state: state, self.disturbance: disturbance})
        self.omega_trace = self._update_e_trace_(omega_gradients_val, self.omega_trace, self.gamma, self.lamb)
        
        omega_change = [delta * eo for eo in self.omega_trace]
        
        feed_dict_omega = {}
        for j in range(len(self.omega_grad_placeholder)):
            feed_dict_omega[self.omega_grad_placeholder[j][0]] = omega_change[j]
        # end for
        
        self.sess.run(self.train_disturber, feed_dict = feed_dict_omega)
        log_gen = self.sess.run(self.log_generator, {self.state: state, self.disturbance: disturbance})

        # update critic
        theta_gradients_eval = self.sess.run(self.theta_gradients, {self.state: state})
        self.theta_trace = self._update_e_trace_(theta_gradients_eval, self.theta_trace, self.gamma, self.lamb)
        
        theta_change = [-delta * et for et in self.theta_trace]
        
        feed_dict_theta = {}
        for j in range(len(self.theta_grad_placeholder)):
            feed_dict_theta[self.theta_grad_placeholder[j][0]] = theta_change[j]
            
        self.sess.run(self.train_critic, feed_dict = feed_dict_theta)
        log_pol = self.sess.run(self.log_policy, {self.state: state, self.action: action})
        
        return delta, log_pol, log_gen

###############################################################################

class RRL_algorithm:
    def __init__(self, **kwargs):
        self.sample = kwargs.get('sample', 1)
        self.episodes = kwargs.get('episodes', 1000)
        self.cut = kwargs.get('cut', 300)
        self.friction = kwargs.get('friction', [0, 0])
        self.Umax = kwargs.get('Umax', 10)
        self.likelihood = kwargs.get('likelihood', None)
        self.likelihood_decreasing = kwargs.get('likelihood_decreasing', False)
        self.path = kwargs.get('path', 'C:\\Users\\my_model_train\\')
        self.file = kwargs.get('file', 'Run')
        self.RL_parameters = kwargs.get('RL_parameters', None)
        
#        self.file = file + '-[{0}]-[{1}]'.format(self.friction, self.sample)        
        self.weight_path = self.path + 'weight\\' + self.file + '-weight\\'
        
        try:
            os.makedirs(self.weight_path)  
            print("Directory:\n" , self.weight_path ,  "\n>> Created ")
        except FileExistsError:
            print("Directory:\n" , self.weight_path ,  "\n>> Already exists")
        
        self.weight_path = self.weight_path + self.file + '-weight'
        self.data_path = self.path + 'data\\' + self.file + '-data.file'
        
        self.results = self._algorithm_()
        self._save_data_(self.results, self.data_path)
        
    def _save_data_(self, data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    def _algorithm_(self):
        reward = np.zeros(self.episodes)
        augmented = np.zeros(self.episodes)
        iterations = np.zeros(self.episodes)
        friction_values = np.zeros(self.episodes)
        success = []
        
        env = gym.make('CartPole-v0').unwrapped # Cart-pole balancing environmenit
#        env.seed(1234) # seed
        env.Fmax = self.Umax
        state_dim = len(env.reset())
        
        state = np.zeros((self.episodes,state_dim))
        action = []
        mean = []
        mean_dis = []
        
        if self.RL_parameters is None:
            agent = RRL_actor_disturber_critic()
        else:
            agent = RRL_actor_disturber_critic(**self.RL_parameters)
        # end if
        
        time_all = time.time()
        for i in range(self.episodes):
            Xt = state[i,:] = env.reset()
            action_temp = []
            mean_temp = []
            mean_dis_temp = []
            
            # car friction to dynamic environments
            env.friction_cart = np.random.uniform(self.friction[0], self.friction[1])
            friction_values[i] = env.friction_cart
            
            time_iter = time.time()
            while True:
                # Choose the action from the agent policy
                Ut, Mt = agent.get_action(self._state_normalize_(Xt))
                Wt, Ot = agent.get_disturbance(self._state_normalize_(Xt))
                # Next state and reward of the cart-pole balancing task
                Xt_, Rt, done, info = env.step(Ut)
                augmented[i] =+ agent.robustness**2 * np.sum(Wt**2)
                
                # Update the RL model
                TD, log_policy, log_generator = agent.learn(self._state_normalize_(Xt), Ut, np.expand_dims(Wt,0), Rt, self._state_normalize_(Xt_))
                
                Xt = Xt_ # Change the current state by the next state
                
                # Feed the objects
                reward[i] += Rt
                iterations[i] += 1
                action_temp.append(Ut)
                mean_temp.append(Mt)
                mean_dis_temp.append(Ot)
                
                # print message with the information of the episode
                lista = [self.sample, i+1, self.episodes, round(reward[i],3), 
                             int(iterations[i]), info, None, None] 
                self._update_progress_(lista)
                
                # Conditions to end the state
                if done == True or iterations[i] >= self.cut:
                    parameter_ = {'robustness': agent.robustness, 'Friction': round(env.friction_cart,3),
                                  'disturbance_dim': agent.disturbance_dim, 'Umax': env.Fmax}
                    info_parameter = self._information_parameter_(parameter_)
                    # print message with the information of the episode
                    time_iter = time.time() - time_iter
                    lista = [self.sample, i+1, self.episodes,
                             round((reward[i] +augmented[i])/iterations[i],3), 
                             int(iterations[i]), info, info_parameter, round(time_iter,2)]
                    self._update_progress_(lista, finished=True)
                    print('')
                    break
                # end if
            # end while
            mean.append(mean_temp)
            action.append(action_temp)
            mean_dis.append(mean_dis_temp)
#            print('td', TD, 'policy', log_policy, 'generator', log_generator)
        # end for  
        
        env.close() # close environment (useful when render is True)
        # Save all object in a dict
        time_all = time.time() - time_all
        print('The cart-pole balancing task ending in {0:.2} minutes'.format(time_all/60))
        agent.save_model(self.weight_path)
        return {'Reward': reward, 'Iterations': iterations, 'Interactions': 0, 'Augmented': augmented,
                'states': state, 'actions': action, 'mean': mean, 'mean_dis': mean_dis,
                'time-seconds': time_all, 'success': success, 'Friction': friction_values}

    def _state_normalize_(self, state, normalize=True):
        """
        state standardization to feed a neural network
        
        Parameters
        ----------
        state: array_like
            The current state vector of the cart-pole
        normalize: boolean
            If 'True' standardizes the state vector, if it does not only 
            add a dimension to the state vector. Defauld is 'True'.
            
        Returns
        -------
        state_stand: array_like
            Standarized state vector
        """
        x, x_dot, theta, theta_dot = state
        if normalize:
            state_stand = np.array([[x/2.4, x_dot/2, theta/(12*np.pi/180), theta_dot/1.5]])
        else:    
            state_stand = np.array([[x, x_dot, theta, theta_dot]])
    #    state_stand = np.array([[x/2, x_dot/3, np.sin(theta)/0.8, np.cos(theta)/0.8, theta_dot/4]])
        # end if
        return state_stand
        
    def _get_advice_(self, action, state, friction=0): #env.env.tau
        """
        Return the feedback for a performed action and the current state in the 
        cart-pole balancing task
        
        Parameters
        ----------
        action: float
            Force to be applied over the car
        state: array_like
            The current state vector of the cart-pole
        friction: float
            Value of the car friction, Default is '0' (Non friction).
            
        Returns
        -------
        out: tuple
            A tuple with the feedback value -1 (left) or 1 (right) and the real force
            to be applied over the car so that in the next state the pole got to 0º 
        """
        np.random.normal
        gravity = 9.8 # gravity constant
        masscart = 1.0 # mass of cart
        masspole = 0.1 # mass of pole
        total_mass = (masspole + masscart) # total mass in the sistem
        length = 0.5 # actually half the pole's length
        polemass_length = (masspole * length)
        tau = 0.02  # seconds between state updates
        friction_pole = 0
        
        x, x_dot, theta, theta_dot = state
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        theta_dotdot = -(theta + 2*tau*theta_dot)/tau**2
        x_dotdot = -(x + 2*tau*x_dot)/tau**2
        k_theta = length * (4.0/3.0 - masspole * costheta * costheta / total_mass)
        
        A = 2*polemass_length*theta_dot**2*sintheta
        B = (total_mass*k_theta/costheta - polemass_length*costheta)*theta_dotdot
        C = (friction_pole/polemass_length)*theta_dot - gravity*sintheta
        
        F2t = A + B + total_mass*C/costheta - total_mass*x_dotdot - 2*friction*np.sign(x_dot)
        F_t = -0.5*F2t
        
#        F_t = (theta_dotdot*k_theta - gravity*sintheta)*total_mass/costheta + \
#            polemass_length * (theta_dot**2) * sintheta - np.sign(x_dot)*friction
        mean_advice = -F_t if np.abs(-F_t)<= 10 else np.sign(-F_t)*10
        return np.sign(mean_advice-action), mean_advice
        
    def _information_parameter_(self, parameter):
        info = ''
        for key, value in parameter.items():
            info += key + '=' + str(value) + '; ' 
        # end for
        return info
        
    def _update_progress_(self, information, finished=False):
        """
        Progress bar in console (requires import sys)
        
        Parameters
        ----------
        output: list
            List that include: step, episodes, info, reward, iteration and feedback
        time: float
            Time to print in progress bar. Default is 'None'.
        """
        agent, step, episodes, reward, iteration, info_task, info_parameter, seconds = information
        if step > episodes: sys.exit() # end if
        length = 20 # modify this to change the length of the bar
        block = int(round(length*step/episodes))
        arrow = "="*(block-1)+ ">" + "-"*(length-block)
        percentage = step/episodes
        
        if finished == True:
            if info_parameter is None:
                msg = "\r *Episode {0} finished in [{1}] with {2} seconds| Average reward {3} in {4} steps\n"
                mss = msg.format(step, info_task, seconds, reward, iteration)
            else:
                msg = "\r *Episode {0} finished in [{1}] with {2} seconds| Average reward {3} in {4} steps \n\t parameter information: {5}\n"
                mss = msg.format(step, info_task, seconds, reward, iteration, info_parameter)
            # end if
        else:
            msg = "\r Agent Nº {0} | Progress: {1}/{2} [{3}] {4:.1%} | Reward {5} in {6} steps "
            mss = msg.format(agent, step, episodes, arrow, percentage, reward, iteration)
        # end if
        sys.stdout.write(mss)
        sys.stdout.flush()

###############################################################################

#path = 'C:\\Users\\crist\\Documents\\Reinforcement Learning\\Implementation\\Cart-pole\\e\\'
#
#friction = [0, 10]
#episodes = 2000
#experiments = 2
#
#parameters_RL = {
#        'alpha_theta': 0.001,
#        'alpha_upsilon': 0.0001,
#        'gamma': 0.9,
#        'lamb': 0.0,
#        'robustness': 0.25,
#        'consistency': 1,
#        'exploration': 1,
#        'hidden_units': [50, 20, 20],
#        'disturbance_dim': 2,
#        'noise': np.identity(2,dtype=np.float32)
#        }
#
#for i in range(experiments):
#    file = 'RRL-CP' + '-[{0}]'.format(np.unique(friction)) 
#    parameters_run = {
#            'sample': i+1,
#            'episodes': episodes,
#            'cut': 300,
#            'friction': friction,
#            'path': path,
#            'file': file + '-[{0}]'.format(i+1),
#            'RL_parameters': parameters_RL
#            }
#
#    results = RRL_algorithm(**parameters_run)
#    reward_path = path + 'reward\\'+ file +'-reward.txt'
#    step_path = path + 'step\\'+ file + '-step.txt'
#    if i == 0:
#        np.savetxt(reward_path,np.array(range(episodes))+1, fmt="%s")
#        np.savetxt(step_path,np.array(range(episodes))+1, fmt="%s")
#    # end if
#    
#    dataset_r = np.genfromtxt(reward_path)
#    dataset_s = np.genfromtxt(step_path)
#    reward = results.results['Reward']/results.results['Iterations']
#    step = results.results['Iterations']
#    output_r = np.append(dataset_r.reshape(dataset_r.shape[0], i+1), reward.reshape(dataset_r.shape[0], 1), 1)
#    output_s = np.append(dataset_s.reshape(dataset_s.shape[0], i+1), step.reshape(dataset_s.shape[0], 1), 1)
#    np.savetxt(reward_path, output_r, fmt="%s")
#    np.savetxt(step_path, output_s, fmt="%s")
##    plt.plot(reward)
##plt.show()