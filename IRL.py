# -*- coding: utf-8 -*-

#from docopt import docopt
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import tensorflow_probability as tfp
import time
    
###############################################################################

class IRL_actor_critic:
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
        self.gamma = kwargs.get('gamma', 0.9)
        self.lamb = kwargs.get('lamb', 0.0)
        
        self.consistency = kwargs.get('consistency', 1.)
        self.shaping = kwargs.get('shaping', 3.)
        
        # Network Parameters
        self.exploration = kwargs.get('exploration', 1)
        self.hidden_units = kwargs.get('hidden_units', [50, 20])
        self.state_dim = kwargs.get('state_dim', 4)
        
        self.hidden_units_critic = [self.state_dim ,self.hidden_units[0]]
        self.hidden_units_actor = [self.state_dim ,self.hidden_units[1]]

        tf.reset_default_graph()
#        tf.random.set_random_seed(1234) # seed
#        np.random.seed(1234) # seed
        
        # tensorflow Graph input
        self.advice, self.state, self.value, self.optimizer_critic, self.theta, self.action, self.mean, self.log_policy, self.optimizer_actor, self.action_choose, self.upsilon = self._actor_critic_model_()
        
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
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 1
            
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def _actor_critic_model_(self):
        
        state = tf.placeholder('float', [None, self.state_dim], 'state')
        action = tf.placeholder('float', None, 'action')
        advice = tf.placeholder('float', None, 'advice')
        
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
            
            ### ADVICE
            with tf.name_scope('advice'):
                important_factor = self.consistency**2 / ( self.consistency**2 + tf.abs(advice)*self.exploration**2)
                advice_mean = tf.add(mean, self.shaping*advice)
                sigma = tf.sqrt(important_factor) * self.exploration
                mean_j = important_factor * mean + (1-important_factor)*advice_mean
            
            policy = tfp.distributions.Normal(mean_j, sigma) # SIGMA
            log_policy = policy.log_prob(action)
            
            # Training and optimize the policy
            with tf.name_scope('training'):
                rate_scale_williams = self.alpha_upsilon * sigma**2 ## SIGMA
                optimizer_actor = tf.train.GradientDescentOptimizer(rate_scale_williams)
            
            # Drawing action of the policy
            with tf.name_scope('selection'):
                action_choose = tf.squeeze(policy.sample(1))
        
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
                
        return advice, state, value, optimizer_critic, variables_critic, action, mean_j, log_policy, optimizer_actor, action_choose, variables_actor
                              
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
    
    def get_action(self, state, advice):
        action, mean = self.sess.run([self.action_choose, self.mean], {self.state: state, self.advice: advice})
        return action, mean
    
    def save_model(self, path):
        self.saver.save(self.sess, path)
    
    def learn(self, state, action, advice, reward, next_state):
        old_value = self.sess.run(self.value, {self.state: state})
        new_value = self.sess.run(self.value, {self.state: next_state})
        value_next = reward + self.gamma * new_value
        delta = value_next - old_value

        # update actor
        upsilon_gradients_eval = self.sess.run(self.upsilon_gradients, 
                                          {self.state: state, self.action: action, self.advice: advice})
        self.upsilon_trace = self._update_e_trace_(upsilon_gradients_eval, self.upsilon_trace, self.gamma, self.lamb)
        
        upsilon_change = [-delta * eu for eu in self.upsilon_trace]
        
        feed_dict_upsilon = {}
        for j in range(len(self.upsilon_grad_placeholder)):
            feed_dict_upsilon[self.upsilon_grad_placeholder[j][0]] = upsilon_change[j]
        # end for
        feed_dict_upsilon[self.advice] = advice
            
        self.sess.run(self.train_actor, feed_dict = feed_dict_upsilon)
        log_pol = self.sess.run(self.log_policy, {self.state: state, self.action: action, self.advice: advice})

        # update critic
        theta_gradients_eval = self.sess.run(self.theta_gradients, {self.state: state})
        self.theta_trace = self._update_e_trace_(theta_gradients_eval, self.theta_trace, self.gamma, self.lamb)
        
        theta_change = [-delta * et for et in self.theta_trace]
        
        feed_dict_theta = {}
        for j in range(len(self.theta_grad_placeholder)):
            feed_dict_theta[self.theta_grad_placeholder[j][0]] = theta_change[j]
        # end for
            
        self.sess.run(self.train_critic, feed_dict = feed_dict_theta)
        
        return delta, log_pol

###############################################################################

class IRL_algorithm:
    def __init__(self, **kwargs):
        self.sample = kwargs.get('sample', 1)
        self.episodes = kwargs.get('episodes', 1000)
        self.cut = kwargs.get('cut', 300)
        self.friction = kwargs.get('friction', [0, 0])
        self.Umax = kwargs.get('Umax', 10)
        self.likelihood = kwargs.get('likelihood', None)
        self.likelihood_decreasing = kwargs.get('likelihood_decreasing', False)
        self.path = kwargs.get('path', 'C:\\Users\\')
        self.file = kwargs.get('file', 'Run')
        self.RL_parameters = kwargs.get('RL_parameters', None)
        
#        self.file = file + '-[{0}]-[{1}]'.format(self.likelihood, self.sample)        
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
        iterations = np.zeros(self.episodes)
        interactions = np.zeros(self.episodes)
        success = []
        
        env = gym.make('CartPole-v0').unwrapped # Cart-pole balancing environmenit
#        env.seed(1234) # seed
        env.Fmax = self.Umax
        state_dim = len(env.reset())
        
        state = np.zeros((self.episodes,state_dim))
        action = []
        mean = []
        
        if self.RL_parameters is None:
            agent = IRL_actor_critic()
        else:
            agent = IRL_actor_critic(**self.RL_parameters)
        # end if
        
        time_all = time.time()
        for i in range(self.episodes):
            Xt = state[i,:] = env.reset()
            action_temp = []
            mean_temp = []
            
            # car friction to dynamic environments
            env.friction_cart = np.random.uniform(self.friction[0], self.friction[1])
            
            time_iter = time.time()
            while True:
                Jt = 0.
                # Choose the action from the agent policy
                Ut, Mt = agent.get_action(self._state_normalize_(Xt), Jt)
                
                if self.likelihood is not None:
                    # Probability of feedback
                    if self.likelihood_decreasing == False:
                        likelihood = self.likelihood
                    else:
                        likelihood = self.likelihood - (self.likelihood-self.likelihood)*(i/self.episodes)
                    # end if
                    
                    rand = np.random.uniform(size=())
                    if rand <= likelihood:
                        # Feedback is given
                        Jt, _ = self._get_advice_(Ut, Xt, env.friction_cart)
                        # Choose the action from the agent-feedback policy
                        Ut, Mt = agent.get_action(self._state_normalize_(Xt), Jt)
                        interactions[i] += 1
                    # end if
                # end if
                
                # Next state and reward of the cart-pole balancing task
                Xt_, Rt, done, info = env.step(Ut)
                
                # Update the RL model
                TD, log_policy = agent.learn(self._state_normalize_(Xt), Ut, Jt, Rt, self._state_normalize_(Xt_))
                
                Xt = Xt_ # Change the current state by the next state
                
                # Feed the objects
                reward[i] += Rt
                iterations[i] += 1
                action_temp.append(Ut)
                mean_temp.append(Mt)
                
                # print message with the information of the episode
                lista = [self.sample, i+1, self.episodes, round(reward[i],3), 
                             int(iterations[i]), info, None, None] 
                self._update_progress_(lista)
                
                # Conditions to end the state
                if done == True or iterations[i] >= self.cut:
                    parameter_ = {'Likelihood': likelihood, 'Friction': round(env.friction_cart,5),
                                  'Umax': env.Fmax, 'Exploration': agent.exploration}
                    info_parameter = self._information_parameter_(parameter_)
                    # print message with the information of the episode
                    time_iter = time.time() - time_iter
                    lista = [self.sample, i+1, self.episodes, round(reward[i]/iterations[i],3), 
                             int(iterations[i]), info, info_parameter, round(time_iter,2)]
                    self._update_progress_(lista, finished=True)
                    print('')
                    break
                # end if
            # end while
            mean.append(mean_temp)
            action.append(action_temp)
        # end for  
        
        env.close() # close environment (useful when render is True)
        # Save all object in a dict
        agent.save_model(self.weight_path)
        time_all = time.time() - time_all
        print('The cart-pole balancing task ending in {0:.2} minutes'.format(time_all/60))
        return {'Reward': reward, 'Iterations': iterations, 'Interactions': interactions,
                'states': state, 'actions': action, 'mean': mean,
                'time-seconds': time_all, 'success': success}
        
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
        
        x, x_dot, theta, theta_dot = state
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        theta_dotdot = -(theta + 2*tau*theta_dot)/tau**2
        k_theta = (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        F_t = (theta_dotdot*k_theta - gravity*sintheta)*total_mass/costheta + \
            polemass_length * (theta_dot**2) * sintheta - np.sign(x_dot)*friction
        mean_advice = -F_t if np.abs(-F_t)<= 10 else np.sign(-F_t)*10
        return np.sign(mean_advice-action), mean_advice
        
    def _information_parameter_(self, parameter):
#        parameter = {'a':1, 'b':2, 'c':3}
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
