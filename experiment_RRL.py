# -*- coding: utf-8 -*-
"""Usage: my_program.py [options]

options:
    --cut NUMBER          [default: 400]
    --epoch NUMBER        [default: 2000]
    --experiment NUMBER  [default: 10]
    --next NUMBER       [default: 0]
    --render
"""

from docopt import docopt
#import matplotlib.pyplot as plt
import numpy as np

from RRL import RRL_algorithm

def main():
    # Read parameters of the code's head (requires import docopt)
    arguments = docopt(__doc__, version='FIXME')
    print(arguments) # print arguments in the line code of prompt (or code's head)

    # Parameters with docopt package
    episodes_doc = int(arguments['--epoch'])    
    cut_doc = int(arguments['--cut'])  
    experiments_doc = int(arguments['--experiment'])
    next_doc = int(arguments['--next'])

    path = 'C:\\Users\\Jogos\\Desktop\\Millan\\Cart-pole\\experiment\\'
    friction = [[0.0005, 0.002]]
    Umax = [0.1, 1, 5, 10, 15]
    
    parameters_RL = {
        'alpha_theta': 0.001,
        'alpha_upsilon': 0.0001,
        'alpha_omega': 0.0001,
        'gamma': 0.9,
        'lamb': 0.0,
        'robustness': 0.45,
        'consistency': 1,
        'exploration': 1,
        'hidden_units': [50, 20, 20],
        'disturbance_dim': 2,
        'noise': np.identity(2,dtype=np.float32)
        }
    
    logger = open(path + "log_RRL-CP.txt","w+")
    logger.write('RRL-CP experiement with {0} epidoses and {1} samples\n Others params {2}\n'.format(episodes_doc, experiments_doc,parameters_RL))
    logger.write('info: param 1: {0}, param 2: {1}\n'.format(friction, Umax))
    for i in range(next_doc,experiments_doc+next_doc):
        for j in range(len(friction)):
            for k in range(len(Umax)):
                file = 'RRL-CP' + '-{0}'.format(j*len(Umax) + k + 1) 
                parameters_run = {
                        'sample': i+1,
                        'episodes': episodes_doc,
                        'cut': cut_doc,
                        'friction': friction[j],
                        'Umax': Umax[k],
                        'path': path,
                        'file': file + '-[{0}]'.format(i+1),
                        'RL_parameters': parameters_RL
                        }
            
                results = RRL_algorithm(**parameters_run)
                reward_path = path + 'reward\\'+ file +'-reward.txt'
                step_path = path + 'step\\'+ file +'-step.txt'
                
                if i == 0:
                    logger.write('{0} refered to param 1: {1}, param 2: {2}\n'.format(file, friction[j], Umax[k]))
                    np.savetxt(reward_path,np.array(range(episodes_doc))+1, fmt="%s")
                    np.savetxt(step_path,np.array(range(episodes_doc))+1, fmt="%s")
                # end if
                
                dataset_r = np.genfromtxt(reward_path)
                dataset_s = np.genfromtxt(step_path)
                reward = results.results['Reward']/results.results['Iterations']
                step = results.results['Iterations']
                output_r = np.append(dataset_r.reshape(dataset_r.shape[0], i+1), reward.reshape(dataset_r.shape[0], 1), 1)
                output_s = np.append(dataset_s.reshape(dataset_s.shape[0], i+1), step.reshape(dataset_s.shape[0], 1), 1)
                np.savetxt(reward_path, output_r, fmt="%s")
                np.savetxt(step_path, output_s, fmt="%s")
#                plt.plot(step)
            # end for
        # end for
    # end for
#    plt.show()
    logger.close()
    
if __name__ == '__main__':
    main()