import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from planar_pusher import Pusher



class Plotter:
    def __init__(self, pusher, dataset, reference_traj = None, save = False):
        self.pusher = pusher
        self.dataset = dataset
        self.reference_traj = reference_traj
        self.save = save

    def plot_traj_stretch_final(self, stretch_factor = 2.0, plot_indices = range(0,100, 10)):
        plt.rcParams['text.usetex'] = True
        if self.reference_traj is None:
            raise ValueError('Reference trajectory not provided')

        state_array = self.state_data_xy
        comparison_array = self.reference_traj
        # if self.save:
        #     plt.style.use(['science'])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True) 


        def add_frame(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1.)
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=0.8, ec='k', linestyle='--', lw=0.1)

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])

        def add_frame_init(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1., label='Slider')
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--', lw=0.1, label='Pusher')

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)
            return block_patch, pusher_patch

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])
        
        lims = [-0.07,0.2,-0.1,0.1]
        lims = [-0.1,0.22,-0.1,0.1]
        lims = [-0.1,0.5,-0.15,0.2]
        # Add a lineplot to the figure showing how the pusher evolves
        state_array_np = np.array(state_array)
        # Compute stretched x-values
        dx = np.diff(state_array_np[:,0])  # Compute differences between adjacent x-values
        stretched_dx = dx * stretch_factor  # Scale the differences
        stretched_x_values = np.concatenate(([state_array_np[0,0]], state_array_np[0,0] + np.cumsum(stretched_dx)))

        # Compute stretched px-values
        # Compute shift for each timestep
        shift = stretched_x_values - state_array_np[:,0]  # How much each point moved

        # Apply the same shift to x2-values
        stretched_px_values = state_array_np[:,2] + shift

        stretched_state_array = np.copy(state_array_np)
        stretched_state_array[:,0] = stretched_x_values
        stretched_state_array[:,2] = stretched_px_values
        for i in plot_indices:
            if i == 0:
                # Add initial with labels
                block_patch, pusher_patch = add_frame_init(self, i, stretched_state_array, comparison_array)
            add_frame(self, i, stretched_state_array, comparison_array)
        add_frame(self, stretched_state_array.shape[0]-1, stretched_state_array, comparison_array)

        # Add a lineplot to the figure showing how the pusher evolves
        c1 = plot_indices[0]
        c2 = plot_indices[1]
        c_arr = self.contact_switches

        # First segment no-contact:
        l_arr = []
        # l1 = ax.plot(stretched_state_array[c1:c2,2], stretched_state_array[c1:c2,3], color='r', linestyle='--',label="Pusher Trajectory: No Contact")
        # l_arr.append(l1)
        for i in range(0,len(c_arr)-1):
            if c_arr[i][1] == 1:
                l = ax.plot(stretched_state_array[c_arr[i][0]:c_arr[i+1][0],2], stretched_state_array[c_arr[i][0]:c_arr[i+1][0],3], color='r', linestyle='-',label="Pusher Trajectory: Contact")
                l_arr.append(l)
            else:
                l = ax.plot(stretched_state_array[c_arr[i][0]:c_arr[i+1][0],2], stretched_state_array[c_arr[i][0]:c_arr[i+1][0],3], color='r', linestyle='--',label="Pusher Trajectory: No Contact")
                l_arr.append(l)
        if c_arr[-1][1] == 1:
            l = ax.plot(stretched_state_array[c_arr[-1][0]:,2], stretched_state_array[c_arr[-1][0]:,3], color='r', linestyle='-',label="Pusher Trajectory: Contact")
            l_arr.append(l)
        else:
            l = ax.plot(stretched_state_array[c_arr[-1][0]:,2], stretched_state_array[c_arr[-1][0]:,3], color='r', linestyle='--',label="Pusher Trajectory: No Contact")
            l_arr.append(l)

        # Second segment contact:
        #l2 = ax.plot(stretched_state_array[c2:,2], stretched_state_array[c2:,3], color='r', linestyle='-',label="Pusher Trajectory: Contact")
        ax.plot()
        plt.title(r"$\textrm{Pusher Repositioning for a Rightward Trajectory}$")
        plt.xlabel(r"$\textrm{X Position (m)}$")
        plt.ylabel(r"$\textrm{Y Position (m)}$")
        #plt.legend()
        plt.grid(False) 
        # Add arrows at regular intervals
        arrow_spacing = 20  # Place an arrow every 2 points
        for i in range(0, stretched_state_array.shape[0] - 1, arrow_spacing):
            plt.annotate("",
                        xy=(stretched_state_array[i+1,2], stretched_state_array[i+1,3]),  # Arrowhead
                        xytext=(stretched_state_array[i,2], stretched_state_array[i,3]),  # Arrow tail
                        arrowprops=dict(arrowstyle="->", lw=1, color="red"))
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        if self.save:
            plt.savefig('results/smallishfric_stretched_traj.png', dpi=600)
        else:
            plt.show()

    def plot_traj_stretch(self, stretch_factor = 2.0, plot_indices = range(0,100, 10)):
        plt.rcParams['text.usetex'] = True
        if self.reference_traj is None:
            raise ValueError('Reference trajectory not provided')

        state_array = self.state_data_xy
        comparison_array = self.reference_traj
        # if self.save:
        #     plt.style.use(['science'])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True) 


        def add_frame(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1.)
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--', lw=0.1)

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])

        def add_frame_init(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1., label='Slider')
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--', lw=0.1, label='Pusher')

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)
            return block_patch, pusher_patch

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])
        
        lims = [-0.07,0.2,-0.1,0.1]
        lims = [-0.1,0.22,-0.1,0.1]
        # Add a lineplot to the figure showing how the pusher evolves
        state_array_np = np.array(state_array)
        # Compute stretched x-values
        dx = np.diff(state_array_np[:,0])  # Compute differences between adjacent x-values
        stretched_dx = dx * stretch_factor  # Scale the differences
        stretched_x_values = np.concatenate(([state_array_np[0,0]], state_array_np[0,0] + np.cumsum(stretched_dx)))

        # Compute stretched px-values
        # Compute shift for each timestep
        shift = stretched_x_values - state_array_np[:,0]  # How much each point moved

        # Apply the same shift to x2-values
        stretched_px_values = state_array_np[:,2] + shift

        stretched_state_array = np.copy(state_array_np)
        stretched_state_array[:,0] = stretched_x_values
        stretched_state_array[:,2] = stretched_px_values
        for i in plot_indices:
            if i == 0:
                # Add initial with labels
                block_patch, pusher_patch = add_frame_init(self, i, stretched_state_array, comparison_array)
            add_frame(self, i, stretched_state_array, comparison_array)
        add_frame(self, stretched_state_array.shape[0]-1, stretched_state_array, comparison_array)

        
        ax.plot(stretched_state_array[:,2], stretched_state_array[:,3], color='r', linestyle='--',label="Pusher Trajectory")
        plt.title(r"$\textrm{Following a Horizontal Trajectory}$")
        plt.xlabel(r"$\textrm{X Position (m)}$")
        plt.ylabel(r"$\textrm{Y Position (m)}$")
        plt.legend()
        plt.grid(False) 
        # Add arrows at regular intervals
        arrow_spacing = 20  # Place an arrow every 2 points
        for i in range(0, stretched_state_array.shape[0] - 1, arrow_spacing):
            plt.annotate("",
                        xy=(stretched_state_array[i+1,2], stretched_state_array[i+1,3]),  # Arrowhead
                        xytext=(stretched_state_array[i,2], stretched_state_array[i,3]),  # Arrow tail
                        arrowprops=dict(arrowstyle="->", lw=1, color="red"))
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        if self.save:
            plt.savefig('results/stretched_traj.png', dpi=600)
        else:
            plt.show()

    def plot_traj_wref(self, interval = 10):
        if self.reference_traj is None:
            raise ValueError('Reference trajectory not provided')

        state_array = self.state_data_xy
        comparison_array = self.reference_traj
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True) 


        def add_frame(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--')
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--')

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])
        
        lims = [-0.1,0.2,-0.1,0.1]
        for i in range(0, len(state_array), interval):
            add_frame(self, i, state_array, comparison_array)
        # Add a lineplot to the figure showing how the pusher evolves
        state_array_np = np.array(state_array)
        ax.plot(state_array_np[:,2], state_array_np[:,3], color='r', linestyle='--')
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        plt.show()

    def find_contact_switches(self):
        """Finds the contact switches in the state data."""
        contact_switches = []
        contact_dist = self.state_data[:,2].copy() - self.pusher.block_diameter/2
        for i in range(1, len(self.state_data)):
            if contact_dist[i] > 0 and contact_dist[i-1] <= 0:
                contact_switches.append([i, -1])
            elif contact_dist[i] <= 0 and contact_dist[i-1] > 0:
                contact_switches.append([i, 1])
        return contact_switches
        
    def animate_dataset(self, dataset, pusher, goal_array, dt=0.01,SAVE=False,str=None,lims=None):
        """Animates the trajectory of the planar pusher model.
        Inputs:
            dataset: a numpy array composed of dictionaries representing a particular run of the pusher model. Each dictionary includes at least the following keys:
                mu_t: the coefficient of static friction
                state_data_xy: a numpy array of shape (n, 6) where n is the number of timesteps and each row is a state of the form [x, y, px, py, xdot, ydot].
                dt: the time step between each state
            goal_array: a numpy array of shape (n, 2) where n is the number of timesteps and each row is a goal state of the form [x, y].
            dt: the time step between each state
            SAVE: a boolean indicating whether to save the animation as a video file
            str: a string to use as the filename for the saved video
            lims: a list of four values representing the limits of the x and y axes in the form [xmin, xmax, ymin, ymax]
        """
        
        def init_ani():
            for pusher_patch in pusher_patches:
                ax.add_patch(pusher_patch)
            for block_patch in block_patches:
                ax.add_patch(block_patch)
            #ax.add_patch(ee_patch)
            ax.add_patch(goal_patch)

        block_patches = []
        pusher_patches = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)

        # Create the patches
        for run in dataset:        
            pusher_patch = plt.Circle((0, 0), pusher.block_diameter/20, fc='r',zorder=2, alpha=0.3, ec='k')
            block_patch = plt.Circle((0, 0), pusher.block_diameter/2, fc='b', alpha=0.3, ec='k')
            block_patches.append(block_patch)
            pusher_patches.append(pusher_patch)
        goal_patch = plt.Circle((0,0), pusher.block_diameter/2, fc='y', alpha=0.5)
        anim = animation.FuncAnimation(fig, self.pusher_animation_func_comparison, init_func=init_ani(),frames=len(dataset[0]['state_data_xy']),
                                       fargs=(dataset, goal_array, pusher_patches, block_patches, goal_patch))
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        if SAVE:
            writervideo = animation.FFMpegWriter(fps=10)
            anim.save(str+'.mp4', writer=writervideo)
        plt.show()

    def pusher_animation_func_comparison(self, frame, dataset, goal_array, pusher_patches, block_patches, goal_patch):
        """Function to animate the pusher."""
        for i, run in enumerate(dataset):
            state_array = run['state_data_xy']
            pusher_patch = pusher_patches[i]
            block_patch = block_patches[i]
            goal_state = goal_array[frame,:]
            current_state = state_array[frame]
            
            block_patch.set_center([current_state[0],current_state[1]])   
            pusher_patch.set_center([current_state[2], current_state[3]])

            goal_state = goal_array[frame]
            goal_patch.set_center([goal_state[0], goal_state[1]])

# Load the data
dataset = np.load('results/bilinear_results_035.npy', allow_pickle=True)
dataset = np.load('results/bilinear_results_singletraj.npy', allow_pickle=True)

SAVE = True
num_states = 4
dt = 0.1

max_timesteps = dataset[0]['state_data_xy'].shape[0]


# Set initial values
mu_p = 0.3
mu_t = 0.35
mass_block = 0.827
block_length = 0.16
pusher_k = 1000
max_energy = None
init_state = np.array(
    [0, 0, block_length/2, np.deg2rad(180), 0, 0])
conduit_traj = np.zeros((max_timesteps,num_states+2))
conduit_traj[:,0] = 200*0.005*dt + init_state[0]
conduit_traj[:,1] = conduit_traj[:,1] + init_state[1]
pusher = Pusher(block_length, mass_block, mu_t, mu_p, pusher_k, init_state)
plotter = Plotter(pusher, dataset, conduit_traj)
plotter.save = SAVE
plotter.animate_dataset(dataset, pusher, conduit_traj, dt=dt, SAVE=SAVE, str='results/bilinear_singletest', lims=[-0.25, 0.25, -0.3, 0.3])


