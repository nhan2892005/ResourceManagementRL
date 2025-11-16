import numpy as np
import math

import job_distribution


class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'

        self.num_epochs = 1000         # number of training epochs
        self.simu_len = 50             # length of the busy cycle that repeats itself
        self.num_ex = 10               # number of sequences

        self.output_freq = 10           # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_res = 3               # number of resources in the system
        self.num_nw = 5                # maximum allowed number of work in the queue

        self.time_horizon = 20         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work

        self.backlog_size = 60         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = int((self.res_slot + self.max_job_size * self.num_nw)
                                       * self.num_res + self.backlog_width + 1)
        # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self._compute_feature_dim()

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def _compute_feature_dim(self):
        """
        Compute the dimension of feature extraction representation
        
        Feature breakdown:
        - Resource features: num_res * 5 features per resource
          (total_capacity, available, used, num_jobs, near_future_util)
        - Job slot features: num_nw * (num_res + 4) features per slot
          (res_vec for each resource, len, total_demand, wait_time, can_schedule)
        - Backlog features: 3 features
          (size, avg_res_demand, avg_len)
        - Running jobs features: 2 features
          (num_running, avg_remaining_time)
        - Temporal features: 2 features
          (time_since_last_job, sim_progress)
        """
        resource_features = self.num_res * 5
        job_slot_features = self.num_nw * (self.num_res + 4)
        backlog_features = 3
        running_features = 2
        temporal_features = 2
        
        self.network_feature_dim = (resource_features + 
                                   job_slot_features + 
                                   backlog_features + 
                                   running_features + 
                                   temporal_features)

    def _compute_text_dim(self):
        """
        Compute the dimension of text-based representation
        
        Each part is encoded separately by SentenceTransformer:
        1. Cluster resources: 1 embedding
        2. Job slots: num_nw embeddings (one per slot)
        3. Backlog: 1 embedding
        4. Running jobs: 1 embedding
        5. Temporal info: 1 embedding
        
        Total embeddings: 1 + num_nw + 1 + 1 + 1 = num_nw + 4
        
        Each embedding from all-MiniLM-L6-v2 is 384-dimensional
        Total dimension = (num_nw + 4) * 384
        """
        # SentenceTransformer all-MiniLM-L6-v2 produces 384-dim embeddings
        embedding_dim = 384
        num_parts = 1 + self.num_nw + 1 + 1 + 1  # cluster + slots + backlog + running + temporal
        
        self.network_text_dim = num_parts * embedding_dim
        
        print(f"Text representation breakdown:")
        print(f"  - 1 cluster resource part: {embedding_dim}D")
        print(f"  - {self.num_nw} job slot parts: {self.num_nw} × {embedding_dim}D")
        print(f"  - 1 backlog part: {embedding_dim}D")
        print(f"  - 1 running jobs part: {embedding_dim}D")
        print(f"  - 1 temporal part: {embedding_dim}D")
        print(f"  Total parts: {num_parts}")
        print(f"  Total dimension: {self.network_text_dim}D")
    
    def _compute_semi_text_dim(self):
        """
        Compute the dimension of semi-text (hybrid) representation
        
        Combines numerical features and text embeddings:
        
        NUMERICAL FEATURES:
        - Resource: num_res * 3 (availability, utilization, num_jobs)
        - Job slots: num_nw * (num_res + 3)
        - Backlog: 1
        - Running: 1
        - Temporal: 2
        
        TEXT EMBEDDINGS (no cache):
        - Backlog: 1 * 384D
        - Running jobs: 1 * 384D
        - Job slots: num_nw * 384D
        - Temporal: 1 * 384D
        Total text parts: (num_nw + 3) * 384D
        """
        embedding_dim = 384
        
        # Numerical features
        numerical_dims = (self.num_res * 3 +
                         self.num_nw * (self.num_res + 3) +
                         1 +  # backlog
                         1 +  # running
                         2)   # temporal
        
        # Text embeddings
        num_text_parts = 2 + self.num_nw + 1  # backlog + running + job_slots + temporal
        text_dims = num_text_parts * embedding_dim
        
        self.network_semi_text_dim = numerical_dims + text_dims
        
        print(f"\nSemi-Text (Hybrid) representation breakdown:")
        print(f"  Numerical features: {numerical_dims}D")
        print(f"    - Resource: {self.num_res} × 3 = {self.num_res * 3}D")
        print(f"    - Job slots: {self.num_nw} × ({self.num_res} + 3) = {self.num_nw * (self.num_res + 3)}D")
        print(f"    - Backlog: 1D")
        print(f"    - Running: 1D")
        print(f"    - Temporal: 2D")
        print(f"  Text embeddings: {text_dims}D ({num_text_parts} parts × {embedding_dim}D each)")
        print(f"  Total dimension: {self.network_semi_text_dim}D")

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = int((self.res_slot + self.max_job_size * self.num_nw)
                                       * self.num_res + self.backlog_width + 1)
        # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        # recompute feature dimension
        self._compute_feature_dim()
        self._compute_text_dim()
        self._compute_semi_text_dim()
        self.network_output_dim = self.num_nw + 1  # + 1 for void action
