import numpy as np
import math
import matplotlib.pyplot as plt
from text_observe_representation import TextObservationEncoder
from semi_text_observe_representation import SemiTextObservationEncoder

import parameters


class Env:
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image', end='no_new_job'):

        self.pa = pa
        self.render = render
        self.repre = repre  # image, feature_extract, text, semi_text
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        if self.repre == 'text':
            self.text_encoder = TextObservationEncoder()
            print(f"Text encoder initialized with {self.text_encoder.embedding_dim}D embeddings")
        
        if self.repre == 'semi_text':
            self.semi_text_encoder = SemiTextObservationEncoder()
            print(f"Semi-text encoder initialized with {self.semi_text_encoder.embedding_dim}D text embeddings")

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            self.nw_len_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            self.workload = np.zeros(pa.num_res)
            for i in range(pa.num_res):
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(pa.res_slot) / \
                    float(len(self.nw_len_seqs))
                print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def generate_sequence_work(self, simu_len):

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_size_seq = np.zeros((simu_len, self.pa.num_res), dtype=int)

        for i in range(simu_len):

            if np.random.rand() < self.pa.new_job_rate:  # a new job comes

                nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()

        return nw_len_seq, nw_size_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):
        if self.repre == 'image':
            return self._observe_image()
        elif self.repre == 'feature_extract':
            return self._observe_feature_extract()
        elif self.repre == 'text':
            return self._observe_text()
        elif self.repre == 'semi_text':
            return self._observe_semi_text()
        else:
            raise ValueError(f"Unknown representation type: {self.repre}")
    
    def _observe_text(self):
        """Text-based observation using SentenceTransformer"""
        return self.text_encoder.encode_state(self)
    
    def _observe_semi_text(self):
        """Semi-text observation using hybrid numerical + text embeddings"""
        return self.semi_text_encoder.encode_state(self)

    def _observe_image(self):
        """Original image-based observation"""
        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))

        image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

        ir_pt = 0

        for i in range(self.pa.num_res):

            image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
            ir_pt += self.pa.res_slot

            for j in range(self.pa.num_nw):

                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    image_repr[: self.job_slot.slot[j].len, ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1

                ir_pt += self.pa.max_job_size

        image_repr[: int(self.job_backlog.curr_size / backlog_width),
                   ir_pt: ir_pt + backlog_width] = 1
        if self.job_backlog.curr_size % backlog_width > 0:
            image_repr[int(self.job_backlog.curr_size / backlog_width),
                       ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1
        ir_pt += backlog_width

        image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                                          float(self.extra_info.max_tracking_time_since_last_job)
        ir_pt += 1

        assert ir_pt == image_repr.shape[1]

        return image_repr.ravel()[np.newaxis, :]

    def _observe_feature_extract(self):
        """Feature-based observation representation"""
        features = []
        
        # ============ MACHINE/RESOURCE FEATURES ============
        for i in range(self.pa.num_res):
            # Feature 1: Total resource capacity (normalized)
            total_capacity = self.pa.res_slot * self.pa.time_horizon
            features.append(total_capacity / float(self.pa.res_slot * self.pa.time_horizon))
            
            # Feature 2: Currently available resource (normalized)
            avbl_slots = np.sum(self.machine.avbl_slot[:, i])
            features.append(avbl_slots / float(total_capacity))
            
            # Feature 3: Currently used resource (normalized)
            used_slots = total_capacity - avbl_slots
            features.append(used_slots / float(total_capacity))
            
            # Feature 4: Number of jobs currently using this resource
            num_jobs_on_res = 0
            for job in self.machine.running_job:
                if job.res_vec[i] > 0:
                    num_jobs_on_res += 1
            features.append(num_jobs_on_res / float(self.pa.job_num_cap))
            
            # Feature 5: Resource utilization in near future (next 5 time steps)
            near_future_util = 0
            horizon_check = min(5, self.pa.time_horizon)
            for t in range(horizon_check):
                near_future_util += (self.pa.res_slot - self.machine.avbl_slot[t, i])
            features.append(near_future_util / float(self.pa.res_slot * horizon_check))
        
        # ============ JOB SLOT FEATURES ============
        for j in range(self.pa.num_nw):
            job = self.job_slot.slot[j]
            
            if job is None:
                # Empty slot - all features are 0
                features.extend([0] * (self.pa.num_res + 4))
            else:
                # Feature 1-num_res: Resource requests for each resource (normalized)
                for i in range(self.pa.num_res):
                    features.append(job.res_vec[i] / float(self.pa.max_job_size))
                
                # Feature: Job length (normalized)
                features.append(job.len / float(self.pa.max_job_len))
                
                # Feature: Total resource demand (normalized)
                total_demand = np.sum(job.res_vec)
                features.append(total_demand / float(self.pa.max_job_size * self.pa.num_res))
                
                # Feature: Waiting time (normalized)
                wait_time = self.curr_time - job.enter_time
                features.append(wait_time / float(self.pa.episode_max_length))
                
                # Feature: Can be scheduled now (binary)
                can_schedule = 1.0 if self._can_schedule_job(job) else 0.0
                features.append(can_schedule)
        
        # ============ BACKLOG FEATURES ============
        # Feature 1: Number of jobs in backlog (normalized)
        features.append(self.job_backlog.curr_size / float(self.pa.backlog_size))
        
        # Feature 2: Average resource demand in backlog
        if self.job_backlog.curr_size > 0:
            total_res_demand = 0
            for job in self.job_backlog.backlog:
                if job is not None:
                    total_res_demand += np.sum(job.res_vec)
            avg_res_demand = total_res_demand / float(self.job_backlog.curr_size)
            features.append(avg_res_demand / float(self.pa.max_job_size * self.pa.num_res))
        else:
            features.append(0.0)
        
        # Feature 3: Average job length in backlog
        if self.job_backlog.curr_size > 0:
            total_len = 0
            for job in self.job_backlog.backlog:
                if job is not None:
                    total_len += job.len
            avg_len = total_len / float(self.job_backlog.curr_size)
            features.append(avg_len / float(self.pa.max_job_len))
        else:
            features.append(0.0)
        
        # ============ RUNNING JOBS FEATURES ============
        # Feature 1: Number of running jobs (normalized)
        features.append(len(self.machine.running_job) / float(self.pa.job_num_cap))
        
        # Feature 2: Average remaining time for running jobs
        if len(self.machine.running_job) > 0:
            total_remaining = 0
            for job in self.machine.running_job:
                remaining = job.finish_time - self.curr_time
                total_remaining += remaining
            avg_remaining = total_remaining / float(len(self.machine.running_job))
            features.append(avg_remaining / float(self.pa.max_job_len))
        else:
            features.append(0.0)
        
        # ============ TEMPORAL FEATURES ============
        # Feature 1: Time since last new job (normalized)
        features.append(self.extra_info.time_since_last_new_job / 
                       float(self.extra_info.max_tracking_time_since_last_job))
        
        # Feature 2: Current simulation progress (normalized)
        features.append(self.seq_idx / float(self.pa.simu_len))
        
        # Convert to numpy array and add batch dimension
        feature_array = np.array(features, dtype=np.float32)
        return feature_array[np.newaxis, :]
    
    def _can_schedule_job(self, job):
        """Check if a job can be scheduled now"""
        for t in range(0, self.pa.time_horizon - job.len):
            new_avbl_res = self.machine.avbl_slot[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                return True
        return False

    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.pa.num_res):

            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.len)

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len)

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len)

        return reward

    def step(self, a, repeat=False):

        status = None

        done = False
        reward = 0
        info = None

        if a == self.pa.num_nw:  # explicit void action
            status = 'MoveOn'
        elif self.job_slot.slot[a] is None:  # implicit void action
            #if self.seq_idx >= self.pa.simu_len and \
                    #len(self.machine.running_job) > 0 and \
                    #all(s is None for s in self.job_backlog.backlog):
                #ob, reward, done, info = self.step(a + 1, repeat=True)
                #return ob, reward, done, info
            #else:
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                status = 'Allocate'

        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            if self.end == "no_new_job":  # end of new job sequence
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True

            if not done:

                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

                    if new_job.len > 0:  # a new job comes

                        to_backlog = True
                        for i in range(self.pa.num_nw):
                            if self.job_slot.slot[i] is None:  # put in new visible job slots
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_job_comes()

            reward = self.get_reward()

            # add new jobs
            self.seq_idx += 1

        elif status == 'Allocate':
            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
            self.job_slot.slot[a] = None

            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()

        info = self.job_record

        if done:
            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()
        
        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)


class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot

        self.running_job = []

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, job, curr_time):

        allocated = False

        for t in range(0, self.time_horizon - job.len):

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_nw = 5
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print("New job is backlogged.")

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print("- Backlog test passed -")


def test_compact_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


def test_image_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


if __name__ == '__main__':
    test_backlog()
    test_compact_speed()
    test_image_speed()