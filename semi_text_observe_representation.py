"""
Semi-text observation representation combining feature extraction and SentenceTransformer
This hybrid approach uses:
- Numerical features (feature extraction) for immediate state indicators
- Text embeddings (SentenceTransformer) for high-level semantic information
"""
import numpy as np
from sentence_transformers import SentenceTransformer


class SemiTextObservationEncoder:
    """
    Hybrid encoder that combines numerical features and text embeddings.
    
    Components breakdown:
    - Numerical features: Resource utilization, job queue status (fast, deterministic)
    - Text embeddings: Job descriptions, backlog status, temporal context (semantic understanding)
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
        print(f"Initialized SemiTextObservationEncoder")
        print(f"Model: {model_name}")
        print(f"Text embedding dimension: {self.embedding_dim}")
    
    def encode_state(self, env):
        """
        Convert environment state to hybrid representation
        
        Returns:
            numpy array combining numerical features and text embeddings
        """
        features = []
        
        # ============ PART 1: NUMERICAL FEATURES (Feature Extraction) ============
        # These are fast, deterministic features for immediate state understanding
        
        # 1.1 Resource Features (deterministic numerical)
        for i in range(env.pa.num_res):
            total_capacity = env.pa.res_slot * env.pa.time_horizon
            avbl_slots = np.sum(env.machine.avbl_slot[:, i])
            used_slots = total_capacity - avbl_slots
            
            # Resource availability ratio
            features.append(avbl_slots / float(total_capacity))
            
            # Resource utilization ratio
            features.append(used_slots / float(total_capacity))
            
            # Number of jobs using this resource
            num_jobs_on_res = sum(1 for job in env.machine.running_job if job.res_vec[i] > 0)
            features.append(num_jobs_on_res / float(env.pa.job_num_cap))
        
        # 1.2 Job Slot Features (deterministic numerical)
        for j in range(env.pa.num_nw):
            job = env.job_slot.slot[j]
            
            if job is None:
                # Empty slot features
                for _ in range(env.pa.num_res + 3):
                    features.append(0.0)
            else:
                # Resource vector normalized
                for i in range(env.pa.num_res):
                    features.append(job.res_vec[i] / float(env.pa.max_job_size))
                
                # Job length normalized
                features.append(job.len / float(env.pa.max_job_len))
                
                # Total resource demand normalized
                total_demand = np.sum(job.res_vec)
                features.append(total_demand / float(env.pa.max_job_size * env.pa.num_res))
                
                # Waiting time normalized
                wait_time = env.curr_time - job.enter_time
                features.append(wait_time / float(env.pa.max_job_len))
        
        # 1.3 Backlog Status (deterministic numerical)
        features.append(env.job_backlog.curr_size / float(env.pa.backlog_size))
        
        # 1.4 Running Jobs Status (deterministic numerical)
        features.append(len(env.machine.running_job) / float(env.pa.job_num_cap))
        
        # 1.5 Temporal Features (deterministic numerical)
        features.append(env.extra_info.time_since_last_new_job / 
                       float(env.extra_info.max_tracking_time_since_last_job))
        features.append(env.seq_idx / float(env.pa.simu_len))
        
        # Convert numerical features to array
        numerical_features = np.array(features, dtype=np.float32)
        
        # ============ PART 2: TEXT EMBEDDINGS (SentenceTransformer) ============
        # These provide semantic understanding without cache overhead
        
        text_embeddings = []
        
        # 2.1 Backlog description (semantic summary)
        backlog_desc = self._describe_backlog(env)
        backlog_emb = self.model.encode([backlog_desc])[0]
        text_embeddings.append(backlog_emb)
        
        # 2.2 Running jobs description (semantic summary)
        running_desc = self._describe_running_jobs(env)
        running_emb = self.model.encode([running_desc])[0]
        text_embeddings.append(running_emb)
        
        # 2.3 Job slot descriptions (one per slot)
        job_descs = []
        for i in range(env.pa.num_nw):
            job_desc = self._describe_job_slot(env, i)
            job_descs.append(job_desc)
        
        if job_descs:
            job_embs = self.model.encode(job_descs)
            for job_emb in job_embs:
                text_embeddings.append(job_emb)
        
        # 2.4 Temporal information description
        temporal_desc = self._describe_temporal_info(env)
        temporal_emb = self.model.encode([temporal_desc])[0]
        text_embeddings.append(temporal_emb)
        
        # Convert text embeddings to array and flatten
        text_features = np.concatenate(text_embeddings)
        
        # ============ COMBINE NUMERICAL + TEXT FEATURES ============
        state = np.concatenate([numerical_features, text_features])
        
        return state[np.newaxis, :]  # Add batch dimension
    
    def _describe_job_slot(self, env, slot_idx):
        """Describe a job in the job slot"""
        job = env.job_slot.slot[slot_idx]
        
        if job is None:
            return f"Job slot {slot_idx} is empty and available."
        
        # Describe resource requirements
        res_reqs = []
        for i in range(env.pa.num_res):
            if job.res_vec[i] > 0:
                res_reqs.append(f"{job.res_vec[i]} units of resource {i}")
        
        res_desc = ", ".join(res_reqs) if res_reqs else "minimal resources"
        
        # Check if job can be scheduled
        can_schedule = self._can_schedule_job(env, job)
        schedule_status = "schedulable" if can_schedule else "not yet schedulable"
        
        wait_time = env.curr_time - job.enter_time
        
        desc = (
            f"Job {job.id} requires {res_desc}, "
            f"duration {job.len} units, "
            f"waited {wait_time} units, {schedule_status}."
        )
        
        return desc
    
    def _describe_backlog(self, env):
        """Describe the backlog queue"""
        if env.job_backlog.curr_size == 0:
            return "Backlog queue is empty."
        
        capacity_pct = (env.job_backlog.curr_size / env.pa.backlog_size) * 100
        
        if capacity_pct < 30:
            pressure = "low pressure"
        elif capacity_pct < 70:
            pressure = "moderate pressure"
        else:
            pressure = "high pressure"
        
        desc = (
            f"Backlog has {env.job_backlog.curr_size} jobs "
            f"({capacity_pct:.0f}% full), indicating {pressure}."
        )
        
        return desc
    
    def _describe_running_jobs(self, env):
        """Describe currently running jobs"""
        num_running = len(env.machine.running_job)
        
        if num_running == 0:
            return "No jobs are currently running."
        
        # Calculate average remaining time
        if num_running > 0:
            total_remaining = sum(job.finish_time - env.curr_time 
                                 for job in env.machine.running_job)
            avg_remaining = total_remaining / num_running
        else:
            avg_remaining = 0
        
        desc = (
            f"{num_running} jobs running with average "
            f"{avg_remaining:.1f} time units remaining."
        )
        
        return desc
    
    def _describe_temporal_info(self, env):
        """Describe temporal information"""
        time_since_new = env.extra_info.time_since_last_new_job
        max_track = env.extra_info.max_tracking_time_since_last_job
        
        if time_since_new == 0:
            recency = "just arrived"
        elif time_since_new < max_track * 0.33:
            recency = "recently arrived"
        elif time_since_new < max_track * 0.66:
            recency = "moderate wait"
        else:
            recency = "long idle period"
        
        progress_pct = (env.seq_idx / env.pa.simu_len) * 100
        
        desc = (
            f"Last job {recency}, simulation {progress_pct:.0f}% complete."
        )
        
        return desc
    
    def _can_schedule_job(self, env, job):
        """Check if a job can be scheduled immediately"""
        for t in range(0, env.pa.time_horizon - job.len):
            new_avbl_res = env.machine.avbl_slot[t: t + job.len, :] - job.res_vec
            if np.all(new_avbl_res[:] >= 0):
                return True
        return False


def compute_semi_text_feature_dim(encoder, num_nw, num_res, max_job_size, max_job_len, 
                                   backlog_size, job_num_cap, max_track_since_new, simu_len):
    """
    Compute dimension for semi-text representation
    
    Breakdown:
    NUMERICAL FEATURES:
    - Resource features: num_res * 3 (availability, utilization, num_jobs)
    - Job slot features: num_nw * (num_res + 3) (res_vec, len, demand, wait_time)
    - Backlog features: 1 (size)
    - Running features: 1 (num_running)
    - Temporal features: 2 (time_since_last, sim_progress)
    
    TEXT EMBEDDINGS:
    - Backlog description: 1 embedding (384D)
    - Running jobs description: 1 embedding (384D)
    - Job slot descriptions: num_nw embeddings (num_nw * 384D)
    - Temporal description: 1 embedding (384D)
    
    Total text embeddings: (num_nw + 3) * 384D
    """
    # Numerical features
    numerical_dims = (num_res * 3 +                    # resource features
                      num_nw * (num_res + 3) +         # job slot features
                      1 +                               # backlog size
                      1 +                               # running jobs count
                      2)                                # temporal features
    
    # Text embeddings
    num_text_parts = 2 + num_nw + 1  # backlog + running + job_slots + temporal
    text_dims = num_text_parts * encoder.embedding_dim
    
    total_dim = numerical_dims + text_dims
    
    print(f"\nSemi-Text representation breakdown:")
    print(f"\nNumerical Features: {numerical_dims}D")
    print(f"  - Resource features: {num_res} × 3 = {num_res * 3}D")
    print(f"  - Job slot features: {num_nw} × ({num_res} + 3) = {num_nw * (num_res + 3)}D")
    print(f"  - Backlog size: 1D")
    print(f"  - Running jobs count: 1D")
    print(f"  - Temporal features: 2D")
    
    print(f"\nText Embeddings: {text_dims}D")
    print(f"  - Backlog description: {encoder.embedding_dim}D")
    print(f"  - Running jobs description: {encoder.embedding_dim}D")
    print(f"  - Job slot descriptions: {num_nw} × {encoder.embedding_dim}D")
    print(f"  - Temporal description: {encoder.embedding_dim}D")
    print(f"  - Total text parts: {num_text_parts}")
    
    print(f"\nTotal dimension: {total_dim}D ({numerical_dims}D + {text_dims}D)")
    
    return total_dim, numerical_dims, text_dims
